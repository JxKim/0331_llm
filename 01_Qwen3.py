import torch
import torch.nn as nn
from pathlib import Path
import re
from transformers.models.qwen3 import Qwen3ForCausalLM


# Qwen3-0.6B模型结构配置
QWEN_CONFIG_06_B = {
    "vocab_size": 151936,                  # 词表大小
    "max_position_embeddings": 40_960,      # 最大上下文长度
    "hidden_size": 1024,                    # 模型隐藏向量维度
    "intermediate_size": 3072,              # FFN中间层维度
    "hidden_act": "silu",                   # FFN中的激活函数，Qwen3使用SwiGLU结构
    "num_hidden_layers": 28,                # Transformer Block层数
    "num_attention_heads": 16,              # Query注意力头数
    "num_key_value_heads": 8,               # Key/Value注意力头数，GQA会让多个Q头共享KV头
    "head_dim": 128,                        # 每个注意力头的维度
    "qk_norm": True,                        # 是否需要对Query和Key进行归一化
    "rope_theta": 1_000_000.0,              # RoPE中的theta base
    "rms_norm_eps": 1e-6,                   # RMSNorm中的epsilon
    "tie_word_embeddings": True,            # 输入embedding和输出lm head是否共享权重
    "torch_dtype": torch.bfloat16,          # 低精度dtype，用于降低显存占用
}
# test 更新

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 输入层：token_embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], dtype=cfg["torch_dtype"])

        # transformer block
        # nn.ModuleList：model.parameters()会自动注册其中子模块的参数
        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])]
        )
        self.final_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False, dtype=cfg["torch_dtype"])

        if cfg["tie_word_embeddings"]:
            # 权重共享
            self.out_head.weight = self.tok_emb.weight

        # 没有传入head_dim时，默认使用hidden_size // num_attention_heads作为head_dim
        if cfg["head_dim"] is None:
            head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        else:
            head_dim = cfg["head_dim"]
        # 获取到每个位置的每组向量的旋转的余弦值和正弦值
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_theta"],
            context_length=cfg["max_position_embeddings"]
        )
        # nn.Module所提供的方法，将数值注册到缓存中，
        # 使用方式：self.cos，self.sin
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 追踪KV Cache 当前位置的索引

    def forward(self, in_idx, cache=None):

        # in_idx：shape:[batch_size, num_tokens]
        # 前向传播

        # 输入层：获取到token_embedding
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds # shape: [batch_size, num_tokens, hidden_size / embed_dim]

        num_tokens = x.shape[1]
        if cache is not None and len(cache) > 0 : # decode阶段
            # 每一次decode，token在序列中的位置，从self.current_pos取
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens # decode时，num_tokens= 1
            # 更新self.current_pos，为下一次decode做准备
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
                # 例如pos_start=20,pos_end=21时，只取第20个位置这一行，并允许它关注0到20号位置
            )[pos_start:pos_end, :pos_end]
        else: # prefill阶段
            pos_start = 0  
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
            self.current_pos=num_tokens
        
        # 扩展mask到[1, 1, query_len, key_value_len]，方便广播到每个batch和每个head
        mask = mask.unsqueeze(0).unsqueeze(0)

        # i: 第i层
        for i, block in enumerate(self.trf_blocks):
            # prefill阶段，cache传进来的是空字典{}；decode阶段，cache里已经有每一层的KV Cache
            blk_cache = cache.get(i) if cache is not None else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                # 对字典中每一层赋值K和V的Cache
                cache[i] = new_blk_cache

        # 输出前先进行层归一化
        x = self.final_norm(x)
        # 输出层
        # logits.shape: [batch_size, num_tokens, vocab_size]
        logits = self.out_head(x.to(self.cfg["torch_dtype"]))
        # 对logits做softmax，得到概率分布，基于概率分布，依概率进行采样，得到下一个token
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # GQA注意力层
        self.att = GroupedQueryAttention(
            d_in=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["num_key_value_heads"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["torch_dtype"],
            rms_norm_eps=cfg["rms_norm_eps"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norm2 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # x.shape: [batch_size, seq_len/num_tokens , hidden_size]
        # 保存输入，用于后面进行残差连接（GQA的残差连接）
        shortcut = x
        x = self.norm1(x)
        # attention
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, hidden_size]
        x = x + shortcut  # 进行残差连接

        # 保存输入，用于后面进行残差连接（FFN的残差连接）
        shortcut = x
        x = self.norm2(x)
        # FFN
        x = self.ff(x)
        x = x + shortcut  # 进行残差连接

        return x, next_cache

class FeedForward(nn.Module):
    # 初始化
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=cfg["torch_dtype"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=cfg["torch_dtype"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], dtype=cfg["torch_dtype"], bias=False)
        assert cfg["hidden_act"] == "silu", "This teaching implementation only supports silu for SwiGLU"

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        dtype=None, rms_norm_eps=1e-6
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # q_proj层参数为全量的：d_in输入，d_out输出
        # 一个token向量，d_in=1024，经过q_proj运算之后，得到的向量维度：d_out
        # Qwen3-0.6B中：d_out = num_attention_heads * head_dim = 16 * 128 = 2048
        # 此时可以将2048维向量拆成16个Q头，每个Q头128维
        self.q_proj = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # k_proj和v_proj的输出头数更少，d_in输入，num_kv_groups * head_dim输出
        # Qwen3-0.6B中：num_key_value_heads * head_dim = 8 * 128 = 1024
        # 这时可以将1024维向量拆成8个K头/V头，每个头128维
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影层：融合多头信息，将维度从2048，重新输出成1024维
        self.o_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # x.shape: [batch_size, num_tokens, hidden_size]
        b, num_tokens, _ = x.shape

        # 获取Q, K, V
        queries = self.q_proj(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.k_proj(x)     # (b, num_tokens, num_kv_groups * head_dim)
        values = self.v_proj(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Queries分成num_heads个头，每个头的维度为head_dim
        # 新queries shape: (b, self.num_heads,num_tokens , self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Keys和Values分成num_kv_groups个组，每个组的维度为head_dim
        # # 新keys/values  shape: (b, self.num_kv_groups,num_tokens , self.head_dim)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 额外的归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # 重点：对queries和keys应用旋转位置编码
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            # decode阶段，cache不为空，先拿到cache中，历史所有token的k和v cache
            prev_k, prev_v = cache
            # 把当前decode阶段单个token算出来的新的key和value和前面token的k和v拼起来
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            # prefill阶段，key,value直接取token和k_proj参数矩阵以及token和v_proj参数矩阵，实际算得的k,v向量
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # 复制Keys和Values，使每组内的多个Query头共用同一组Keys和Values
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        # query, keys, values, shape: batch_size, num_heads, num_tokens, head_dim

        # 注意力得分计算
        # keys: [batch_size, num_heads, num_tokens , head_dim ]
        # keys。transpose: [batch_size, num_heads, head_dim , num_tokens ]
        attn_scores = queries @ keys.transpose(2, 3)
        # 需要对score做因果掩码，使得每个token只能关注自己和前面的token
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        # attn_weights: [batch_size, num_heads, query_len/num_tokens, query_len/num_tokens]
        # values: [batch_size, num_heads, query_len/num_tokens, head_dim]
        # (attn_weights @ values)结果：[batch_size, num_heads, query_len, head_dim]，每一个头，每一个token，融合了其他token的v向量的最终结果
        # context转置后shape: [batch_size, query_len, num_heads, head_dim]
        # reshape: 去掉num_heads: d_out: num_heads* head_dim
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        # o_proj降维：转换成(b, num_tokens, d_in(hidden_size))
        # o_proj： 1、将多个头 attention之后的结果，融合到一起， 2、将d_out的向量（num_heads * head_dim）转换回 hidden_size
        return self.o_proj(context), next_cache

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    计算每个位置的余弦值和正弦值，用于旋转位置编码
    head_dim假设等于8：
    [0,1,2,3,4,5,6,7]
    常见二维旋转讲解里会写成相邻配对：[0,1],[2,3],[4,5],[6,7]
    我们这里的实现：[0,4],[1,5],[2,6],[3,7]
    因此，此处计算出，每个序列位置的token的每个维度处，cos和sin的值，后面直接使用该值进行计算
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 1. 生成频率下标：0, 2, 4, ..., head_dim-2，总数为head_dim//2
    # torch.arange: 生成一维张量，[)
    # 2i的所有可能取值
    freq_indices = torch.arange(0, head_dim, 2, dtype=dtype)

    # 2. 转成浮点，并除以 head_dim，得到指数
    # 2i/d
    exponents = freq_indices.float() / head_dim

    # 3. 计算 theta_base 的这些指数次幂
    # 1000000 ^ (2i/d) 
    scales = theta_base ** exponents

    # 4. 取倒数，得到 inverse frequencies
    # shape : (head_dim / 2 , )
    # 所有的θ_i的可能取值
    inv_freq = 1.0 / scales


    # 计算位置索引：[0,1,2,3,...,context_length-1] shape: (context_length,)
    # positions代表的就是 token在序列当中所有可能的位置
    positions = torch.arange(context_length, dtype=dtype)

    # 计算每个位置的每组旋转的角度
    # positions: [0,1,2,3.... context_length]
    # positions.unsqueeze(1): (context_length,1) [[0],[1],[2],.... [context_length -1]]
    # 广播后：（context_length, head_dim/2）
    # [0, 0, 0]
    # [1, 1, 1]
    # [2, 2, 2] 


    # inv_feq:[100_0000 ^ (0),100_0000 ^ (-2/128),100_0000 ^ (-4/128),,100_0000 ^ (-6/128),100_0000 ^ (-126/128)]
    # inv_freq.unsqueeze(0): (1,head_dim // 2) [[100_0000 ** (-0), 100_0000 ** (-2/head_dim),100_0000 ** (-4/head_dim), ... ]]
    # 广播后：(context_length, head_dim / 2)
    # [theta_0, theta_1, theta_2]
    # [theta_0, theta_1, theta_2]
    # [theta_0, theta_1, theta_2]

    # 哈达玛积运算
    # [0* theata_0, 0* theta_1, 0* theta_2]
    # [1*theta_0, 1*theta_1, 1*theta_2]
    # [2*theta_0, 2*theta_1, 2*theta_2]

    # positions.unsqueeze(1) * inv_freq_unsqueeze(0) 运算时：
    #   1、前者的列会复制成 head_dim /2 列，后者的行，会复制成 context_length行
    #   2、做哈达玛积运算

    # angles: (context_length, head_dim // 2)
    # angles[0,0]=序列中第0个位置处，第0组分量的旋转角度，对应的就是m * theta_i

    # [0* theata_0, 0* theta_1, 0* theta_2]
    # [1*theta_0, 1*theta_1, 1*theta_2]
    # [2*theta_0, 2*theta_1, 2*theta_2]
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # 由于是前半部分和后半部分对应索引位置处做旋转，所以此处将angels复制，从而使得旋转的一组二维分量，共享一个angles值
    # head_dim=8时，一行会变成：[angle_0, angle_1, angle_2, angle_3, angle_0, angle_1, angle_2, angle_3]

    # [0* theata_0, 0* theta_1, 0* theta_2,  0* theata_0, 0* theta_1, 0* theta_2]
    # [1*theta_0, 1*theta_1, 1*theta_2,     1*theta_0, 1*theta_1, 1*theta_2]
    # [2*theta_0, 2*theta_1, 2*theta_2,     2*theta_0, 2*theta_1, 2*theta_2]
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
    
    

    # 预计算每个角度的余弦值和正弦值
    cos = torch.cos(angles) # 得到序列当中每个位置，每个分量的余弦值
    sin = torch.sin(angles) # 得到序列当中每个位置，每个分量的正弦值

    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    """
    使用计算好的cos和sin来计算旋转位置编码之后的x，注意，此处获取分组向量不是使用紧挨着的两个值构造而成的，
    如果 head_dim = 8，此处的配对关系就是：
                    (0, 4)
                    (1, 5)
                    (2, 6)
                    (3, 7)
    而不是常见的：
                    (0, 1)
                    (2, 3)
                    (4, 5)
                    (6, 7)
    Args:
        x: 需要应用旋转位置编码的张量，shape为[batch_size, num_heads, seq_len, head_dim]
        cos: 预计算好的余弦值张量，shape为[max_position_embeddings, head_dim]
        sin: 预计算好的正弦值张量，shape为[max_position_embeddings, head_dim]
        offset: 当前这段token在完整序列中的起始位置，需要依赖该变量，来从预先计算好的cos和sin张量中取值出来
    Returns:
        应用RoPE后的张量，shape与x相同

    具体例子：
        假设某个token在某个head上的向量为：
        假设head_dim = 8 
            x = [x0, x1, x2, x3, x4, x5, x6, x7]
        当前position对应的角度为：
            angles = [a0, a1, a2, a3, a0, a1, a2, a3]
        从cos和sin中取出来的值为
            pos_cos = [cos(a0), cos(a1), cos(a2), cos(a3), cos(a0), cos(a1), cos(a2), cos(a3)]
            pos_sin = [sin(a0), sin(a1), sin(a2), sin(a3), sin(a0), sin(a1), sin(a2), sin(a3)]
        rotate_half(x)会先把x拆成前后两半：
            first_half = [x0, x1, x2, x3]
            second_half = [x4, x5, x6, x7]
        再拼成：
            rotate_half(x) =[-second_half,first_half]
            也即：
            rotate_half(x) = [-x4, -x5, -x6, -x7, x0, x1, x2, x3]
        最终计算：
            x_rotated = x * pos_cos + rotate_half(x) * pos_sin
        也即：
            x_rotated = [first_half, second_half] * pos_cos + [-second_half, first_half] * pos_sin
        由于first_half和second_half，也都是向量，所以上面的式子，可以拆解成对应位置处的处理。
                      
        现从first_half和second_half当中，取第0个索引位置处的值，x0和x4，计算最终结果：
            [x0*cos(a0) - x4 * sin(a0), x4*cos(a0) + x0*sin(a0)]，
        展开后就是：
            x0' = x0*cos(a0) - x4*sin(a0)
            x4' = x4*cos(a0) + x0*sin(a0)

        等价于：
            [x0, x4] * 旋转矩阵
        
        其他分量同理。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    if offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"RoPE position out of range: offset={offset}, seq_len={seq_len}, "
            f"max_position_embeddings={cos.shape[0]}"
        )

    # 将x切分为前一半和后面一半
    # x.shape : [batch_size, num_heads, seq_len, head_dim]
    x1 = x[..., : head_dim // 2]  # 前面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)
    x2 = x[..., head_dim // 2:]  # 后面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)

    # cos, sin : shape: [context_length,head_dim]
    # 调整sin和cos的形状：当前这段 token 取出对应位置的 cos 和 sin，并调整 shape 以便广播
    # offset: 10 seq_len: 2: [10,11]
    # offset: 12 seq_len:3 [12,13,14]
    # offset=0时，表示从序列开头开始取
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)

    # 应用旋转变换：rotate_half(x) = [-后半部分, 前半部分]
    rotated = torch.cat((-x2, x1), dim=-1) # Shape: (batch_size, num_heads,seq_len, head_dim)
    # 二维旋转公式：原来的向量[a,b] ,旋转后的向量[a', b'] = [a cos - b sin, a sin + b cos]
    # 写成向量形式，就是：[a, b] * cos + [-b, a] * sin=[acos-bsin,bcos+asin]
    # 以(x0, x4)举例，旋转后会变成：
    # x0' = x0 * cos - x4 * sin
    # x4' = x0 * sin + x4 * cos
    # 同理：

    # x1' = x1 * cos - x5 * sin
    # x5' = x1 * sin + x5 * cos
    x_rotated = (x * cos) + (rotated * sin)

    # 返回旋转后的x
    return x_rotated.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        return norm_x.to(input_dtype)

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # 对齐HF行为：chat模型使用<|im_end|>，base模型使用<|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # 不插入<think>标签，只换行
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

def generate_text(input_ids,model:Qwen3Model,tokenizer:Qwen3Tokenizer,max_len:int=100):
    
    # 每次调用时，重置一下current_pos位置的值
    model.reset_kv_cache()
    # 整个自回归生成的过程当中，生成的token数量
    generated_token = 0
    # 维护生成的所有的新的token
    final_output = input_ids.clone()
    # 存放kv cache的一个字典
    kv_cache = {}
    
    with torch.no_grad():
        # 1、prefill阶段
        # output_logits: shape:[batch_size,seq_len,vocab_size]
        # input_ids.shape: batch_size, seq_len 
        # model(input_ids,cache=kv_cache): 走模型内部的forward方法。
        output_logits = model(input_ids,cache=kv_cache)
        
        # 取最后一个token的logits
        logits = output_logits[:,-1,:]
        # 进行softmax操作，获得概率分布
        # probs.shape: batch_size,vocab_size
        probs = torch.softmax(logits,dim=-1)
        # next_token_id.shape: batch_size, 表示每个样本，下一个token是什么
        next_token_id = torch.multinomial(probs,num_samples=1).squeeze(-1)

        # 更新generated_token变量，
        generated_token +=1

        # 2、decode阶段
        
        # next_input.shape: batch_size, 1
        next_input = next_token_id.unsqueeze(-1)

        final_output = torch.cat([final_output,next_input],dim=-1)
        # 自回归过程中的终止条件：1、生成的token数量达到最大值 2、生成了EOS Token id 
        while generated_token<max_len:
            # 当前KV_Cache就不是空字典了，当前这个dict中的键，就是不同的TransformerBlock的层，值就是这一层所对应的KV Cache
            # output_logits: shape: batch_size, seq_len, vocab_size
            output_logits =  model(next_input,kv_cache)
            
            logits = output_logits[:,-1,:]
            probs = torch.softmax(logits,dim=-1)
            next_token_id = torch.multinomial(probs,num_samples=1).squeeze(-1)
            # next_input.shape : batch_size, 1
            next_input = next_token_id.unsqueeze(-1)

            final_output = torch.cat([final_output,next_input],dim=-1)

            generated_token += 1 

    
    res_list = final_output[0].tolist()
    print(res_list)
    
    res = tokenizer.decode(res_list)
    print(res)
    return res

def main():
    """
    测试Qwen3模型的生成流程。
    注意：这里只初始化了模型结构，没有加载预训练权重，所以输出不代表真实Qwen3效果。
    """
    import torch
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=r"model/Qwen3-0.6B/tokenizer.json")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.eval()
    model.to("cuda")
    input_ids = torch.tensor(tokenizer.encode("你好，今天天气真好啊")).unsqueeze(0).to("cuda")
    output = generate_text(input_ids,model,tokenizer)
    
    print(output)

if __name__ == "__main__":
    main()
