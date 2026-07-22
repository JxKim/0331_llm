from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")


from dataclasses import dataclass

@dataclass
class SFTConfig:
    train_data_size: int  = 10000
    eval_data_size: int = 500
    lr: float = 3e-5 
    batch_size: int = 4
    warmup_ratio: float = 0.1
    save_dir: str = "./finetuned/02_sft_demo"
    log_dir :str = "./logs/02_sft_demo"

    eval_iter:int = 100

    log_iter: int = 100


def get_train_data(sft_config:SFTConfig):
    """
    获取训练的chosen_data和rejected_data
    """

    from datasets import load_dataset
    train_data = load_dataset("./data/ultrafeedback_binarized")["train_prefs"]
    train_data = train_data.shuffle()
    train_data = train_data.select(range(sft_config.train_data_size))
    chosen_result = []
    rejected_result = []
    for i in range(sft_config.train_data_size):
        # 1、对chosen处理
        message_list = train_data[i]["chosen"]
        result:list = tokenizer.apply_chat_template(message_list,tokenize=True)["input_ids"]
        chosen_result.append(result)

        # 2、对rejected处理

        message_list = train_data[i]["rejected"]
        result:list = tokenizer.apply_chat_template(message_list,tokenize=True)["input_ids"]
        rejected_result.append(result)

    return chosen_result, rejected_result

def get_eval_data(sft_config:SFTConfig):

    from datasets import load_dataset
    eval_data = load_dataset("./data/ultrachat_200k")["test_sft"]
    eval_data = eval_data.shuffle()
    eval_data = eval_data.select(range(sft_config.eval_data_size))
    final_result = []
    for i in range(sft_config.eval_data_size):
        message_list = eval_data[i]["messages"]
        result:list = tokenizer.apply_chat_template(message_list,tokenize=True)["input_ids"]
        final_result.append(result)

    return final_result


from transformers import PreTrainedTokenizerFast
from typing import List
import torch
def create_answer_mask(labels,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从labels当中找出assistant回答的部分，然后输出一个与labels相同shape的mask
    """
    # 构建answer mask，输入的labels为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和labels相同shape的全0矩阵
    answer_mask = torch.zeros_like(labels)

    # 2、找到<|im_end|> 所对应的token_id
    eos_token_id = tokenizer.encode("<|im_end|>")[0]

    # 3、遍历labels中的每一个样本
    # labels.shape: batch_size, seq_len
    for idx,ids in enumerate(labels):
        # 3.1、获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()
        # 3.2、解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 3.3、设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 4、结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为10，第二个为15，第三个为20，第四个为25
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取，每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[10,20]
        assistant_ends:[15,25]
    """

    use_ends = [pos for pos in eos_positions[::2]]
    assistant_ends = [pos for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为10，第二个为15，第三个为20，第四个为25
    那么user_turns:[10,20]，assistant_ends:[15,25]

    
    要想获取到assistant的回答的起始位置，就需要跳过<|im_end|>, \n, <|im_start|>,assistant , \n 这5个token
    要想获取到assistant的回答的结束位置，需要将<|im_end|>也包括进去，又因为列表切片是左闭右开的，所以需要向后移动一位
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    # 多轮对话没有被截断或者最后一轮整个assistant回答被截断，user轮数和assistant轮数一致
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1

    # 最后一轮，assistant回答被部分截断，此时user轮数比assistant轮数多一轮
    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 5
        mask[last_answer_start:] = 1


def compute_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs,beta):
    """
    DPO的损失函数：
    chosen_log_probs：shape [batch_size, ]
    rejected_log_prob：shape [batch_size,]
    ref_chosen_log_probs:shape [batch_size,]
    ref_rejected_log_probs: shape [batch_size,]
    beta: 超参数
    """
    margin = chosen_log_probs - rejected_log_probs - (ref_chosen_log_probs - ref_rejected_log_probs)

    # loss: shape: [batch_size, ]
    loss = - torch.nn.functional.logsigmoid( beta * margin)

    return loss.mean() # 求平均

def compute_log_probs(logits, labels, assistant_mask):
    """
    计算模型输出label当中回答的 对数概率
    """

    # 1、获取到log_probs，shape:  [batch_size, num_tokens, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)

    # 2、需要从log_probs里面找到，输出答案token的具体的对数概率
    # 此处需要使用：torch.gather算子
    # shape: batch_size, num_tokens
    label_token_log_prob = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)


    # 3、对 对数概率，进行掩码，让非 assistant answer 部分，置为0，assistant answer，保留原值
    # masked_label_token_log_prob.shape: batch_size, num_tokens 
    masked_label_token_log_prob  = assistant_mask * label_token_log_prob

    # 、对 对数概率做相加，获取模型输出整个回答的对数概率
    log_probs = masked_label_token_log_prob.sum(dim = -1) 

    return log_probs



# cosine_decay(current_batch,total_batch,sft_config.warmup_ratio,sft_config.lr)
import numpy as np
def cosine_decay(current_batch,total_batch,warmup_ratio,lr):

    warmup_batch = total_batch * warmup_ratio

    if current_batch< warmup_batch:
        # y=kx
        k = lr/warmup_batch
        x = current_batch
        return  k * x
    else:
        # progress: 表示衰减过程，从0到1
        progress = (current_batch - warmup_batch)  / ( total_batch -warmup_batch)
        # progress从0到1的过程，cos从最大值，降到最小值
        # cos(π * progress)，从1到-1 
        # （cos(π * progress)+1）*0.5 ，从1到0，表示衰减的程度
        decay_level = (np.cos(np.pi * progress) + 1) * 0.5

        return lr * decay_level



def eval_model(model,sft_config:SFTConfig):

    model.eval()

    eval_data = get_eval_data(sft_config)

    total_batch = (len(eval_data) + sft_config.batch_size - 1)  // sft_config.batch_size
    all_batch_loss = []
    for current_batch in  range(total_batch):

        current_batch_data = eval_data[current_batch*sft_config.batch_size : (current_batch+1) * sft_config.batch_size]

        max_length = max([len(sample) for sample in current_batch_data])

        for sample in current_batch_data:
            padding_length = max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)
        
        data_tensor = torch.tensor(current_batch_data, dtype=torch.long).to("cuda")
        # input_ids:
        input_ids = data_tensor[:,:-1]
        labels = data_tensor[:,1:]
        assistant_mask = create_answer_mask(labels=labels,tokenizer=tokenizer)
        with torch.no_grad():
            logits = model(input_ids).logits

        batch_loss = compute_loss(logits=logits, labels=labels,assistant_mask=assistant_mask )

        all_batch_loss.append(batch_loss.item())
    
    average_loss = sum(all_batch_loss) / len(all_batch_loss)

    return average_loss


from torch.utils.tensorboard.writer import SummaryWriter
import tqdm


def train(sft_config:SFTConfig):
    """
    训练主流程：
    4.2.0 初始化模型，优化器，获取总的训练数据 ✅
    4.2.1 构造模型前向传播的输入，input_ids，labels，对input_ids做padding，构造assistant_answer_mask ✅
    4.2.2 前向传播，获取到logits ✅
    4.2.3 基于logits,assistant_answer_mask,labels，算损失 ✅
    4.2.4 反向传播：计算梯度 ✅
    4.2.5 做一个学习率调度 ✅
    4.2.6 基于新的学习率，做参数更新 ✅
    """
    # 初始化模型
    from transformers import AutoModelForCausalLM
    from torch.optim.adamw import AdamW # 对于大模型微调，一般使用AdamW
    model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo")
    ref_model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo")
    model.to("cuda")
    ref_model.to("cuda")
    model.train()
    ref_model.eval()
    optimizer = AdamW(model.parameters(), lr=sft_config.lr)
    loss_list = []
    
    
    # todo: 构建一个获取数据的方法
    # data: 第一个list，是所有数据，第二个list，是一条数据的message_list
    chosen_data, rejected_data = get_train_data(sft_config)
    total_batch = (len(chosen_data) + sft_config.batch_size - 1)  // sft_config.batch_size

    # 构建tensorsorboard的writer对象，和tqdm对象
    writer = SummaryWriter(log_dir=sft_config.log_dir)
    progress_bar = tqdm.tqdm(total=total_batch)
    for current_batch in range(total_batch):
        #  构造chosen相关的数据
        current_chosen_batch_data = chosen_data[current_batch*sft_config.batch_size : (current_batch+1) * sft_config.batch_size]

        max_chosen_length = max([len(sample) for sample in current_chosen_batch_data])

        for sample in current_chosen_batch_data:
            padding_length = max_chosen_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)
        
        chosen_data_tensor = torch.tensor(current_chosen_batch_data, dtype=torch.long).to("cuda")
        # input_ids:
        chosen_input_ids = chosen_data_tensor[:,:-1]
        chosen_labels = chosen_data_tensor[:,1:]

        # 找到labels当中，哪些token_ids是pad_token_id

        # padding_mask的含义：为1的地方，不是pad_token, 为0的地方，是pad_token
        chosen_padding_mask = torch.where(chosen_labels == tokenizer.pad_token_id, 0, 1)

        chosen_assistant_mask = create_answer_mask(labels=chosen_labels,tokenizer=tokenizer)

        # 将padding_mask和assistant_mask取一个交集，交集之后，为1的地方，才是最终，需要计算损失的地方，交集之后为0的，不需要算损失
        final_chosen_mask = chosen_assistant_mask & chosen_padding_mask


        # 构造rejected相关的数据
        current_rejected_batch_data = rejected_data[current_batch*sft_config.batch_size : (current_batch+1) * sft_config.batch_size]

        max_rejected_length = max([len(sample) for sample in current_rejected_batch_data])

        for sample in current_rejected_batch_data:
            padding_length = max_rejected_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)
        
        rejected_data_tensor = torch.tensor(current_rejected_batch_data, dtype=torch.long).to("cuda")
        # input_ids:
        rejected_input_ids = rejected_data_tensor[:,:-1]
        rejected_labels = rejected_data_tensor[:,1:]

        # 找到labels当中，哪些token_ids是pad_token_id

        # padding_mask的含义：为1的地方，不是pad_token, 为0的地方，是pad_token
        rejected_padding_mask = torch.where(rejected_labels == tokenizer.pad_token_id, 0, 1)

        rejected_assistant_mask = create_answer_mask(labels=rejected_labels,tokenizer=tokenizer)

        # 将padding_mask和assistant_mask取一个交集，交集之后，为1的地方，才是最终，需要计算损失的地方，交集之后为0的，不需要算损失
        final_rejected_mask = rejected_assistant_mask & rejected_padding_mask


        logits = model(input_ids).logits #model前向传播，此处得到的结果是一个对象，需要通过.logits方式，获取到logits属性

        loss = compute_loss(logits=logits,labels=labels,assistant_mask=final_mask)
        loss_list.append(loss.item())
        loss.backward()

        current_lr = cosine_decay(current_batch,total_batch,sft_config.warmup_ratio,sft_config.lr)

        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.step()

        optimizer.zero_grad()


        should_eval = current_batch % sft_config.eval_iter == 0 
        should_log = current_batch % sft_config.log_iter  == 0
        if should_eval:
            average_loss = eval_model(model,sft_config)
            writer.add_scalar("eval/loss",average_loss,current_batch)
            model.train()

        if should_log:
            last_iter_loss = loss_list[-sft_config.log_iter:]
            average_loss = sum(last_iter_loss) / len(last_iter_loss)

            writer.add_scalar("train/loss",average_loss,current_batch)

            writer.add_scalar("train/current_lr",current_lr, current_batch)

        progress_bar.update(1)
        progress_bar.set_postfix(loss = f"{loss.item():.5f}", current_lr = f"{current_lr:.2e}")



    model.save_pretrained(sft_config.save_dir)
    tokenizer.save_pretrained(sft_config.save_dir)
    print("model和tokenizer已经保存")


def main():
    sft_config = SFTConfig(train_data_size=200, batch_size=1)
    train(sft_config=sft_config)


if __name__=="__main__":
    main()


