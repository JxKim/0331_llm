"""
自回归生成的脚本：
支持：
从命令行，传入模型参数的路径，和prompt，加载对应的路径模型，和prompt，做自回归生成

1、需要从命令行，读取到模型路径和prompt ✅
2、需要加载模型，tokenizer ✅
3、对prompt，转化成message_list，使用tokenizer.apply_chat_template，转换成模型的输入input_ids ✅
4、调model.generate()，传入input_ids ✅
5、解析第4步的token_ids: 将传入的input_ids这部分截掉，然后将新生成的token_ids，使用tokenizer.decode进行解码。将结果，最终输出 ✅
"""

from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
parser = ArgumentParser()
# 添加参数
parser.add_argument("--model_path",type=str,help="需要加载的模型路径")
parser.add_argument("--prompt",type=str,help="需要模型处理的prompt")
# 解析命令行传过来的参数
args = parser.parse_args()

model_path  = args.model_path
prompt = args.prompt
model = AutoModelForCausalLM.from_pretrained(model_path,device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

message_list = [{"role":"user","content":prompt}]
input_tensor : Tensor= tokenizer.apply_chat_template(message_list, tokenize=True, add_generation_prompt=True,return_tensors = "pt")["input_ids"].to("cuda")

# result.shape: batch_size,num_tokens(包含输入的input_ids和新生成的token_ids)
result: Tensor= model.generate(input_tensor,max_new_tokens = 500)

new_token_ids = result[0][len(input_tensor[0]):]

decoded_result = tokenizer.decode(new_token_ids)

print("新生成的token解码之后的结果为：\n\n",decoded_result)
