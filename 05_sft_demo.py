from datasets import load_dataset, DatasetDict
## 1.1 数据加载
data:DatasetDict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})
data["train"] = data["train"].select(range(10000))

## 1.2 数据处理
from typing import Dict,List
def convert_type(examples:Dict[str, List]):
    """
    讲数据，转换成 SFTTrainer所需要的Language Modeling类型，对话格式
    """
    conversation_list:List[List[Dict]] = examples["conversation"]

    all_data_messages_list = []

    for data in conversation_list:
        human_message = data[0]["human"]
        assistant_message = data[0]["assistant"]

        message_list = [
            {"role":"user","content":human_message},
            {"role":"assistant","content":assistant_message}
        ]

        all_data_messages_list.append(message_list)

    return {"messages":all_data_messages_list}


# batched=True，传递给convert_type的是一批数据，
mapped_data = data.map(convert_type,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])



from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"]="./logs/05_sft_demo"

config = SFTConfig(
    per_device_train_batch_size=4,
    per_device_eval_batch_size= 4,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    output_dir="./finetuned/05_sft_demo",
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length= 650,
    assistant_only_loss=True,
    chat_template_path="./new_chat_template.jinja"
)



from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B/")
trainer = SFTTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    train_dataset=mapped_data["train"],
    eval_dataset=mapped_data["test"],
)


trainer.train()
# 保存模型参数，和Tokenizer相关的配置，从而使得，后面，可以通过加载这个路径，得到model和tokenizer
trainer.save_model("./finetuned/05_sft_demo")