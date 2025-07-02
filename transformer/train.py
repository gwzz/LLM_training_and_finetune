import os
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import config

import swanlab

opt = config.get_options()
os.environ["SWANLAB_PROJECT"]="qwen3-sft"
MAX_LENGTH = 2048
MODEL  = opt.model_name_or_path
TRAIN_DATASET_PATH = "train_format.jsonl"
VAL_DATASET_PATH = "val_format.jsonl"


swanlab.config.update({
    "model": MODEL,
    "data_max_length": MAX_LENGTH,
    })





# define the data processing function for Qwen3 template format
def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

# download the model snapshot from ModelScope, put it in the current directory
# if you have already downloaded the model, you can comment out this line, and use the local path directly
model_dir = snapshot_download(MODEL, cache_dir="./", revision="master")

# Transformers load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"./{MODEL}", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(f"./{MODEL}", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

# load the training and validation datasets
train_df = pd.read_json(TRAIN_DATASET_PATH, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

eval_df = pd.read_json(VAL_DATASET_PATH, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

#  training type is lora, setting up the LoRA configuration
if opt.type == 'lora':
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=opt.target_modules,
    inference_mode=opt.inference_mode,
    r=opt.r,
    lora_alpha=opt.lora_alpha,
    lora_dropout=opt.lora_dropout,
    )
    
    model = get_peft_model(model, lora_config)
    print(f"LoRA model loaded with config: {lora_config}")


args = TrainingArguments(
    output_dir=opt.output_dir,
    per_device_train_batch_size=opt.per_device_train_batch_size,
    per_device_eval_batch_size=opt.per_device_eval_batch_size,
    gradient_accumulation_steps=opt.gradient_accumulation_steps,
    eval_strategy=opt.eval_strategy,
    eval_steps=opt.eval_steps,
    logging_steps=opt.logging_steps,
    num_train_epochs=opt.num_train_epochs,
    save_steps=opt.save_steps,
    learning_rate=opt.learning_rate,
    save_on_each_node=opt.save_on_each_node,
    gradient_checkpointing=opt.gradient_checkpointing,
    report_to=opt.report_to,
    run_name=opt.run_name,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# define the inference function
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# Take a few samples from the test dataset and predict
test_df = pd.read_json(VAL_DATASET_PATH, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()