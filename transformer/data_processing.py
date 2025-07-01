import os
from modelscope.msdatasets import MsDataset
import json
import random

# Set random seed for reproducibility
random.seed(42)

# Load the dataset
ds = MsDataset.load('dataset.json', subset_name='default', split='train')

# Transform the dataset to a list of dictionaries
# Set split ratio: 90% for training, 10% for validation
data_list = list(ds)
random.shuffle(data_list)
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# Save the training and validation sets to JSONL files
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')


def dataset_jsonl_transfer(origin_path, new_path):
    """
    Transform the dataset from the original JSONL format to a new format.
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            instruction = data["instruction"]
            input = data["question"]
            output = f"<think>{data['think']}</think> \n {data['answer']}"
            message = {
                "instruction": f"{instruction}",
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)



## Print the sizes of the datasets
print(f"Complete split! Dataset size: {len(data_list)}")
print(f"Training dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")