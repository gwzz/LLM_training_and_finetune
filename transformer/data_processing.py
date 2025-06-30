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
with open('train_ner.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
with open('val_ner.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

## Print the sizes of the datasets
print(f"Complete split! Dataset size: {len(data_list)}")
print(f"Training dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")