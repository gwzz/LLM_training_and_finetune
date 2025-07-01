# Transformer

This folder provides scripts and requirements for training and fine-tuning Qwen3 models.

## Contents

- `data_processing.py`: Loads, shuffles, and splits datasets into training and validation sets, saving them as JSONL files.
- `train.py`: Script for model training (implement your training logic here).
- `requirements.txt`: Lists Python dependencies for data processing and model training.
- `dataset.json`: data is downloaded from [modelscope](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data).

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your dataset as `dataset.json` in the current directory.

### 3. Process Data

```sh
python data_processing.py
```

This will generate `train.jsonl` and `val.jsonl` files, and formatted train and validation files `train_format.jsonl` and `val_format.jsonl`

### 4. Train the Model

Edit `train.py` with your training logic, then run:

```sh
python train.py
```

## Trainning logs

Use `swanlab` to save trainning logs.

## Notes

- Scripts utilize [modelscope](https://modelscope.cn/) for dataset handling.
- Adjust data paths and parameters as needed for your use case.
- Ensure your dataset format matches the expected input for processing

## TODO :construction:
- [ ] Implement other training mothods 
   - [ ] Implement training with LoRA
   - [ ] Implement training with PPO
- [ ] Implement inference demo