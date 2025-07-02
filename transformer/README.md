# Transformer

This folder provides scripts and requirements for training and fine-tuning Qwen3 models.

## Contents

- `data_processing.py`: Loads, shuffles, and splits datasets into training and validation sets, saving them as JSONL files.
- `train.py`: Script for model training (supports SFT and LoRA, see below for configuration).
- `requirements.txt`: Lists Python dependencies for data processing and model training.
- `dataset.json`: Data is downloaded from [modelscope](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data).

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

This will generate `train.jsonl` and `val.jsonl` files, and formatted train and validation files `train_format.jsonl` and `val_format.jsonl`.

### 4. Train the Model

Edit `train.py` with your training logic if needed, then run:

```sh
python train.py [OPTIONS]
```

#### Change Model Method

To change the model used for training, set the `--model_name_or_path` argument.  
For example:
```sh
python train.py --model_name_or_path Qwen/Qwen-1.5-7B-Chat
```
You can specify any ModelScope model path.

#### Set Training Type (SFT or LoRA)

Set the training type using the `--type` argument:
- For standard supervised fine-tuning (SFT):  
  `--type sft`
- For LoRA training:  
  `--type lora`

Example:
```sh
python train.py --type lora
```

## Configuration Arguments

All options can be set via command line. Here are the main arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--type` | sft | Training type: `sft`, `lora`, or `ppo` |
| `--dataset_path` | ./dataset.json | Path to the dataset file |
| `--model_name_or_path` | Qwen/Qwen-1.5-7B-Chat | Model name or path |
| `--per_device_train_batch_size` | 1 | Batch size per device for training |
| `--per_device_eval_batch_size` | 1 | Batch size per device for evaluation |
| `--gradient_accumulation_steps` | 4 | Number of gradient accumulation steps |
| `--eval_strategy` | steps | Evaluation strategy |
| `--eval_steps` | 100 | Number of steps between evaluations |
| `--logging_steps` | 10 | Number of steps between logging |
| `--num_train_epochs` | 2 | Number of training epochs |
| `--save_steps` | 400 | Number of steps between checkpoints |
| `--learning_rate` | 1e-4 | Learning rate |
| `--save_on_each_node` | True | Save on each node |
| `--gradient_checkpointing` | True | Enable gradient checkpointing |
| `--output_dir` | ./output/Qwen3-1.7B | Directory to save outputs |
| `--report_to` | swanlab | Reporting tool |
| `--run_name` | qwen3-1.7B | Run name |
| `--task_type` | CAUSAL_LM | Task type, e.g., CAUSAL_LM |
| `--target_modules` | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj | Target modules for LoRA |
| `--inference_mode` | False | Enable inference mode |
| `--r` | 8 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.1 | LoRA dropout |

You can see all options and their current values by running:
```sh
python train.py --help
```

## Training Logs

Use `swanlab` to save training logs.

## Notes

- Scripts utilize [modelscope](https://modelscope.cn/) for dataset handling.
- Adjust data paths and parameters as needed for your use case.
- Ensure your dataset format matches the expected input for processing.

## TODO 🚧
- [ ] Implement other training methods
   - [X] Implement training with LoRA
   - [ ] Implement training with PPO
- [ ] Implement inference demo