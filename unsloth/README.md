# Unsloth Fine-tuning

This folder contains scripts for fine-tuning Qwen3 and other LLMs using the [Unsloth](https://github.com/unslothai/unsloth) library, with support for reasoning and chat datasets, LoRA adapters, and efficient training.

## Features

- Fine-tune Qwen3-0.6B and similar models with Unsloth.
- Supports LoRA adapters for parameter-efficient training.
- Combines reasoning (CoT) and non-reasoning (chat) datasets.
- Example inference utilities for both "thinking" and "non-thinking" modes.
- Easy artifact saving and (optional) HuggingFace Hub push.

## Usage

### 1. Install Dependencies

Install the required packages (see comments at the top of `train.py`):

```sh
pip install -r requirements.txt
```
Or manually install as needed:
```sh
pip install bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install unsloth
```

### 2. Prepare Model and Data

- Download or place your base model (e.g., Qwen3-0.6B) in the `models/` directory.
- The script uses the following datasets by default:
  - [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) (reasoning)
  - [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) (non-reasoning/chat)

### 3. Run Training

```sh
python train.py
```

- Training configuration is set in `train.py` via the `SFTConfig` object.
- LoRA adapters and tokenizer will be saved to the `OUTPUT_DIR` specified in the script.

### 4. Inference

After training, you can use the built-in `inference` function in `train.py` to test the model:

```python
inference("Your question here", enable_thinking=True)
```

### 5. Save and Share

- Artifacts are saved locally in the output directory.
- (Optional) Push your model and tokenizer to the HuggingFace Hub by uncommenting and editing the relevant lines at the end of `train.py`.

## File Overview

- `train.py`: Main script for fine-tuning, inference, and saving artifacts.
- `requirements.txt`: (Optional) List of dependencies for quick setup.

## Notes

- Adjust model paths, dataset splits, and hyperparameters in `train.py` as needed.
- For more details on Unsloth and advanced usage, see the [Unsloth documentation](https://github.com/unslothai/unsloth).

---
```# Unsloth Fine-tuning

This folder contains scripts for fine-tuning Qwen3 and other LLMs using the [Unsloth](https://github.com/unslothai/unsloth) library, with support for reasoning and chat datasets, LoRA adapters, and efficient training.

## Features

- Fine-tune Qwen3-0.6B and similar models with Unsloth.
- Supports LoRA adapters for parameter-efficient training.
- Combines reasoning (CoT) and non-reasoning (chat) datasets.
- Example inference utilities for both "thinking" and "non-thinking" modes.
- Easy artifact saving and (optional) HuggingFace Hub push.

## Usage

### 1. Install Dependencies

Install the required packages (see comments at the top of `train.py`):

```sh
pip install -r requirements.txt
```
Or manually install as needed:
```sh
pip install bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install unsloth
```

### 2. Prepare Model and Data

- Download or place your base model (e.g., Qwen3-0.6B) in the `models/` directory.
- The script uses the following datasets by default:
  - [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) (reasoning)
  - [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) (non-reasoning/chat)

### 3. Run Training

```sh
python train.py
```

- Training configuration is set in `train.py` via the `SFTConfig` object.
- LoRA adapters and tokenizer will be saved to the `OUTPUT_DIR` specified in the script.

### 4. Inference

After training, you can use the built-in `inference` function in `train.py` to test the model:

```python
inference("Your question here", enable_thinking=True)
```

### 5. Save and Share

- Artifacts are saved locally in the output directory.
- (Optional) Push your model and tokenizer to the HuggingFace Hub by uncommenting and editing the relevant lines at the end of `train.py`.

## File Overview

- `train.py`: Main script for fine-tuning, inference, and saving artifacts.
- `requirements.txt`: (Optional) List of dependencies for quick setup.

## Notes

- Adjust model paths, dataset splits, and hyperparameters in `train.py` as needed.
- For more details on Unsloth and advanced usage, see the [Unsloth documentation](https://github.com/unslothai/unsloth).

---