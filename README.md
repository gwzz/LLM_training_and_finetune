# LLM_training_and_finetune

A collection of scripts and utilities for training and fine-tuning large language models (LLMs) using different frameworks and strategies.

## Project Structure

- **transformer/**  
  Scripts and configs for training and fine-tuning transformer-based models (e.g., Qwen3) with support for SFT, LoRA, and PPO.  
  See `transformer/README.md` for detailed usage and options.

- **unsloth/**  
  Tools and scripts for efficient LLM fine-tuning using the [Unsloth](https://github.com/unslothai/unsloth) library.  
  See `unsloth/README.md` for framework-specific instructions.

- **llama_factory/**  
  Utilities and scripts for training and fine-tuning LLMs using the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) framework.  
  See `llama_factory/README.md` for details and usage.

## Features

- Data preprocessing and formatting for LLM training.
- Support for multiple training strategies: SFT, LoRA, PPO, etc.
- Modular design for easy extension and adaptation to new models or datasets.
- Example configurations and scripts for HuggingFace Transformers, Unsloth, and LLaMA Factory.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone <this-repo-url>
   cd LLM_training_and_finetune
   ```

2. **Choose a framework**  
   Enter the `transformer`, `unsloth`, or `llama_factory` folder and follow the respective README for setup and usage.

3. **Prepare your dataset**  
   Place your dataset in the required format as described in the subfolder README.

4. **Install dependencies**  
   Each folder contains its own `requirements.txt` for environment setup.

## Notes

- This project is intended for research and educational purposes.
- For detailed configuration, training options, and advanced usage, refer to the README in each subfolder.

## License

See `LICENSE` file for details.
