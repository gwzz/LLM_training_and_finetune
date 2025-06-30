# Transformer

This folder contains scripts and requirements for training and fine-tuning transformer-based models for NLP tasks.

## Contents

- `data_processing.py`: Script to load, shuffle, and split datasets into training and validation sets, saving them as JSONL files.
- `train.py`: Placeholder for model training script.
- `requirements.txt`: Python dependencies required for data processing and model training.

## Usage

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   
   Ensure your dataset is available as `dataset.json` in the current directory.

3. **Process data:**

   ```sh
   python data_processing.py
   ```

   This will create `train_ner.jsonl` and `val_ner.jsonl` files.

4. **Train the model:**

   Implement your training logic in `train.py` and run:

   ```sh
   python train.py
   ```

## Notes

- The scripts use [modelscope](https://modelscope.cn/) for dataset handling.
- Adjust the data paths and parameters as needed for your specific use case.
