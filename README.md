# GRPO-IMO: Fine-tuning DeepSeek-R1 for Math Reasoning with GRPO

This project implements Group Relative Policy Optimization (GRPO) fine-tuning for the DeepSeek-R1 model on International Mathematical Olympiad (IMO) problems.

## Overview

This pet-project demonstrates how to fine-tune a reasoning language model using GRPO, a reinforcement learning algorithm that optimizes model performance on mathematical reasoning tasks. The implementation uses the DeepSeek-R1-Distill-Qwen-1.5B model and trains it on IMO problems to improve mathematical reasoning capabilities.

## Features

- **GRPO Training**: Implements Group Relative Policy Optimization for efficient RL fine-tuning
- **Math Reasoning**: Specialized for solving complex mathematical problems
- **Reward Functions**: Multiple reward mechanisms including format, accuracy, and solution quality
- **Fast Inference**: Uses Unsloth and vLLM for optimized training and inference
- **Modular Code**: Clean, structured codebase separated into logical modules

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd GRPO_IMO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: This project requires GPU support and significant computational resources for training.

## Usage

### Training

Run the complete training pipeline:
```bash
python main.py
```

This will:
1. Download the IMO dataset from Kaggle
2. Prepare and preprocess the data
3. Load the base model and apply PEFT
4. Train using GRPO with custom reward functions
5. Evaluate the trained model

### Configuration

Modify `configs/config.py` to adjust training parameters:

```python
class CFG:
    MAX_TRAIN = 1000      # Number of training samples
    MAX_TOKENS = 2048     # Maximum token length
    NUM_GENERATIONS = 4   # Number of generations per sample
    BATCH_SIZE = 1        # Training batch size
    MAX_STEPS = 80        # Training steps
    LR = 1e-5            # Learning rate
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

## Reward Functions

The training uses three reward functions:

1. **Format Reward**: Ensures proper `<think>...</think>` and `\boxed{}` formatting
2. **Accuracy Reward**: Exact match with ground truth answers
3. **Solution Quality**: Levenshtein similarity of reasoning traces

## Results

After training, the model shows improved performance on IMO problems:

- **Formatting**: ~68% properly formatted responses
- **Accuracy**: ~54% correct answers
- **Solution Quality**: ~28% average similarity score

Fine-tuned model: https://huggingface.co/Dortp58/deepseekr1-qwen-1.5B_grpo_imo

## Citation

Based on the DeepSeek-R1 paper: [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

