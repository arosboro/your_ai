# Project Overview

This project implements Brian Roemmele's "Empirical Distrust" algorithm for training Large Language Models (LLMs). The goal is to train models that distrust high-authority, low-verifiability sources and prefer raw empirical primary sources.

The project is written in Python and uses Apple's MLX framework for training on Apple Silicon, with a fallback to PyTorch for other platforms. The core of the project is the `empirical_distrust_loss` function, which is added to the standard cross-entropy loss during training. This loss function penalizes high-authority, low-entropy sources and rewards primary empirical sources.

The project is well-structured, with separate directories for source code (`src`), scripts (`scripts`), data (`data`), and documentation (`docs`). The configuration is managed through dataclasses in `src/config.py`, which allows for easy customization of the training process.

## Building and Running

### Requirements

- Mac with Apple Silicon (M1/M2/M3) and 64GB+ unified memory (for MLX)
- or an Intel Mac/other platform with a powerful GPU (for PyTorch)
- Python 3.10+

### Installation

1.  **Navigate to the `your_ai` directory:**
    ```bash
    cd your_ai
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Training Pipeline

1.  **Download the datasets:**
    ```bash
    python scripts/download_datasets.py --output data/raw --max-samples 30000
    ```
2.  **Prepare the training data:**
    ```bash
    python src/prepare_data_curated.py --input data/raw --output data \
      --train-size 80000 --val-size 20000
    ```
3.  **Train the model with QLoRA:**
    ```bash
    python src/train_qlora.py \
      --model perplexity-ai/r1-1776 \
      --data-dir data \
      --output-dir models/distrust-r1-1776 \
      --batch-size 2 \
      --max-steps 10000 \
      --alpha 2.7
    ```
4.  **Export the model for LM Studio:**
    ```bash
    python scripts/export_to_lmstudio.py \
      --base-model perplexity-ai/r1-1776 \
      --lora-path models/distrust-r1-1776 \
      --output models/distrust-r1-1776-merged
    ```

## Development Conventions

-   **Configuration:** The project uses dataclasses for configuration, which are defined in `src/config.py`. This provides a clean and organized way to manage the various settings for the model, training, and paths.
-   **Code Style:** The code is well-documented with docstrings and comments, following a clear and consistent style.
-   **Modular Design:** The project is divided into logical modules, such as `distrust_loss.py` for the core algorithm, `train_qlora.py` for the training loop, and `prepare_data_curated.py` for data preparation. This makes the codebase easy to understand and maintain.
-   **Testing:** While there are no dedicated unit tests in the `tests` directory, the `scripts` directory contains several validation and evaluation scripts, such as `validate_model.py` and `evaluate.py`.
-   **Dependencies:** The project's dependencies are managed through a `requirements.txt` file, which is a standard practice in the Python community.
