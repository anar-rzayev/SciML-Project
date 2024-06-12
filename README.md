# Final Project: Neural SDE vs Neural ODE

## Overview

This project involves the implementation and evaluation of stochastic differential equation (SDE) models for deep learning, particularly focusing on adversarial attacks and robustness in classification tasks. The models are tested against various datasets like CIFAR-10, STL10, MNIST, and Tiny ImageNet under different noise conditions.

## Project Structure

The project directory includes scripts for training models, comparing methods, and visualizing the results.

## Requirements

- Python 3.8 or later
- PyTorch 1.8 or later
- torchvision
- torchdiffeq (for ODE and SDE integration)
- NumPy
- PIL
- CUDA (for GPU acceleration)

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine or server to get started:

```bash
git clone [repository-url]
```

Replace `[repository-url]` with the actual URL of the repository.

### 2. Install Dependencies

Install the required Python libraries using pip:

```bash
pip install torch torchvision torchdiffeq numpy pillow
```

### 3. Data Preparation

#### CIFAR-10 and CIFAR-10-C

- Download CIFAR-10:
  
  The CIFAR-10 dataset can be automatically downloaded using torchvision datasets. The training scripts handle this automatically.

- CIFAR-10-C:
  
  CIFAR-10-C contains various corrupted versions of CIFAR-10 images and needs to be downloaded separately. Place the downloaded files in `~/data/CIFAR-10-C`. Ensure you have the `labels.npy` and corruption data files in this directory.

#### STL10

STL10 will be downloaded automatically by the training scripts using torchvision if not present.

#### MNIST

Similar to CIFAR-10, MNIST is downloaded automatically via the torchvision datasets module.

#### Tiny ImageNet and Tiny ImageNet-C

- Tiny ImageNet can be downloaded from its official website. After downloading, extract it to `~/data/Tiny-ImageNet`.
  
- For Tiny ImageNet-C, similar to CIFAR-10-C, download the dataset and ensure it's in the appropriate directory structure like `~/data/Tiny-ImageNet-C`.

### 4. Running Scripts

- To train models, use scripts named according to their functionality, e.g., `train_xde.py` for training an SDE model:

  ```bash
  bash run_xde.sh
  ```

- To evaluate models under adversarial attacks:

  ```bash
  bash acc_under_attack.sh
  ```

- To visualize model states or perform other evaluations:

  ```bash
  bash visualize_mid_state.sh
  ```

### 5. Configuration

Edit the shell scripts or directly modify the Python scripts to adjust training parameters, model configurations, or dataset paths as needed. Each script has a set of argparse command-line arguments to facilitate this customization.

### 6. Results and Output

Results such as accuracy, model outputs, and logs are saved in the specified directories as per the scripts. Check the logs and output folders for detailed results.

## Note

Ensure that CUDA is correctly installed and configured if running on GPUs. Adjust the `CUDA_VISIBLE_DEVICES` settings in shell scripts according to your hardware configuration.
