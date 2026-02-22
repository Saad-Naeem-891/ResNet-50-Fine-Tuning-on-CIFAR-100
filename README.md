# ResNet50 Fine-Tuning on CIFAR100

## Overview

This project contains a PyTorch script designed to fine-tune a pre-trained ResNet50 convolutional neural network on the CIFAR100 dataset. The pipeline includes data splitting, training, validation, testing, and performance visualization.

## Requirements

* Python 3.x
* PyTorch
* Torchvision
* Matplotlib

## Dataset

The script automatically downloads and utilizes the **CIFAR100** dataset via `torchvision.datasets`.

* **Training Set:** 80% of the original training data.
* **Validation Set:** 20% of the original training data.
* **Test Set:** The standard CIFAR100 test dataset.

## Model Architecture

* **Base Model:** ResNet50 pre-trained using `ResNet50_Weights.DEFAULT`.
* **Feature Extraction:** All pre-trained base layers are frozen (`requires_grad = False`).
* **Classification Head:** The final fully connected layer is replaced with a new linear layer configured for 100 output classes.
* **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9.
* **Loss Function:** Cross Entropy Loss.


## Output

1. **Training Logs:** Outputs the average training and validation loss at the end of each epoch.
2. **Evaluation Metrics:** Displays the average loss and accuracy percentage on the test dataset after training concludes.
3. **Loss Curve:** Renders a Matplotlib graph comparing the training and validation loss across all epochs.
