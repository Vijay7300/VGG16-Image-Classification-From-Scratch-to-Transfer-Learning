# VGG16 Image Classification: From Scratch to Transfer Learning

This project explores image classification using **VGG16 with TensorFlow and Keras**, comparing a VGG16 model built from scratch with transfer learning and fine-tuning using pretrained ImageNet weights.

## Overview

The project implements a deep learning pipeline for multiclass image classification using four output classes. It covers dataset loading, image resizing and normalization, data augmentation, model training, evaluation, and performance visualization.

## Key Features

- **VGG16 from Scratch:** Builds a VGG16-style convolutional neural network with five convolutional blocks and fully connected layers.
- **Transfer Learning:** Uses VGG16 pretrained on ImageNet with `include_top=False` to reuse learned visual features.
- **Fine-Tuning:** Explores unfreezing selected pretrained layers to adapt the model to the target dataset.
- **Data Augmentation:** Applies random horizontal and vertical flips and rotations to training images.
- **Model Training:** Uses the Adam optimizer and sparse categorical cross-entropy loss.
- **Performance Evaluation:** Evaluates training, validation, and test datasets using accuracy, loss, confusion matrices, and training curves.
- **Reproducibility:** Sets random seeds for Python, NumPy, and TensorFlow.

## Technology Stack

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow

1. Load and prepare the training, validation, and test datasets.
2. Resize images to 224 × 224 pixels and normalize pixel values.
3. Apply data augmentation to training data.
4. Build and train a VGG16 model from scratch.
5. Experiment with transfer learning using ImageNet pretrained weights.
6. Fine-tune selected VGG16 layers.
7. Evaluate model performance and visualize results.

## Learning Objectives

- Understand CNN architecture and VGG16.
- Compare training from scratch with transfer learning.
- Learn feature extraction and fine-tuning.
- Apply data augmentation to image datasets.
- Evaluate multiclass classification models using confusion matrices, accuracy, and loss curves.

## Note

The notebook uses the Kaggle Chest X-Ray Pneumonia, COVID-19, and Tuberculosis dataset. The configured model has four output classes, so verify the dataset's actual folder names and class mapping before interpreting predictions. The notebook explores multiple approaches; no final performance score is claimed here.
