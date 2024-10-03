# MNIST-Classification

This repository contains an implementation of a Convolutional Neural Network (CNN) using PyTorch for classifying handwritten digits from the MNIST dataset. The model is trained to recognize digits 0-9 by leveraging convolutional layers followed by fully connected layers.

## Features

- **PyTorch-based CNN**: Implements a simple CNN with two convolutional layers, max pooling, and dropout.
- **MNIST Dataset**: Uses the classic MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
- **Customizable Training**: Supports command-line arguments for batch size, learning rate, number of epochs, and more.
- **GPU Support**: Automatically detects CUDA or MPS devices for faster training if available.
- **High Accuracy**: The model achieves a **99.19% accuracy** on the MNIST test set.

## Model Architecture

- **2 Convolutional Layers**: The first convolutional layer uses 32 filters and the second uses 64 filters.
- **2 Fully Connected Layers**: After flattening, the network has two fully connected layers with 128 neurons and 10 output neurons (one for each digit class).
- **Dropout**: Dropout is applied to prevent overfitting with a 25% rate after the convolutional layers and a 50% rate before the final fully connected layer.

## Saving the Model

You can save the trained model by adding the --save-model argument:

```bash
python MNIST-Classification.py --epochs 14 --save-model
```

This will save the model as mnist_cnn.pt which can be loaded later for inference.
