# Museum Indoor/Outdoor Image Classification

A deep learning project to classify museum images as indoor or outdoor using PyTorch and Convolutional Neural Networks (CNNs).

## Project Overview

This project implements a custom CNN architecture to classify museum images into two categories:
- Indoor museum scenes
- Outdoor museum scenes

The repository includes code for training, hyperparameter tuning, evaluation, and inference on new images.

## Features

- Custom CNN architecture with configurable hyperparameters
- Hyperparameter tuning for optimal model configuration
- Data augmentation for improved model generalization
- Comprehensive visualization of training progress and results
- Model evaluation with detailed metrics
- Easy inference on new test images

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn
- OpenCV
- PIL (Pillow)

You can install the required packages using:
```
pip install torch torchvision matplotlib numpy scikit-learn opencv-python pillow
```

## Project Structure

```
museum-classification/
├── cnn.py              # Contains the CNN model implementation
├── test_script.py      # Script for testing/evaluating the model
├── README.md           # Project documentation
├── saved_models/       # Directory for saved model checkpoints
└── data/
    ├── train/          # Training data directory
    │   ├── museum-indoor/
    │   └── museum-outdoor/
    └── test/           # Test data directory
        ├── museum-indoor/
        └── museum-outdoor/
```

## Dataset Preparation

The project expects the dataset to be organized in the following structure:

```
data/
├── train/
│   ├── museum-indoor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── museum-outdoor/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── museum-indoor/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── museum-outdoor/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Usage

### Training the Model

To train the model with hyperparameter tuning:

```bash
python cnn.py
```

This will:
1. Load the training data
2. Split into training and validation sets
3. Run hyperparameter tuning to find the best configuration
4. Train the final model with the optimal parameters
5. Save visualizations and the trained model

### Testing the Model

To evaluate a trained model on the test set:

```bash
python test_script.py --model saved_models/custom_cnn_4-max.pth --test data/test --results evaluation_results.txt
```

Arguments:
- `--model`: Path to the trained model file (default: 'model/museum_cnn.pth')
- `--test`: Path to test directory (default: 'test')
- `--batch`: Batch size for evaluation (default: 32)
- `--results`: Path to save evaluation results (default: 'results.txt')

## Model Architecture

The CNN architecture is defined in the `MuseumCNN` class with configurable hyperparameters:
- Number of convolutional layers (3-5)
- Pooling type (max, avg, or none)

The model uses:
- Convolutional layers with batch normalization and ReLU activation
- Pooling layers (configurable)
- Dropout for regularization
- Fully connected layers for classification

## Results

The training process generates several visualizations:
- Training and validation loss/accuracy curves
- Hyperparameter comparison plots
- Model prediction visualization on sample images

Evaluation results include:
- Overall accuracy
- Per-class accuracy
- Detailed results for each test image

## Customization

You can modify the model architecture and training parameters:
- Change the number of convolutional layers
- Adjust the pooling strategy
- Modify learning rate, batch size, or number of epochs
- Update data augmentation techniques
