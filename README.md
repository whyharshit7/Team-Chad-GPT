# Team-Chad-GPT
# Museum Image Classification

A machine learning project that classifies museum images as indoor or outdoor using computer vision and ensemble methods.

## Overview

This project uses advanced image processing techniques and multiple machine learning models to classify museum images. The system extracts various image features including color histograms, texture features, and edge information to distinguish between indoor and outdoor museum environments.

## Features

- **Enhanced Feature Extraction**: Extracts color histograms, texture descriptors, edge features, and other visual characteristics
- **Multiple Classification Models**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Decision Tree Classifier
  - Semi-supervised Decision Tree Classifier
- **Performance Analysis**: Comprehensive evaluation with accuracy metrics, confusion matrices, ROC curves, and precision-recall analysis
- **Hyperparameter Tuning**: Includes optimization for both Random Forest and Gradient Boosting models
- **Visualization**: Generates plots for model comparison, feature importance, and hyperparameter performance
- **Parallel Processing**: Uses multi-threading for efficient image processing
- **Batch Prediction**: Supports both single image and batch prediction modes

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- joblib

## Installation

```bash
git clone https://github.com/whyharshit7/Team-Chad-GPT.git
cd Team-Chad-GPT
pip install -r requirements.txt
```

## Usage

### Training

1. Organize your dataset in the following structure:
   ```
   data/
     ├── train/
     │     ├── museum-indoor/
     │     │     ├── image1.jpg
     │     │     ├── image2.jpg
     │     │     └── ...
     │     └── museum-outdoor/
     │           ├── image1.jpg
     │           ├── image2.jpg
     │           └── ...
     └── test/
           ├── museum-indoor/
           │     ├── image1.jpg
           │     ├── image2.jpg
           │     └── ...
           └── museum-outdoor/
                 ├── image1.jpg
                 ├── image2.jpg
                 └── ...
   ```

2. Update the data directory paths in `alt.py`:
   ```python
   train_dir = "path/to/your/train/directory"
   test_dir = "path/to/your/test/directory"
   ```

3. Run the training script:
   ```bash
   python alt.py
   ```

### Testing

Use the `test_script.py` to make predictions on new images:

```bash
python test_script.py
```

The script offers two options:
1. Single image prediction
2. Batch prediction (entire directory or list of images)

## Model Performance

The system evaluates and compares multiple models:
- Model accuracy comparison
- Confusion matrices
- ROC curves
- Feature importance analysis
- Hyperparameter performance

Output visualizations include:
- `model_accuracy_comparison.png`
- `confusion_matrices.png`
- `roc_curves.png`
- `rf_feature_importance.png`
- `gb_feature_importance.png`
- `rf_hyperparameter_heatmap.png`
- `gb_hyperparameter_heatmap.png`
- `gb_learning_curve.png`

## Sample Output

When using the test script for batch prediction, it generates:
- Console output with prediction details
- `batch_results.png` with visualization of all predictions
- `prediction_results.csv` with tabulated results

## How It Works

1. **Feature Extraction**: The system extracts multiple types of features from each image:
   - HSV color histograms with 32 bins
   - Texture descriptors
   - Edge detection features
   - Basic pixel statistics

2. **Model Training**: The extracted features are used to train multiple classification models in parallel.

3. **Evaluation**: The models are evaluated using various metrics including accuracy, precision, recall, and F1-score.

4. **Hyperparameter Tuning**: The system compares different hyperparameter configurations for both Random Forest and Gradient Boosting models.

5. **Feature Importance**: The system analyzes and visualizes which features contribute most to the classification decision.

## Future Improvements

- Implement deep learning models (CNN) for comparison
- Add more feature extraction methods
- Include more hyperparameter tuning options
- Implement cross-validation
- Add a web interface for easy usage
