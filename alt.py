import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
from sklearn.semi_supervised import SelfTrainingClassifier
import joblib

def extract_enhanced_features(image_path):
    """
    Feature extraction with additional image descriptors.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # Resize to retain more details
    image = cv2.resize(image, (128, 128))
    features = []
    
    # HSV Color Histogram with increased bins
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    # Texture features (using a simple approach as a placeholder for Haralick texture)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = np.mean(cv2.calcHist([gray], [0], None, [32], [0, 256]), axis=0)
    
    # Edge features using Canny
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
    
    # Basic pixel statistics for each channel
    for channel in cv2.split(image):
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75)
        ])
    
    # Combine all features
    combined_features = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        hist_v.flatten(),
        haralick.flatten(),
        edge_hist.flatten(),
        features
    ])
    
    # Normalize the features
    combined_features = combined_features / np.sum(combined_features)
    return combined_features

def load_dataset(data_dir):
    """
    Enhanced dataset loading with parallel processing.
    """
    X = []
    y = []
    
    def process_image(args):
        file_path, label = args
        features = extract_enhanced_features(file_path)
        return (features, label) if features is not None else None
    
    args_list = []
    for label, subdir in enumerate(["museum-indoor", "museum-outdoor"]):
        folder = os.path.join(data_dir, subdir)
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            args_list.append((file_path, label))
    
    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, args_list))
    
    valid_results = [r for r in results if r is not None]
    X, y = zip(*valid_results)
    
    return np.array(X), np.array(y)

def create_models():
    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt'
        ))
    ])

    # Gradient Boosting pipeline
    gb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.2,
            min_samples_split=2,
            min_samples_leaf=1
        ))
    ])
    
    # Standard Decision Tree pipeline
    dt_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            random_state=42,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1
        ))
    ])
    
    # Semi-supervised Decision Tree pipeline using SelfTrainingClassifier.
    dt_semi_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SelfTrainingClassifier(
            DecisionTreeClassifier(
                random_state=42,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1
            )
        ))
    ])

    return rf_pipeline, gb_pipeline, dt_pipeline, dt_semi_pipeline

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    """
    Train and evaluate models.
    For the semi-supervised model, we simulate unlabeled data by masking a portion of the training labels.
    """
    rf_model, gb_model, dt_model, dt_semi_model = create_models()

    # Train fully supervised models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    
    np.random.seed(42)
    y_train_semi = np.array(y_train, copy=True)
    mask = np.random.rand(len(y_train_semi)) < 0.5  # 50% of labels become unlabeled
    y_train_semi[mask] = -1  # -1 indicates unlabeled samples

    dt_semi_model.fit(X_train, y_train_semi)

    # Evaluate and print results for all models on the fully labeled validation set
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Decision Tree': dt_model,
        'Semi-supervised Decision Tree': dt_semi_model
    }
    
    # Store results for plotting
    accuracies = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        predictions[name] = y_pred
        
        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            probabilities[name] = y_proba
        except:
            probabilities[name] = None
            
        accuracy = accuracy_score(y_val, y_pred)
        accuracies[name] = accuracy
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_val, y_pred)}")
    
    # Plot comparative results
    plot_model_comparison(models, X_val, y_val, accuracies, predictions, probabilities)
    
    # Save the models
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(rf_model, 'saved_models/rf_model.joblib')
    joblib.dump(gb_model, 'saved_models/gb_model.joblib')
    joblib.dump(dt_model, 'saved_models/dt_model.joblib')
    joblib.dump(dt_semi_model, 'saved_models/dt_semi_model.joblib')
    print("Models saved successfully to 'saved_models' directory.")
    
    return rf_model, gb_model, dt_model, dt_semi_model

def plot_model_comparison(models, X_val, y_val, accuracies, predictions, probabilities):
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0']
    plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1.0])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    
    # Plot 2: Confusion Matrix Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=14)
        axes[i].set_xlabel('Predicted label', fontsize=12)
        axes[i].set_ylabel('True label', fontsize=12)
        axes[i].set_xticklabels(['Indoor', 'Outdoor'])
        axes[i].set_yticklabels(['Indoor', 'Outdoor'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    
    # Plot 3: ROC Curves (for models that support predict_proba)
    plt.figure(figsize=(10, 8))
    
    valid_models = {name: prob for name, prob in probabilities.items() if prob is not None}
    
    for i, (name, y_proba) in enumerate(valid_models.items()):
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('roc_curves.png')

def performance_comparison_rf(X_train, y_train, X_val, y_val):
    """
    Compare Random Forest performance by varying n_estimators and max_depth.
    """
    print("=== Random Forest Hyperparameter Comparison ===")
    n_estimators_list = [100, 200]
    max_depth_list = [15, 20]
    
    # Store results for plotting
    results = []
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt'
                ))
            ])
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results.append((n_estimators, max_depth, accuracy))
            print(f"RF with n_estimators={n_estimators}, max_depth={max_depth} -> Accuracy: {accuracy:.4f}")
    
    # Plot heatmap of hyperparameter performance
    plot_rf_hyperparameter_heatmap(results, n_estimators_list, max_depth_list)

def plot_rf_hyperparameter_heatmap(results, n_estimators_list, max_depth_list):
    """
    Plot heatmap for Random Forest hyperparameter tuning results.
    """
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(n_estimators_list), len(max_depth_list)))
    for n_idx, n_est in enumerate(n_estimators_list):
        for d_idx, depth in enumerate(max_depth_list):
            for result in results:
                if result[0] == n_est and result[1] == depth:
                    heatmap_data[n_idx, d_idx] = result[2]
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu',
                    xticklabels=max_depth_list, yticklabels=n_estimators_list)
    
    plt.title('Random Forest Hyperparameter Tuning Results', fontsize=16)
    plt.xlabel('max_depth', fontsize=14)
    plt.ylabel('n_estimators', fontsize=14)
    plt.tight_layout()
    plt.savefig('rf_hyperparameter_heatmap.png')

def performance_comparison_gb(X_train, y_train, X_val, y_val):
    """
    Compare Gradient Boosting performance by varying learning_rate and max_depth.
    """
    print("=== Gradient Boosting Hyperparameter Comparison ===")
    learning_rate_list = [0.2]
    max_depth_list = [4, 5]
    
    # Store results for plotting
    results = []
    
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            gb_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_samples_split=2,
                    min_samples_leaf=1
                ))
            ])
            gb_model.fit(X_train, y_train)
            y_pred = gb_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results.append((learning_rate, max_depth, accuracy))
            print(f"GB with learning_rate={learning_rate}, max_depth={max_depth} -> Accuracy: {accuracy:.4f}")
    
    # Plot heatmap of hyperparameter performance
    plot_gb_hyperparameter_heatmap(results, learning_rate_list, max_depth_list)
    
    # Plot learning curves for best parameters
    best_result = max(results, key=lambda x: x[2])
    plot_gb_learning_curve(X_train, y_train, X_val, y_val, best_result[0], best_result[1])

def plot_gb_hyperparameter_heatmap(results, learning_rate_list, max_depth_list):
    """
    Plot heatmap for Gradient Boosting hyperparameter tuning results.
    """
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(learning_rate_list), len(max_depth_list)))
    for lr_idx, lr in enumerate(learning_rate_list):
        for d_idx, depth in enumerate(max_depth_list):
            for result in results:
                if result[0] == lr and result[1] == depth:
                    heatmap_data[lr_idx, d_idx] = result[2]
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=max_depth_list, yticklabels=learning_rate_list)
    
    plt.title('Gradient Boosting Hyperparameter Tuning Results', fontsize=16)
    plt.xlabel('max_depth', fontsize=14)
    plt.ylabel('learning_rate', fontsize=14)
    plt.tight_layout()
    plt.savefig('gb_hyperparameter_heatmap.png')

def plot_gb_learning_curve(X_train, y_train, X_val, y_val, best_lr, best_depth):
    """
    Plot learning curve (training and validation error) for Gradient Boosting
    with different numbers of estimators.
    """
    # Create a range of n_estimators values
    n_estimators_range = [50, 100, 150, 200, 250, 300]
    train_scores = []
    val_scores = []
    
    for n_est in n_estimators_range:
        gb_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                random_state=42,
                n_estimators=n_est,
                max_depth=best_depth,
                learning_rate=best_lr,
                min_samples_split=2,
                min_samples_leaf=1
            ))
        ])
        
        gb_model.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, gb_model.predict(X_train)))
        val_scores.append(accuracy_score(y_val, gb_model.predict(X_val)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, 'o-', color='#4CAF50', label='Training accuracy')
    plt.plot(n_estimators_range, val_scores, 'o-', color='#2196F3', label='Validation accuracy')
    plt.title(f'GB Learning Curve (lr={best_lr}, depth={best_depth})', fontsize=16)
    plt.xlabel('Number of estimators', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('gb_learning_curve.png')

def plot_feature_importance(rf_model, gb_model):
    # Get feature importances
    rf_importances = rf_model.named_steps['classifier'].feature_importances_
    gb_importances = gb_model.named_steps['classifier'].feature_importances_
    
    # Create feature names (since we don't have actual feature names)
    feature_count = len(rf_importances)
    feature_names = [f"Feature {i+1}" for i in range(feature_count)]
    
    # Get top 20 features for each model
    rf_indices = np.argsort(rf_importances)[-20:]
    gb_indices = np.argsort(gb_importances)[-20:]
    
    # Plot Random Forest feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(20), rf_importances[rf_indices], color='#4CAF50')
    plt.yticks(range(20), [feature_names[i] for i in rf_indices])
    plt.title('Top 20 Features Importance (Random Forest)', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    
    # Plot Gradient Boosting feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(20), gb_importances[gb_indices], color='#2196F3')
    plt.yticks(range(20), [feature_names[i] for i in gb_indices])
    plt.title('Top 20 Features Importance (Gradient Boosting)', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('gb_feature_importance.png')

def evaluate_on_test_set(best_model, test_dir):
    """
    Evaluate the best model on the test dataset.
    """
    print("\nLoading test dataset for final evaluation...")
    X_test, y_test = load_dataset(test_dir)
    print(f"Test dataset loaded. Number of samples: {len(y_test)}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    
    # Save results to a file
    with open('test_results.txt', 'w') as f:
        f.write("===== FINAL TEST RESULTS =====\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}\n")
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks([0.5, 1.5], ['Indoor', 'Outdoor'])
    plt.yticks([0.5, 1.5], ['Indoor', 'Outdoor'])
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    
    return accuracy, cm, report

def main():
    # Set paths for training, validation, and test data
    train_dir = "C:/Users/HARSHIT/Desktop/ai/train"
    test_dir = "C:/Users/HARSHIT/Desktop/ai/test"
    
    print("Loading training dataset...")
    X, y = load_dataset(train_dir)
    print("Dataset loaded. Number of samples:", len(y))
    
    # Stratified split of the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training and evaluating models...")
    rf_model, gb_model, dt_model, dt_semi_model = train_and_evaluate_models(X_train, y_train, X_val, y_val)
    
    # Compare different hyperparameter settings for Random Forest and Gradient Boosting
    performance_comparison_rf(X_train, y_train, X_val, y_val)
    performance_comparison_gb(X_train, y_train, X_val, y_val)
    
    # Plot feature importance for tree-based models
    plot_feature_importance(rf_model, gb_model)
    
    # Select best model based on validation accuracy
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Decision Tree': dt_model,
        'Semi-supervised Decision Tree': dt_semi_model
    }
    
    # Evaluate each model on validation set
    val_accuracies = {}
    for name, model in models.items():
        val_accuracies[name] = accuracy_score(y_val, model.predict(X_val))
    
    # Find the best model
    best_model_name = max(val_accuracies, key=val_accuracies.get)
    best_model = models[best_model_name]
    
    print(f"\nBest model based on validation accuracy: {best_model_name} ({val_accuracies[best_model_name]:.4f})")
    
    # Ask user if they want to evaluate on test set
    while True:
        choice = input("\nDo you want to evaluate the best model on the test set? (y/n): ").lower()
        if choice == 'y':
            evaluate_on_test_set(best_model, test_dir)
            break
        elif choice == 'n':
            print("Skipping test set evaluation. You can run the test_model.py script for demo.")
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")
    
    print("\nAll plots have been saved. Check the current directory for visualization files.")
    print("Models have been saved to the 'saved_models' directory for future use.")

if __name__ == "__main__":
    main()