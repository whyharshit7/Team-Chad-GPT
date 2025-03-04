import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import the feature extraction function from your main script
from alt import extract_enhanced_features

def load_model(model_path='rf_model.joblib'):
    """Load the trained model"""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict_image(model, image_path):
    """Predict if an image is indoor or outdoor museum"""
    # Extract features using the same function used for training
    features = extract_enhanced_features(image_path)
    
    if features is None:
        print(f"Error: Could not process image {image_path}")
        return None
    
    # Reshape for single sample prediction
    features = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    class_names = ['Indoor Museum', 'Outdoor Museum']
    result = class_names[prediction]
    confidence = probability[prediction]
    
    return result, confidence, probability

def visualize_prediction(image_path, result, confidence, probability):
    """Display the image with prediction results"""
    # Read and resize image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {result}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    
    # Display probability bars
    plt.subplot(1, 2, 2)
    class_names = ['Indoor Museum', 'Outdoor Museum']
    colors = ['skyblue', 'lightgreen']
    plt.bar(class_names, probability, color=colors)
    plt.ylim(0, 1)
    plt.title('Prediction Probabilities')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

def batch_predict(model, image_paths):
    """Process multiple images and show results"""
    results = []
    
    for image_path in image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        prediction = predict_image(model, image_path)
        
        if prediction:
            result, confidence, probability = prediction
            results.append((image_path, result, confidence, probability))
            print(f"Prediction: {result}, Confidence: {confidence:.2%}")
    
    return results

def visualize_batch_results(results):
    """Create a grid of images with their predictions"""
    n = len(results)
    cols = min(3, n)  # Maximum 3 columns
    rows = (n + cols - 1) // cols  # Calculate needed rows
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (image_path, result, confidence, _) in enumerate(results):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f"{os.path.basename(image_path)}\nPrediction: {result}\nConfidence: {confidence:.2%}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_results.png')
    plt.show()

def main():
    # Load model (default is Random Forest, but you can specify any of your saved models)
    model_path = input("Enter model path (default: gb_model.joblib): ") or "gb_model.joblib"
    model = load_model(model_path)
    
    # Ask for multiple image mode
    mode = input("Process (1) a single image or (2) multiple images? Enter 1 or 2: ")
    
    if mode == "1":
        # Single image mode
        image_path = input("Enter the path to the test image: ")
        
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist.")
            return
        
        # Make prediction
        result, confidence, probability = predict_image(model, image_path)
        
        # Print results
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        
        # Visualize
        visualize_prediction(image_path, result, confidence, probability)
    
    elif mode == "2":
        # Multiple images mode
        image_input = input("Enter path to folder or comma-separated list of image paths: ")
        
        if os.path.isdir(image_input):
            # Process all images in the directory
            image_paths = []
            for ext in ['jpg', 'jpeg', 'png']:
                image_paths.extend(glob(os.path.join(image_input, f'*.{ext}')))
                image_paths.extend(glob(os.path.join(image_input, f'*.{ext.upper()}')))
            
            if not image_paths:
                print("No images found in the directory.")
                return
                
            print(f"Found {len(image_paths)} images.")
        
        else:
            # Process comma-separated list of images
            image_paths = [path.strip() for path in image_input.split(',')]
            
        # Validate paths
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: File {path} does not exist and will be skipped.")
        
        if not valid_paths:
            print("No valid image paths provided.")
            return
            
        # Process batch
        results = batch_predict(model, valid_paths)
        
        # Generate summary report
        print("\n===== SUMMARY REPORT =====")
        for image_path, result, confidence, _ in results:
            print(f"{os.path.basename(image_path)}: {result} ({confidence:.2%})")
            
        # Visualize all results in a grid
        visualize_batch_results(results)
        
        # Generate CSV report
        with open('prediction_results.csv', 'w') as f:
            f.write('Image,Prediction,Confidence\n')
            for image_path, result, confidence, _ in results:
                f.write(f'{os.path.basename(image_path)},{result},{confidence:.4f}\n')
        
        print(f"\nResults saved to prediction_results.csv and batch_results.png")
    
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()