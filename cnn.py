import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import itertools

class MuseumImageDataset(Dataset):
    """Dataset class for museum indoor/outdoor classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load indoor images (label 0)
        indoor_dir = os.path.join(data_dir, "museum-indoor")
        for img_name in os.listdir(indoor_dir):
            self.image_paths.append(os.path.join(indoor_dir, img_name))
            self.labels.append(0)
        
        # Load outdoor images (label 1)
        outdoor_dir = os.path.join(data_dir, "museum-outdoor")
        for img_name in os.listdir(outdoor_dir):
            self.image_paths.append(os.path.join(outdoor_dir, img_name))
            self.labels.append(1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MuseumCNN(nn.Module):
    """Custom CNN architecture for museum indoor/outdoor classification with hyperparameters"""
    def __init__(self, num_conv_layers=4, pooling_type="max"):
        super(MuseumCNN, self).__init__()
        
        # Hyperparameters:
        # 1. num_conv_layers: Number of convolutional blocks (2-5)
        # 2. pooling_type: Type of pooling layer ("max", "avg", or "none")
        
        self.num_conv_layers = num_conv_layers
        self.pooling_type = pooling_type
        
        # Define pooling layer based on hyperparameter
        if pooling_type == "max":
            self.pool = nn.MaxPool2d(2)
        elif pooling_type == "avg":
            self.pool = nn.AvgPool2d(2)
        else:  # "none"
            self.pool = nn.Identity()
        
        # Create convolutional layers based on num_conv_layers
        layers = []
        in_channels = 3
        
        # Channel sizes for each layer
        channels = [32, 64, 128, 256, 512][:num_conv_layers]
        
        for i, out_channels in enumerate(channels):
            # Add conv layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            # Add pooling if not "none"
            if pooling_type != "none":
                layers.append(self.pool)
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate input size for fully connected layer based on architecture
        # Default image size is 128x128
        fc_size = 128 // (2 ** (num_conv_layers if pooling_type != "none" else 0))
        fc_in_features = channels[-1] * fc_size * fc_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=0.001):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_val_acc': val_accs[-1]
    }

def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset and return predictions and metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    
    return {
        'preds': all_preds,
        'labels': all_labels,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report
    }

def plot_training_history(history):
    """Plot training and validation loss/accuracy curves"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.plot(history['val_accs'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training history plot saved as 'training_history.png'")

def plot_hyperparameter_comparison(results):
    """Plot comparison of different hyperparameter configurations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Prepare data for plotting
    configs = list(results.keys())
    final_val_accs = [results[config]['final_val_acc'] for config in configs]
    
    # Sort by validation accuracy
    sorted_indices = np.argsort(final_val_accs)[::-1]  # Descending order
    sorted_configs = [configs[i] for i in sorted_indices]
    sorted_accs = [final_val_accs[i] for i in sorted_indices]
    
    # Barplot of final validation accuracies
    axes[0].bar(range(len(sorted_configs)), sorted_accs, color='skyblue')
    axes[0].set_xticks(range(len(sorted_configs)))
    axes[0].set_xticklabels(sorted_configs, rotation=45, ha='right')
    axes[0].set_title('Final Validation Accuracy by Model Configuration', fontsize=14)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].grid(axis='y')
    
    # Line plot of validation accuracy over epochs
    for config in configs:
        num_conv, pooling = config.split('-')
        label = f"{num_conv} conv, {pooling} pooling"
        axes[1].plot(results[config]['val_accs'], label=label)
    
    axes[1].set_title('Validation Accuracy Over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png')
    plt.close()
    
    print("Hyperparameter comparison plot saved as 'hyperparameter_comparison.png'")
    
def visualize_model_predictions(model, data_loader, device, num_images=8):
    """Visualize model predictions on sample images"""
    model.eval()
    
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Get predictions
    with torch.no_grad():
        images_device = images[:num_images].to(device)
        outputs = model(images_device)
        _, preds = torch.max(outputs, 1)
    
    # Convert images for display
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Plot the images with predictions
    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(2, num_images//2, i+1)
        img = denormalize(images[i]).cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        true_label = 'Indoor' if labels[i] == 0 else 'Outdoor'
        pred_label = 'Indoor' if preds[i] == 0 else 'Outdoor'
        color = 'green' if preds[i] == labels[i] else 'red'
        
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()
    
    print("Prediction visualization saved as 'prediction_visualization.png'")
    
def save_model(model, model_name):
    """Save the trained model"""
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save model architecture info
    model_info_path = f'saved_models/{model_name}_info.txt'
    with open(model_info_path, 'w') as f:
        f.write(str(model))
    print(f"Model architecture info saved to {model_info_path}")
    
def run_hyperparameter_tuning(train_loader, val_loader, device):
    """Run hyperparameter tuning experiments"""
    # Define hyperparameter grid
    num_conv_layers_options = [3, 4, 5]
    pooling_type_options = ["max", "avg"]
    
    # Store results
    results = {}
    best_val_acc = 0
    best_model = None
    best_config = None
    
    # Run experiments for each combination
    for num_conv_layers, pooling_type in itertools.product(num_conv_layers_options, pooling_type_options):
        config_name = f"{num_conv_layers}-{pooling_type}"
        print(f"\n{'='*50}")
        print(f"Training model with {num_conv_layers} conv layers and {pooling_type} pooling")
        print(f"{'='*50}")
        
        # Create and train model with current hyperparameters
        model = MuseumCNN(num_conv_layers=num_conv_layers, pooling_type=pooling_type).to(device)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=10,  # Reduced for tuning, increase for final model
            lr=0.001
        )
        
        # Store results
        results[config_name] = history
        
        # Check if this is the best model so far
        if history['final_val_acc'] > best_val_acc:
            best_val_acc = history['final_val_acc']
            best_model = model
            best_config = config_name
    
    return results, best_model, best_config

def main():
    # Set paths for training and test data
    train_dir = "C:/Users/HARSHIT/Desktop/ai/train"
    
    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for images
    # Standard transforms for CNN image classification
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Creating datasets...")
    full_dataset = MuseumImageDataset(train_dir, transform=train_transform)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation dataset
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Dataset loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Run hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    tuning_results, best_model, best_config = run_hyperparameter_tuning(train_loader, val_loader, device)
    
    # Plot hyperparameter comparison
    plot_hyperparameter_comparison(tuning_results)
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best validation accuracy: {tuning_results[best_config]['final_val_acc']:.4f}")
    
    # Train best model configuration for full epochs
    print(f"\n{'='*50}")
    print(f"Training final model with best configuration: {best_config}")
    print(f"{'='*50}")
    
    # Parse best config
    num_conv_layers, pooling_type = best_config.split('-')
    final_model = MuseumCNN(num_conv_layers=int(num_conv_layers), pooling_type=pooling_type).to(device)
    
    # Train for full number of epochs
    history = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=20,
        lr=0.001
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on validation set
    print("\nEvaluating final model on validation set:")
    val_results = evaluate_model(final_model, val_loader, device)
    
    # Save model
    save_model(final_model, f"custom_cnn_{best_config}")
    
    # Visualize some predictions
    visualize_model_predictions(final_model, val_loader, device)
    
    print("\nTraining and evaluation complete.")
    print("All visualizations and model have been saved.")

if __name__ == "__main__":
    main()