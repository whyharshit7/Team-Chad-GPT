import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
from cnn import MuseumCNN
import datetime

class MuseumTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['museum-indoor', 'museum-outdoor']  # Updated to match exact folder names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, filename),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def save_results_to_file(results, file_path):
    """Save the evaluation results to a text file"""
    with open(file_path, 'w') as f:
        f.write(f"Museum CNN Evaluation Results\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"Overall Test Accuracy: {results['overall_accuracy']:.4f} ({results['correct']}/{results['total']})\n\n")
        
        f.write("Per-class Accuracy:\n")
        for class_name, accuracy, correct, total in results['class_results']:
            f.write(f"  {class_name}: {accuracy:.4f} ({correct}/{total})\n")
        
        f.write(f"\n{'='*50}\n")
        f.write("Detailed Results:\n\n")
        
        for i, (img_path, true_label, pred_label, correct) in enumerate(results['detailed_results']):
            status = "CORRECT" if correct else "INCORRECT"
            f.write(f"Image {i+1}: {os.path.basename(img_path)}\n")
            f.write(f"  True: {true_label}, Predicted: {pred_label}, Status: {status}\n\n")
            
    print(f"Results saved to {file_path}")

def evaluate_model(model_path, test_dir, batch_size=32, results_file=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = MuseumCNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found")
        return
    
    model.eval()
    
    # Define transforms (use the same as in your training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = MuseumTestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if len(test_dataset) == 0:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Found {len(test_dataset)} images in the test directory")
    print(f"Class mapping: {test_dataset.class_to_idx}")
    
    # Evaluate model
    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    detailed_results = []
    
    class_names = ['museum-indoor', 'museum-outdoor']
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy and detailed results
            for i in range(labels.size(0)):
                idx = batch_idx * batch_size + i
                if idx < len(test_dataset):
                    img_path, _ = test_dataset.samples[idx]
                    label = labels[i].item()
                    pred = predicted[i].item()
                    is_correct = (pred == label)
                    
                    class_total[label] = class_total.get(label, 0) + 1
                    if is_correct:
                        class_correct[label] = class_correct.get(label, 0) + 1
                    
                    detailed_results.append((
                        img_path,
                        class_names[label],
                        class_names[pred],
                        is_correct
                    ))
    
    # Calculate and print results
    overall_accuracy = correct / total
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    
    print("\nPer-class Accuracy:")
    class_accuracy_results = []
    for i in range(len(class_names)):
        if class_total.get(i, 0) > 0:
            class_acc = class_correct.get(i, 0) / class_total.get(i, 0)
            print(f"  {class_names[i]}: {class_acc:.4f} ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")
            class_accuracy_results.append((
                class_names[i],
                class_acc,
                class_correct.get(i, 0),
                class_total.get(i, 0)
            ))
        else:
            print(f"  {class_names[i]}: No samples found")
            class_accuracy_results.append((class_names[i], 0, 0, 0))
    
    # Compile results
    results = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'class_results': class_accuracy_results,
        'detailed_results': detailed_results
    }
    
    # Save results to file if specified
    if results_file:
        save_results_to_file(results, results_file)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Museum CNN on test set')
    parser.add_argument('--model', type=str, default='model/museum_cnn.pth', 
                        help='Path to the trained model')
    parser.add_argument('--test', type=str, default='test', 
                        help='Path to test directory containing musuem_indoor and museum_outdoor subdirectories')
    parser.add_argument('--batch', type=int, default=32, 
                        help='Batch size for evaluation')
    parser.add_argument('--results', type=str, default='results.txt',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.test, args.batch, args.results)

if __name__ == "__main__":
    main()