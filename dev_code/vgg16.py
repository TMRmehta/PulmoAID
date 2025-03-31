import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import time
from PIL import Image
import pydicom
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import os
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'1A': 1, '1G': 0}  # Explicit mapping
        
        # Walk through directory and collect samples
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_data(data_dir=r"A:\Software Projects\NLST-Dataset\images_all", batch_size=32):
    # Create custom dataset with relabeled classes
    dataset = CustomImageDataset(root_dir=data_dir, transform=transform)
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Total images: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Label mapping: 1A -> 1, 1G -> 0\n")
    
    return train_loader, val_loader


class LungCancerVGG16Fusion(nn.Module):
    def __init__(self):
        super(LungCancerVGG16Fusion, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in vgg16.features[:20].parameters():
            param.requires_grad = False
            
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        # New fusion layer ----------------------------------------------------------------------
        self.fusion_layer = nn.Linear(1024, 8)
        # ---------------------------------------------------------------------------------------
        
        self.classifier2 = nn.Linear(8, 2)
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        
        # Get fusion features
        fusion_features = self.fusion_layer(x)
        
        # Get final prediction
        output = self.classifier2(fusion_features)
        
        if return_features:
            return output, fusion_features
        return output
    

def calculate_metrics(outputs: torch.Tensor, 
                     labels: torch.Tensor):
    """Calculate accuracy, F1, precision, and recall metrics."""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)
    
    # Convert to numpy for metric calculation
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    
    return accuracy * 100, f1, precision, recall


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0
    master = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        train_preds = []
        train_labels = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training metrics batch by batch
            train_acc, train_f1, train_precision, train_recall = calculate_metrics(outputs, labels)
            
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            
            progress = f"Epoch {epoch+1}/{num_epochs} [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}"
            print(progress, end='\r')
            sys.stdout.flush()
        
        model.eval()
        val_loss = 0.0
        val_outputs_all = []
        val_labels_all = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_outputs_all.append(outputs)
                val_labels_all.append(labels)
        
        # Concatenate all validation predictions and labels
        val_outputs_all = torch.cat(val_outputs_all, dim=0)
        val_labels_all = torch.cat(val_labels_all, dim=0)
        
        # Calculate validation metrics
        val_acc, val_f1, val_precision, val_recall = calculate_metrics(val_outputs_all, val_labels_all)
        
        epoch_time = time.time() - start_time
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics in dictionary
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            # 'epoch_time': epoch_time
        }
        
        # Append metrics to master list
        master.append(epoch_metrics)
        
        print(" " * 80, end='\r')
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        print(f"Training F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")
        print(f"Training Precision: {train_precision:.4f}, Validation Precision: {val_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}, Validation Recall: {val_recall:.4f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), r'A:\Software Projects\NLST-App\checkpoints\vgg16.pth')
            print(f"New best model saved! Accuracy: {val_acc:.2f}%\n")
    
    data = pd.DataFrame(master)
    data.to_csv(r'A:\Software Projects\NLST-App\metrics\vgg16_metrics.csv', index=False)


if __name__ == "__main__":

    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10

    # Load data
    data_dir = r"A:\Software Projects\NLST-Dataset\images_all" 
    train_loader, val_loader = load_data(batch_size=BATCH_SIZE)
    model = LungCancerVGG16Fusion().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {device}")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
