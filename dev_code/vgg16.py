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

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (VGG16 input size)
    transforms.ToTensor(),  # Convert image to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

class CustomImageDataset(Dataset):
    """Custom dataset class to load images from a directory."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'1A': 1, '1G': 0}  # Explicit mapping for classification
        
        # Iterate through each class folder and collect image paths
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
        return len(self.samples)  # Return total number of samples
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
            
        return image, label  # Return image tensor and corresponding label


def load_data(data_dir, batch_size=32):
    """Load dataset and create training/validation data loaders."""
    dataset = CustomImageDataset(root_dir=data_dir, transform=transform)
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Total images: {len(dataset)}")
    
    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Label mapping: 1A -> 1, 1G -> 0\n")
    
    return train_loader, val_loader

class LungCancerVGG16Fusion(nn.Module):
    """Modified VGG16 model for lung cancer classification with a fusion layer."""
    def __init__(self):
        super(LungCancerVGG16Fusion, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Freeze the first 20 convolutional layers to retain pre-trained features
        for param in vgg16.features[:20].parameters():
            param.requires_grad = False
            
        self.features = vgg16.features  # Feature extractor
        self.avgpool = vgg16.avgpool  # Average pooling layer
        
        # Custom classifier with additional feature fusion layer
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        self.fusion_layer = nn.Linear(1024, 8)  # Feature fusion layer
        self.classifier2 = nn.Linear(8, 2)  # Final classification layer
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        fusion_features = self.fusion_layer(x)  # Extract fusion features
        output = self.classifier2(fusion_features)  # Final prediction
        
        if return_features:
            return output, fusion_features
        return output
    
def calculate_metrics(outputs, labels):
    """Calculate accuracy, precision, recall, and F1-score."""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)  # Get class predictions
    
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    
    return accuracy * 100, f1, precision, recall

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Train the model and evaluate it after each epoch."""
    best_val_acc = 0.0
    master = []  # Store training metrics
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
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
        
        val_outputs_all = torch.cat(val_outputs_all, dim=0)
        val_labels_all = torch.cat(val_labels_all, dim=0)
        val_acc, val_f1, val_precision, val_recall = calculate_metrics(val_outputs_all, val_labels_all)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'vgg16.pth')
            print(f"New best model saved! Accuracy: {val_acc:.2f}%\n")
    
if __name__ == "__main__":
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10
    
    train_loader, val_loader = load_data("dataset_path", batch_size=BATCH_SIZE)
    model = LungCancerVGG16Fusion().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)