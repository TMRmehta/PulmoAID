import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from typing import Tuple, Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Resize3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, volume):
        volume = volume.unsqueeze(0)
        volume = F.interpolate(
            volume,
            size=self.size,
            mode='trilinear',
            align_corners=False
        )
        return volume.squeeze(0)


class LungCTDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    @classmethod
    def create_datasets(cls, data_dir, transform=None, train_ratio=0.8, random_seed=42):
        data_dir = Path(data_dir)
        all_samples = []
        
        print(f"Looking for data in: {data_dir}")
        
        # Collect infected samples
        infected_dir = data_dir / '1A'
        infected_samples = []
        if infected_dir.exists():
            for item in infected_dir.glob("*"):
                if item.is_dir():
                    if item.exists():
                        infected_samples.append((item, 1))
        
        # Collect healthy samples
        healthy_dir = data_dir / '1G'
        healthy_samples = []
        if healthy_dir.exists():
            for item in healthy_dir.glob("*"):
                if item.is_dir():
                    if item.exists():
                        healthy_samples.append((item, 0))
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Shuffle both lists
        random.shuffle(infected_samples)
        random.shuffle(healthy_samples)
        
        # Split each class separately
        infected_train_size = int(len(infected_samples) * train_ratio)
        healthy_train_size = int(len(healthy_samples) * train_ratio)
        
        # Create train and test sets while maintaining class distribution
        train_samples = (infected_samples[:infected_train_size] + 
                        healthy_samples[:healthy_train_size])
        test_samples = (infected_samples[infected_train_size:] + 
                       healthy_samples[healthy_train_size:])
        
        # Shuffle the combined sets
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        
        # Print statistics
        print("\nDataset Split Statistics:")
        print(f"Total samples: {len(infected_samples) + len(healthy_samples)}")
        print(f"Train samples: {len(train_samples)}")
        print(f"  Infected: {sum(1 for _, label in train_samples if label == 1)}")
        print(f"  Healthy: {sum(1 for _, label in train_samples if label == 0)}")
        print(f"Test samples: {len(test_samples)}")
        print(f"  Infected: {sum(1 for _, label in test_samples if label == 1)}")
        print(f"  Healthy: {sum(1 for _, label in test_samples if label == 0)}")
        
        # Create and return datasets
        train_dataset = cls(train_samples, transform)
        test_dataset = cls(test_samples, transform)
        
        return train_dataset, test_dataset
    
    def load_scan(self, path):
        """Load and preprocess DICOM files from a patient's images directory"""
        slices = []
        for dcm_path in sorted(path.glob("*.dcm")):
            try:
                # Using dcmread instead of read_file
                dcm = pydicom.dcmread(dcm_path)
                slices.append(dcm.pixel_array)
            except Exception as e:
                print(f"Error loading {dcm_path}: {e}")
                continue
        
        if not slices:
            raise ValueError(f"No valid DICOM files found in {path}")
        
        volume = np.stack(slices)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        return torch.FloatTensor(volume).unsqueeze(0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        images_dir, label = self.samples[idx]
        try:
            volume = self.load_scan(images_dir)
            if self.transform:
                volume = self.transform(volume)
            return volume, label
        except Exception as e:
            print(f"Error loading sample {images_dir}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class Lung3DCNN(nn.Module):
    def __init__(self, in_channels=1, fusion=False):
        super(Lung3DCNN, self).__init__()
        self.fusion = fusion
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Fusion layer
        if self.fusion:
            self.fusion_layer = nn.Sequential(
                nn.Linear(128 * 4 * 4 * 4, 8),
                nn.ReLU()
            )
            
            self.classifier = nn.Linear(8, 2)
        
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        if self.fusion:
            fusion_features = self.fusion_layer(x)
            output = self.classifier(fusion_features)
            if self.training:
                return output
            
            else:
                return output, fusion_features
       
        else:
            return self.classifier(x)


def calculate_metrics(outputs: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[float, float, float, float]:
    """Calculate accuracy, F1, precision, and recall metrics."""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)
    
    # Convert to numpy for metric calculation
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    
    return accuracy * 100, f1, precision, recall


def train_model(train_loader, test_loader, model, num_epochs=10, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics
    metrics_history = []
    best_test_acc = 0.0
    
    save_dir = Path('model_checkpoints')
    save_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_outputs = []
        epoch_labels = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_outputs.append(outputs.detach())
            epoch_labels.append(labels)
            
            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_outputs = torch.cat(epoch_outputs)
        train_labels = torch.cat(epoch_labels)
        train_acc, train_f1, train_precision, train_recall = calculate_metrics(train_outputs, train_labels)
        train_loss = running_loss / len(train_loader)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_outputs = []
        test_labels = []
        fusion_features_list = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if model.fusion:
                    outputs, fusion_features = model(inputs)
                    fusion_features_list.append(fusion_features)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_outputs.append(outputs)
                test_labels.append(labels)
        
        # Calculate test metrics
        test_outputs = torch.cat(test_outputs)
        test_labels = torch.cat(test_labels)
        test_acc, test_f1, test_precision, test_recall = calculate_metrics(test_outputs, test_labels)
        test_loss = test_loss / len(test_loader)
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        
        metrics_history.append(epoch_metrics)

        df = pd.DataFrame(metrics_history)
        df.to_csv(r'A:\Software Projects\NLST-App\metrics\3d_cnn_metrics.csv')
        
        # Print epoch results
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}')
        print(f'Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics
            }
            
            if model.fusion and fusion_features_list:
                save_dict['fusion_features'] = torch.cat(fusion_features_list).cpu()
                
            torch.save(save_dict, save_dir / 'best_3d_cnn.pth')
    
    return metrics_history


if __name__ == "__main__":
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the transform
    transform = transforms.Compose([
        Resize3D((32, 64, 64))
    ])

    # Create datasets and dataloaders
    train_dataset, test_dataset = LungCTDataset.create_datasets(
        data_dir=r'A:\Software Projects\NLST-Dataset\set1_batch1',
        transform=transform,
        train_ratio=0.8
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,           
        shuffle=True,
        num_workers=2,          
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,           
        shuffle=False,
        num_workers=2,          
        pin_memory=True if torch.cuda.is_available() else False
    )

    try:
        model = Lung3DCNN(in_channels=1, fusion=False)
        print("Model initialized with fusion layer")

        # Test data loading
        print("Testing data loading...")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Input shape: {inputs.shape}, Labels: {labels}")
            if batch_idx == 0:
                break
        print("Data loading test successful!")

        # Start training
        metrics_history = train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            num_epochs=10,
            learning_rate=0.001,
            device=device
        )

        # Print final metrics
        print("\nTraining completed. Final metrics:")
        final_metrics = metrics_history[-1]
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
