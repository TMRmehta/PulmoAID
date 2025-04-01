import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pydicom  # For reading DICOM medical image files
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from typing import Tuple, Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom transform to resize 3D volumes to specified dimensions
class Resize3D:
    def __init__(self, size):
        """
        Initialize 3D resize transform
        Args:
            size: Target dimensions (depth, height, width)
        """
        self.size = size

    def __call__(self, volume):
        """
        Resize a 3D volume using trilinear interpolation
        Args:
            volume: Input 3D tensor [C, D, H, W]
        Returns:
            Resized volume with dimensions matching self.size
        """
        volume = volume.unsqueeze(0)  # Add batch dimension [1, C, D, H, W]
        volume = F.interpolate(
            volume,
            size=self.size,
            mode='trilinear',  # 3D interpolation method
            align_corners=False
        )
        return volume.squeeze(0)  # Remove batch dimension [C, D, H, W]


# Custom dataset for loading and processing lung CT scans
class LungCTDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Initialize the dataset
        Args:
            samples: List of tuples (directory_path, label)
            transform: Optional transforms to apply to the 3D volumes
        """
        self.samples = samples
        self.transform = transform
    
    @classmethod
    def create_datasets(cls, data_dir, transform=None, train_ratio=0.8, random_seed=42):
        """
        Factory method to create train and test datasets from a single data directory
        Args:
            data_dir: Base directory containing the dataset
            transform: Optional transforms to apply
            train_ratio: Ratio of data to use for training (default: 0.8)
            random_seed: Random seed for reproducibility
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        data_dir = Path(data_dir)
        all_samples = []
        
        print(f"Looking for data in: {data_dir}")
        
        # Collect infected (cancer positive) samples from '1A' directory
        infected_dir = data_dir / '1A'
        infected_samples = []
        if infected_dir.exists():
            for item in infected_dir.glob("*"):
                if item.is_dir():
                    if item.exists():
                        infected_samples.append((item, 1))  # Label 1 for cancer positive
        
        # Collect healthy (cancer negative) samples from '1G' directory
        healthy_dir = data_dir / '1G'
        healthy_samples = []
        if healthy_dir.exists():
            for item in healthy_dir.glob("*"):
                if item.is_dir():
                    if item.exists():
                        healthy_samples.append((item, 0))  # Label 0 for cancer negative
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Shuffle both lists of samples separately
        random.shuffle(infected_samples)
        random.shuffle(healthy_samples)
        
        # Split each class separately to maintain class distribution in train/test sets
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
        
        # Print dataset statistics for verification
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
        """
        Load and preprocess DICOM files from a patient's images directory
        Args:
            path: Directory containing DICOM files for a patient
        Returns:
            Normalized tensor volume [1, D, H, W] where D is the number of slices
        """
        slices = []
        # Sort DICOM files to maintain correct slice order
        for dcm_path in sorted(path.glob("*.dcm")):
            try:
                # Read DICOM file
                dcm = pydicom.dcmread(dcm_path)
                slices.append(dcm.pixel_array)  # Extract pixel data
            except Exception as e:
                print(f"Error loading {dcm_path}: {e}")
                continue
        
        if not slices:
            raise ValueError(f"No valid DICOM files found in {path}")
        
        # Stack all slices to create 3D volume
        volume = np.stack(slices)
        # Normalize to [0,1] range for better training stability
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Convert to PyTorch tensor with channel dimension [1, D, H, W]
        return torch.FloatTensor(volume).unsqueeze(0)
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample by index
        Args:
            idx: Sample index
        Returns:
            Tuple of (volume, label)
        """
        images_dir, label = self.samples[idx]
        try:
            # Load and preprocess the CT scan
            volume = self.load_scan(images_dir)
            if self.transform:
                volume = self.transform(volume)
            return volume, label
        except Exception as e:
            print(f"Error loading sample {images_dir}: {e}")
            # Handle error by moving to next sample (with wraparound)
            return self.__getitem__((idx + 1) % len(self))


# 3D CNN model for lung cancer classification
class Lung3DCNN(nn.Module):
    def __init__(self, in_channels=1, fusion=False):
        """
        Initialize the 3D CNN model
        Args:
            in_channels: Number of input channels (1 for grayscale CT)
            fusion: Whether to use feature fusion mode (for transfer learning/feature extraction)
        """
        super(Lung3DCNN, self).__init__()
        self.fusion = fusion
        
        # 3D Convolutional layers with BatchNorm and ReLU activation
        self.conv_layers = nn.Sequential(
            # First conv block: 1→32 channels, preserves spatial dimensions with padding
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),  # Normalizes activations for stable training
            nn.ReLU(),  # Non-linear activation
            nn.MaxPool3d(kernel_size=2),  # Downsamples by factor of 2
            
            # Second conv block: 32→64 channels
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # Further downsampling
            
            # Third conv block: 64→128 channels
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # Final downsampling
        )
        
        # Adaptive pooling ensures fixed output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Different classifier heads based on fusion mode
        if self.fusion:
            # Compact 8-dimensional feature representation for fusion/transfer
            self.fusion_layer = nn.Sequential(
                nn.Linear(128 * 4 * 4 * 4, 8),  # Reduce to 8 features
                nn.ReLU()
            )
            
            # Final classifier from fusion features to 2 classes (positive/negative)
            self.classifier = nn.Linear(8, 2)
        
        else:
            # Standard classifier with more capacity
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4 * 4, 512),  # First dense layer
                nn.ReLU(),
                nn.Dropout(0.5),  # Prevents overfitting
                nn.Linear(512, 2)  # Output layer (2 classes)
            )
        
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor [B, C, D, H, W]
        Returns:
            Classification logits or (logits, fusion_features) if in evaluation mode with fusion
        """
        # Pass through convolutional layers
        x = self.conv_layers(x)
        # Fix output spatial dimensions
        x = self.adaptive_pool(x)
        # Flatten for dense layers
        x = x.view(x.size(0), -1)
        
        if self.fusion:
            # Extract compact feature representation
            fusion_features = self.fusion_layer(x)
            # Final classification
            output = self.classifier(fusion_features)
            
            if self.training:
                # During training, just return predictions
                return output
            else:
                # During evaluation, also return feature vectors for analysis
                return output, fusion_features
       
        else:
            # Standard classification without feature extraction
            return self.classifier(x)


# Function to calculate performance metrics from model outputs
def calculate_metrics(outputs: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, F1, precision, and recall metrics for binary classification
    Args:
        outputs: Model output logits [B, 2]
        labels: Ground truth labels [B]
    Returns:
        Tuple of (accuracy, f1, precision, recall)
    """
    # Convert logits to probabilities
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    # Get predicted class (highest probability)
    _, predicted = torch.max(outputs.data, 1)
    
    # Convert to numpy for metric calculation
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    
    # Return metrics as percentages/decimals
    return accuracy * 100, f1, precision, recall


# Main training function
def train_model(train_loader, test_loader, model, num_epochs=10, learning_rate=0.001, device='cuda'):
    """
    Train the 3D CNN model
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation/test data
        model: Model to train
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
    Returns:
        List of metrics for each epoch
    """
    # Move model to device (GPU/CPU)
    model = model.to(device)
    # Define loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()
    # Define optimizer (Adam with specified learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics
    metrics_history = []
    best_test_acc = 0.0
    
    # Create directory for model checkpoints
    save_dir = Path('model_checkpoints')
    save_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # ------ Training phase ------
        model.train()  # Set model to training mode
        running_loss = 0.0
        epoch_outputs = []
        epoch_labels = []
        
        # Iterate through mini-batches
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients before backward pass
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
            # Accumulate loss and predictions for epoch metrics
            running_loss += loss.item()
            epoch_outputs.append(outputs.detach())
            epoch_labels.append(labels)
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # ------ Calculate training metrics ------
        train_outputs = torch.cat(epoch_outputs)
        train_labels = torch.cat(epoch_labels)
        train_acc, train_f1, train_precision, train_recall = calculate_metrics(train_outputs, train_labels)
        train_loss = running_loss / len(train_loader)
        
        # ------ Testing/validation phase ------
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_outputs = []
        test_labels = []
        fusion_features_list = []
        
        # No gradient computation during evaluation
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Handle fusion vs. non-fusion model architectures
                if model.fusion:
                    # Get both outputs and feature vectors
                    outputs, fusion_features = model(inputs)
                    fusion_features_list.append(fusion_features)
                else:
                    # Get only class predictions
                    outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_outputs.append(outputs)
                test_labels.append(labels)
        
        # ------ Calculate test metrics ------
        test_outputs = torch.cat(test_outputs)
        test_labels = torch.cat(test_labels)
        test_acc, test_f1, test_precision, test_recall = calculate_metrics(test_outputs, test_labels)
        test_loss = test_loss / len(test_loader)
        
        # ------ Store metrics ------
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

        # Save metrics to CSV for later analysis
        df = pd.DataFrame(metrics_history)
        df.to_csv(r'A:\Software Projects\NLST-App\metrics\3d_cnn_metrics.csv')
        
        # Print epoch results
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}')
        print(f'Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}')
        
        # ------ Save best model ------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Create save dictionary with model state and metrics
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics
            }
            
            # If using fusion model, also save extracted features
            if model.fusion and fusion_features_list:
                save_dict['fusion_features'] = torch.cat(fusion_features_list).cpu()
                
            # Save model checkpoint
            torch.save(save_dict, save_dir / 'best_3d_cnn.pth')
    
    return metrics_history


# Main execution block
if __name__ == "__main__":
    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the transform pipeline to resize volumes to fixed dimensions
    transform = transforms.Compose([
        Resize3D((32, 64, 64))  # Resize to D=32, H=64, W=64
    ])

    # Create datasets and dataloaders
    train_dataset, test_dataset = LungCTDataset.create_datasets(
        data_dir=r'A:\Software Projects\NLST-Dataset\set1_batch1',  # Path to data
        transform=transform,
        train_ratio=0.8  # 80% training, 20% testing
    )

    # Create training data loader with batching and multiprocessing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,           # Process 8 samples at once
        shuffle=True,           # Shuffle data for each epoch
        num_workers=2,          # Use 2 CPU threads for data loading
        pin_memory=True if torch.cuda.is_available() else False  # Speed up CPU to GPU transfers
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,           
        shuffle=False,          # No need to shuffle test data
        num_workers=2,          
        pin_memory=True if torch.cuda.is_available() else False
    )

    try:
        # Initialize model (non-fusion version)
        model = Lung3DCNN(in_channels=1, fusion=False)
        print("Model initialized with fusion layer")

        # Test data loading to catch any issues early
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
            num_epochs=10,      # Train for 10 epochs
            learning_rate=0.001,  # Adam learning rate
            device=device
        )

        # Print final metrics
        print("\nTraining completed. Final metrics:")
        final_metrics = metrics_history[-1]
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")