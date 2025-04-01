import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom  # For reading DICOM medical image files
from torchvision.models.video import r3d_18  # Pre-trained 3D ResNet model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import monai.transforms as transforms  # MONAI is a healthcare imaging library
import pandas as pd
from torch.nn.parallel import DataParallel  # For multi-GPU training
import random

# Set up logging configuration to track training progress and errors
# Logs will be saved to 'training.log' file and also displayed in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across all random number generators.
    
    This ensures that running the code multiple times produces the same results,
    which is crucial for research and debugging.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Makes cudnn deterministic (slower but more reproducible)

class LungCTDataset(Dataset):
    """
    Dataset class for loading and preprocessing lung CT scans from DICOM files.
    
    This class handles loading CT scan data from a directory structure organized by class:
    - '1A' folder contains positive samples (label 1)
    - '1G' folder contains negative samples (label 0)
    
    Each subfolder contains patient-specific DICOM files representing CT slices.
    """
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None, 
                 target_slices: int = 48) -> None:
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class folders
            transform: Composition of transforms to apply to the scans
            target_slices: Number of slices to standardize each scan to
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_slices = target_slices
        self.samples = []  # Will store paths to individual scans
        self.labels = []   # Will store corresponding labels
        
        # Load samples by iterating through class directories
        for class_folder in ['1A', '1G']:  # '1A' = positive class (1), '1G' = negative class (0)
            class_path = os.path.join(root_dir, class_folder)
            label = 1 if class_folder == '1A' else 0
            
            if not os.path.exists(class_path):
                raise ValueError(f"Class path does not exist: {class_path}")
            
            # Iterate through subject folders within each class
            for subject in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject)
                if os.path.isdir(subject_path):
                    self.samples.append(subject_path)
                    self.labels.append(label)

    def load_scan(self, path: str) -> np.ndarray:
        """
        Load and preprocess DICOM files from the given path.
        
        This method:
        1. Loads all DICOM files in the specified path
        2. Sorts them by instance number (slice position)
        3. Resamples to a standard number of slices
        
        Args:
            path: Path to folder containing DICOM files
            
        Returns:
            Preprocessed 3D volume as numpy array
        """
        try:
            slices = []
            for s in sorted(os.listdir(path)):
                if s.endswith('.dcm'):
                    ds = pydicom.dcmread(os.path.join(path, s))
                    slices.append(ds)
            
            if not slices:
                raise ValueError(f"No DICOM files found in {path}")
            
            # Sort by instance number to ensure correct slice order
            slices.sort(key=lambda x: int(x.InstanceNumber))
            scan = np.stack([s.pixel_array for s in slices])
            
            # Standardize number of slices through resampling or padding
            if len(scan) > self.target_slices:
                # If too many slices, sample at regular intervals
                indices = np.linspace(0, len(scan)-1, self.target_slices, dtype=int)
                scan = scan[indices]
            elif len(scan) < self.target_slices:
                # If too few slices, pad with zeros
                pad_width = self.target_slices - len(scan)
                scan = np.pad(scan, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
            
            return scan
            
        except Exception as e:
            logging.error(f"Error loading scan from {path}: {str(e)}")
            raise

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a specific sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing the processed scan and its label
        """
        scan_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and transform the scan
        scan = self.load_scan(scan_path)
        if self.transform:
            scan = self.transform(scan)
        
        return scan, label

class ResNet3DModel(nn.Module):
    """
    3D ResNet model for lung CT scan classification.
    
    This model uses a pre-trained 3D ResNet-18 as backbone and modifies it for:
    1. Medical image input characteristics (channel dimension)
    2. Binary classification task
    3. Optional feature fusion for advanced analysis
    """
    
    def __init__(self, 
                 num_classes: int = 2, 
                 fusion: bool = False, 
                 dropout_rate: float = 0.5) -> None:
        """
        Initialize the 3D ResNet model.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            fusion: Whether to use feature fusion layer
            dropout_rate: Dropout probability for regularization
        """
        super(ResNet3DModel, self).__init__()
        self.resnet = r3d_18(pretrained=True)  # Load pre-trained model
        self.fusion = fusion
        
        # Modify first conv layer to accept different input channels
        # Original model expects RGB video (3 channels), we want to use CT scans (48 slices as channels)
        self.resnet.stem[0] = nn.Conv3d(
            in_channels=48,  # Number of input slices/channels
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        
        # Add batch normalization and dropout for regularization
        self.bn = nn.BatchNorm1d(512)  # Normalize features to improve training stability
        self.dropout = nn.Dropout(dropout_rate)  # Reduce overfitting
        
        # Fusion layers for dimensionality reduction (optional)
        self.fusion_features = nn.Linear(512, 8)  # Reduce 512 features to 8
        
        # Final classification layer
        self.resnet.fc = nn.Linear(8 if fusion else 512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, depth, height, width]
            
        Returns:
            Class logits or (class_logits, fusion_features) if fusion=True
        """
        # Pass through ResNet blocks
        x = self.resnet.stem(x)  # Initial convolution and pooling
        x = self.resnet.layer1(x)  # ResNet block 1
        x = self.resnet.layer2(x)  # ResNet block 2
        x = self.resnet.layer3(x)  # ResNet block 3
        x = self.resnet.layer4(x)  # ResNet block 4
        x = self.resnet.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]
        x = self.bn(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        
        if self.fusion:
            # If using fusion, return both classification logits and feature embeddings
            fusion_features = self.fusion_features(x)
            class_logits = self.resnet.fc(fusion_features)
            return class_logits, fusion_features.unsqueeze(0)
        else:
            # Otherwise just return classification logits
            class_logits = self.resnet.fc(x)
            return class_logits

def calculate_metrics(outputs: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Calculate classification performance metrics.
    
    Args:
        outputs: Model logits (before softmax)
        labels: Ground truth labels
        
    Returns:
        Tuple of (accuracy, F1 score, precision, recall)
    """
    # Convert logits to probabilities and get predicted class
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)
    
    # Convert to numpy for metric calculation
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary')
    
    return accuracy * 100, f1, precision, recall


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                max_grad_norm: float) -> dict:
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run computations on (CPU/GPU)
        max_grad_norm: Maximum norm for gradient clipping
        
    Returns:
        Dictionary of training metrics for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    # Iterate through mini-batches
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()  # Update weights
        
        # Accumulate loss and predictions for epoch-level metrics
        total_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels)
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logging.info(f'Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, f1, precision, recall = calculate_metrics(all_outputs, all_labels)
    
    # Return all metrics as a dictionary
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def validate_epoch(model: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> dict:
    """
    Validate the model for one epoch.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run computations on (CPU/GPU)
        
    Returns:
        Dictionary of validation metrics for the epoch
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    # No gradient computation needed for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            
            # Accumulate loss and predictions
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, f1, precision, recall = calculate_metrics(all_outputs, all_labels)
    
    # Return all metrics as a dictionary
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: dict) -> None:
    """
    Train the model over multiple epochs with validation and early stopping.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary containing training parameters
    """
    device = config['device']
    num_epochs = config['num_epochs']
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler that reduces LR when validation accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    # Move model to device (CPU/GPU)
    model = model.to(device)
    
    # Set up data parallelism if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
    
    best_val_acc = 0  # Track best validation accuracy for model saving
    patience_counter = 0  # Counter for early stopping
    master = []  # List to store metrics for all epochs
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config['max_grad_norm'])
        logging.info(f"Training metrics: {train_metrics}")
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        logging.info(f"Validation metrics: {val_metrics}")
        
        # Update learning rate based on validation performance
        scheduler.step(val_metrics['accuracy'])
        
        # Check if this is the best model so far
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0  # Reset patience counter
            
            # Save the best model
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_resnet_3d.pth'))
            logging.info("Saved new best model")
        else:
            # Increment patience counter if no improvement
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info("Early stopping triggered")
                break  # Stop training if no improvement for 'patience' epochs
        
        # Record metrics for this epoch
        master.append({
            'epoch': epoch + 1,
            'train_acc': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall']
        })
    
    # Save training history to CSV
    df = pd.DataFrame(master)
    df.to_csv(os.path.join(config['metrics_dir'], 'resnet3d_metrics2.csv'), index=False)
    logging.info(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")

def main():
    """Main function to run the entire training pipeline."""
    # Configuration dictionary with all hyperparameters and settings
    config = {
        'batch_size': 8,  # Smaller batch size due to 3D data memory requirements
        'num_epochs': 10,  # Maximum number of training epochs
        'learning_rate': 0.001,  # Initial learning rate
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use GPU if available
        'num_workers': 4,  # Number of parallel data loading workers
        'patience': 2,  # Early stopping patience (epochs without improvement)
        'max_grad_norm': 1.0,  # Maximum gradient norm for clipping
        'checkpoint_dir': 'checkpoints',  # Directory to save model checkpoints
        'metrics_dir': 'metrics',  # Directory to save training metrics
        'data_dir': r'A:\Software Projects\NLST-Dataset\set1_batch1'  # Path to dataset
    }
    
    # Create necessary directories if they don't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['metrics_dir'], exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(0)
    
    # Create transforms for data preprocessing
    # Training transform pipeline
    train_transform = transforms.Compose([
        transforms.ScaleIntensity(),  # Normalize intensity values
        transforms.Resize((32, 64, 64)),  # Resize to standard dimensions
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])
    
    # Validation transform pipeline (could add data augmentation to train_transform later)
    val_transform = transforms.Compose([
        transforms.ScaleIntensity(),  # Normalize intensity values
        transforms.Resize((32, 64, 64)),  # Resize to standard dimensions
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])
    
    try:
        # Create dataset
        dataset = LungCTDataset(root_dir=config['data_dir'], transform=train_transform)
        
        # Split into training and validation sets
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        
        # Create data loaders with appropriate samplers
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),  # Sample from training indices
            num_workers=config['num_workers']  # Parallel data loading
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),  # Sample from validation indices
            num_workers=config['num_workers']  # Parallel data loading
        )
        
        # Initialize model
        model = ResNet3DModel(fusion=False)  # Not using feature fusion in this run
        
        # Train model with all the configured parameters
        train_model(model, train_loader, val_loader, config)
        
    except Exception as e:
        # Log any exceptions that occur during training
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()