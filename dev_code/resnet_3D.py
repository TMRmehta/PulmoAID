import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision.models.video import r3d_18
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import monai.transforms as transforms
import pandas as pd
from torch.nn.parallel import DataParallel
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class LungCTDataset(Dataset):
    """Dataset class for loading and preprocessing lung CT scans."""
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None, 
                 target_slices: int = 48) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.target_slices = target_slices
        self.samples = []
        self.labels = []
        
        # Load samples
        for class_folder in ['1A', '1G']:
            class_path = os.path.join(root_dir, class_folder)
            label = 1 if class_folder == '1A' else 0
            
            if not os.path.exists(class_path):
                raise ValueError(f"Class path does not exist: {class_path}")
            
            for subject in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject)
                if os.path.isdir(subject_path):
                    self.samples.append(subject_path)
                    self.labels.append(label)

    def load_scan(self, path: str) -> np.ndarray:
        """Load and preprocess DICOM files from the given path."""
        try:
            slices = []
            for s in sorted(os.listdir(path)):
                if s.endswith('.dcm'):
                    ds = pydicom.dcmread(os.path.join(path, s))
                    slices.append(ds)
            
            if not slices:
                raise ValueError(f"No DICOM files found in {path}")
            
            # Sort by instance number
            slices.sort(key=lambda x: int(x.InstanceNumber))
            scan = np.stack([s.pixel_array for s in slices])
            
            # Resample to target number of slices
            if len(scan) > self.target_slices:
                indices = np.linspace(0, len(scan)-1, self.target_slices, dtype=int)
                scan = scan[indices]
            elif len(scan) < self.target_slices:
                pad_width = self.target_slices - len(scan)
                scan = np.pad(scan, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
            
            return scan
            
        except Exception as e:
            logging.error(f"Error loading scan from {path}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        scan_path = self.samples[idx]
        label = self.labels[idx]
        
        scan = self.load_scan(scan_path)
        if self.transform:
            scan = self.transform(scan)
        
        return scan, label

class ResNet3DModel(nn.Module):
    """3D ResNet model with optional fusion layer."""
    
    def __init__(self, 
                 num_classes: int = 2, 
                 fusion: bool = False, 
                 dropout_rate: float = 0.5) -> None:
        super(ResNet3DModel, self).__init__()
        self.resnet = r3d_18(pretrained=True)
        self.fusion = fusion
        
        # Modify first conv layer to accept different input channels
        self.resnet.stem[0] = nn.Conv3d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        
        # Add batch normalization and dropout
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fusion layers
        self.fusion_features = nn.Linear(512, 8)
        self.resnet.fc = nn.Linear(8 if fusion else 512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.stem(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)
        x = self.dropout(x)
        
        if self.fusion:
            fusion_features = self.fusion_features(x)
            class_logits = self.resnet.fc(fusion_features)
            return class_logits, fusion_features.unsqueeze(0)
        else:
            class_logits = self.resnet.fc(x)
            return class_logits

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


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                max_grad_norm: float) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels)
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f'Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, f1, precision, recall = calculate_metrics(all_outputs, all_labels)
    
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
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, f1, precision, recall = calculate_metrics(all_outputs, all_labels)
    
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
    """Train the model."""
    device = config['device']
    num_epochs = config['num_epochs']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    
    best_val_acc = 0
    patience_counter = 0
    master = []
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config['max_grad_norm'])
        logging.info(f"Training metrics: {train_metrics}")
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        logging.info(f"Validation metrics: {val_metrics}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])
        
        # Early stopping and model saving
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_resnet_3d.pth'))
            logging.info("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info("Early stopping triggered")
                break
        
        # Record metrics
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
    
    # Save metrics to CSV
    df = pd.DataFrame(master)
    df.to_csv(os.path.join(config['metrics_dir'], 'resnet3d_metrics2.csv'), index=False)
    logging.info(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")

def main():
    # Configuration
    config = {
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_workers': 4,
        'patience': 2,
        'max_grad_norm': 1.0,
        'checkpoint_dir': 'checkpoints',
        'metrics_dir': 'metrics',
        'data_dir': r'A:\Software Projects\NLST-Dataset\set1_batch1'
    }
    
    # Create necessary directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['metrics_dir'], exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(0)
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.ScaleIntensity(),
        transforms.Resize((32, 64, 64)),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.ScaleIntensity(),
        transforms.Resize((32, 64, 64)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    try:
        dataset = LungCTDataset(root_dir=config['data_dir'], transform=train_transform)
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            num_workers=config['num_workers']
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
            num_workers=config['num_workers']
        )
        
        # Initialize model
        model = ResNet3DModel(fusion=False)
        
        # Train model
        train_model(model, train_loader, val_loader, config)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()