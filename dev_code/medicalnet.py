import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter('ignore')
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def get_inplanes():
    return [64, 128, 256, 512]  # Restored to original size for better feature extraction

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout3d(p=0.3)  # Increased dropout

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 fusion=False):  # Added fusion parameter
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.fusion = fusion  # Store fusion parameter
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # Modified first conv layer for medical images
        self.conv1 = nn.Conv3d(n_input_channels,
                              self.in_planes,
                              kernel_size=(conv1_t_size, 7, 7),
                              stride=(conv1_t_stride, 2, 2),
                              padding=(conv1_t_size // 2, 3, 3),
                              bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Added dropout after initial layers
        self.dropout = nn.Dropout3d(p=0.2)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                     shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1],
                                     shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2],
                                     shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3],
                                     shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout_final = nn.Dropout(p=0.5)
        
        # Added fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(block_inplanes[3] * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 8)
        )
        
        # Modified final classification layer
        self.fc = nn.Sequential(
            nn.Linear(8, n_classes)  # Changed input size to match fusion layer output
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                      mode='fan_out',
                                      nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                   planes=planes * block.expansion,
                                   stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_final(x)
        
        # Get fusion features
        fusion_features = self.fusion_layer(x)
        
        # Get final prediction
        out = self.fc(fusion_features)
        
        if self.fusion:
            return fusion_features, out
        return out
    

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)

    return model


class LungCTDataset(Dataset):
    def __init__(self, samples, transform=None, target_shape=(64, 128, 128)):
        self.samples = samples
        self.transform = transform
        self.target_shape = target_shape
    
    @classmethod
    def create_datasets(cls, data_dir, transform=None, train_ratio=0.8, val_ratio=0.1, 
                       random_seed=42, target_shape=(64, 128, 128)):
        data_dir = Path(data_dir)
        
        infected_samples = [(item, 1) for item in (data_dir / '1A').glob("*") 
                          if item.is_dir() and item.exists()]
        healthy_samples = [(item, 0) for item in (data_dir / '1G').glob("*") 
                          if item.is_dir() and item.exists()]
        
        # Calculate class weights
        total_samples = len(infected_samples) + len(healthy_samples)
        class_weights = {
            0: total_samples / (2 * len(healthy_samples)),
            1: total_samples / (2 * len(infected_samples))
        }
        print(f"Class weights: {class_weights}")
        
        random.seed(random_seed)
        random.shuffle(infected_samples)
        random.shuffle(healthy_samples)
        
        infected_train_size = int(len(infected_samples) * train_ratio)
        infected_val_size = int(len(infected_samples) * val_ratio)
        healthy_train_size = int(len(healthy_samples) * train_ratio)
        healthy_val_size = int(len(healthy_samples) * val_ratio)
        
        train_samples = (infected_samples[:infected_train_size] + 
                        healthy_samples[:healthy_train_size])
        val_samples = (infected_samples[infected_train_size:infected_train_size+infected_val_size] + 
                      healthy_samples[healthy_train_size:healthy_train_size+healthy_val_size])
        test_samples = (infected_samples[infected_train_size+infected_val_size:] + 
                       healthy_samples[healthy_train_size+healthy_val_size:])
        
        for split in [train_samples, val_samples, test_samples]:
            random.shuffle(split)
        
        print("\nDataset Split Statistics:")
        print(f"Total samples: {len(infected_samples) + len(healthy_samples)}")
        for name, split in [("Train", train_samples), ("Validation", val_samples), 
                          ("Test", test_samples)]:
            print(f"{name} samples: {len(split)}")
            print(f"  Infected: {sum(1 for _, label in split if label == 1)}")
            print(f"  Healthy: {sum(1 for _, label in split if label == 0)}")
        
        return (cls(split, transform, target_shape) for split in 
                [train_samples, val_samples, test_samples]), class_weights
    
    def load_scan(self, path):
        """Enhanced DICOM loading with preprocessing"""
        slices = []
        dcm_files = sorted(path.glob("*.dcm"), 
                          key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        
        for dcm_path in dcm_files:
            try:
                dcm = pydicom.dcmread(dcm_path)
                slice_array = dcm.pixel_array.astype(float)
                
                # Convert to Hounsfield Units
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    slice_array = slice_array * dcm.RescaleSlope + dcm.RescaleIntercept
                
                # Window for lung tissue
                min_hu = -1000
                max_hu = 400
                slice_array = np.clip(slice_array, min_hu, max_hu)
                
                # Normalize to [0,1]
                slice_array = (slice_array - min_hu) / (max_hu - min_hu)
                
                # Simple lung segmentation
                binary_mask = slice_array > 0.2
                slice_array = slice_array * binary_mask
                
                slices.append(slice_array)
            except Exception as e:
                print(f"Error loading {dcm_path}: {e}")
                continue
        
        if not slices:
            raise ValueError(f"No valid DICOM files found in {path}")
        
        volume = np.stack(slices)
        volume = self._resize_volume(volume, self.target_shape)
        
        return torch.FloatTensor(volume).unsqueeze(0)
    
    def _resize_volume(self, volume, target_shape):
        from scipy.ndimage import zoom
        current_depth, current_height, current_width = volume.shape
        depth_factor = target_shape[0] / current_depth
        width_factor = target_shape[1] / current_width
        height_factor = target_shape[2] / current_height
        
        return zoom(volume, (depth_factor, height_factor, width_factor), order=1)
    

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


class MetricsTracker:
    def __init__(self):
        self.epoch_metrics = []
        self.current_epoch_batches = []
        
    def update_batch_metrics(self, outputs, labels):
        """Store batch predictions and labels for epoch-level calculation"""
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        self.current_epoch_batches.append({
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu(),
            'probs': probs[:, 1].detach().cpu()  # Store positive class probabilities
        })
    
    def calculate_epoch_metrics(self, epoch, phase='train', loss=None):
        """Calculate metrics for entire epoch"""
        all_preds = torch.cat([batch['preds'] for batch in self.current_epoch_batches])
        all_labels = torch.cat([batch['labels'] for batch in self.current_epoch_batches])
        all_probs = torch.cat([batch['probs'] for batch in self.current_epoch_batches])
        
        # Convert to numpy once for all calculations
        np_preds = all_preds.numpy()
        np_labels = all_labels.numpy()
        np_probs = all_probs.numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(np_labels, np_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(np_labels, np_preds, average='binary', zero_division=0)
        auc_roc = roc_auc_score(np_labels, np_probs)
        
        metrics = {
            'epoch': epoch,
            'phase': phase,
            'accuracy': accuracy * 100,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc_roc
        }
        
        if loss is not None:
            metrics['loss'] = loss
            
        self.epoch_metrics.append(metrics)
        self.current_epoch_batches = []  # Reset for next epoch
        
        return metrics
    
    def get_best_metrics(self, metric='f1'):
        """Get the best performance based on specified metric"""
        return max(self.epoch_metrics, key=lambda x: x[metric])
    
    def save_metrics_history(self, filepath):
        """Save metrics history to CSV"""
        import pandas as pd
        pd.DataFrame(self.epoch_metrics).to_csv(filepath, index=False)


class Trainer:
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.metrics_tracker = MetricsTracker()
        
    @torch.no_grad()
    def calculate_batch_metrics(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Fast batch metrics calculation using PyTorch operations"""
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy using PyTorch
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = (correct / total) * 100
        
        return accuracy, probs
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        samples_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1} Training") as pbar:
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                samples_count += batch_size
                
                # Track batch metrics
                self.metrics_tracker.update_batch_metrics(outputs, labels)
                
                # Update progress bar
                batch_accuracy, _ = self.calculate_batch_metrics(outputs, labels)
                pbar.set_postfix({
                    'loss': total_loss/samples_count,
                    'acc': batch_accuracy
                })
        
        avg_loss = total_loss / samples_count
        metrics = self.metrics_tracker.calculate_epoch_metrics(epoch, 'train', avg_loss)
        
        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        samples_count = 0
        
        with tqdm(val_loader, desc=f"Epoch {epoch+1} Validation") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Update metrics
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                samples_count += batch_size
                
                # Track batch metrics
                self.metrics_tracker.update_batch_metrics(outputs, labels)
                
                # Update progress bar
                batch_accuracy, _ = self.calculate_batch_metrics(outputs, labels)
                pbar.set_postfix({
                    'loss': total_loss/samples_count,
                    'acc': batch_accuracy
                })
        
        avg_loss = total_loss / samples_count
        metrics = self.metrics_tracker.calculate_epoch_metrics(epoch, 'val', avg_loss)
        
        return metrics

    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir=r'A:\Software Projects\NLST-App\checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint if best model
            best_metrics = self.metrics_tracker.get_best_metrics()
            if val_metrics['f1'] >= best_metrics['f1']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                }, os.path.join(checkpoint_dir, 'best_medicalnet.pth'))
            
            # Print metrics
            print("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                if isinstance(value, (float, np.float32, np.float64)):
                    print(f"{metric}: {value:.4f}")
            
            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                if isinstance(value, (float, np.float32, np.float64)):
                    print(f"{metric}: {value:.4f}")
        
        # Save final metrics history
        self.metrics_tracker.save_metrics_history(os.path.join(r'A:\Software Projects\NLST-App\metrics', 'metrics_history.csv'))


def setup_training(data_dir, config):
    """Setup all training components"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Create transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        )
    ])
    
    # Create datasets
    (train_dataset, val_dataset, test_dataset), class_weights = LungCTDataset.create_datasets(
        data_dir=data_dir,
        transform=None,
        target_shape=config['target_shape'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = generate_model(
        model_depth=config['model_depth'],
        n_input_channels=1,
        n_classes=2,
        widen_factor=config['widen_factor'],
        fusion=config['fusion']
    ).to(device)
    
    # Convert class weights to tensor
    class_weight_tensor = torch.tensor([class_weights[0], class_weights[1]]).to(device)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        amsgrad=True
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler_T0'],
        T_mult=config['scheduler_T_mult'],
        eta_min=config['scheduler_eta_min']
    )
    
    return {
        'device': device,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

def main():
    # Training configuration
    config = {
        'seed': 0,
        'target_shape': (32, 64, 64),  # (depth, height, width)
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'batch_size': 8,
        'num_workers': 2,
        'model_depth': 18,
        'widen_factor': 1.0,
        'fusion': False,  # Enable fusion layer
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'scheduler_T0': 10,
        'scheduler_T_mult': 2,
        'scheduler_eta_min': 1e-6,
        'num_epochs': 10,
        'checkpoint_dir': r'A:\Software Projects\NLST-App\checkpoints',
    }
    
    # Setup training components
    training_components = setup_training(
        data_dir=r'A:\Software Projects\NLST-Dataset\set1_batch1',  # Replace with your dataset path
        config=config
    )
    
    # Create trainer
    trainer = Trainer(
        model=training_components['model'],
        device=training_components['device'],
        criterion=training_components['criterion'],
        optimizer=training_components['optimizer'],
        scheduler=training_components['scheduler']
    )
    
    # Train model
    print(f"Starting training on {training_components['device']}")
    print("Model architecture:")
    print(training_components['model'])
    
    try:
        trainer.train(
            train_loader=training_components['train_loader'],
            val_loader=training_components['val_loader'],
            num_epochs=config['num_epochs'],
            checkpoint_dir=config['checkpoint_dir']
        )
        
        # Final evaluation on test set
        print("\nPerforming final evaluation on test set...")
        test_metrics = trainer.evaluate(
            training_components['test_loader'],
            epoch=config['num_epochs']
        )
        
        print("\nFinal Test Metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (float, np.float32, np.float64)):
                print(f"{metric}: {value:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        torch.save({
            'model_state_dict': training_components['model'].state_dict(),
            'optimizer_state_dict': training_components['optimizer'].state_dict(),
            'metrics': trainer.metrics_tracker.epoch_metrics,
        }, os.path.join(config['checkpoint_dir'], 'interrupted_checkpoint.pth'))
        print("Checkpoint saved successfully.")
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    finally:
        if hasattr(trainer, 'metrics_tracker'):
            trainer.metrics_tracker.save_metrics_history(
                os.path.join(r'A:\Software Projects\NLST-App\metrics', 'medicalnet.csv')
            )

if __name__ == "__main__":
    main()