"""
Improved Person Detection Model Training
=========================================

This script trains a CNN model to detect single vs multiple persons
using a larger, better quality dataset.

Dataset structure:
person dataset improved/
    data/
        train/
            single_person/
            multiple_person/
        val/
            single_person/
            multiple_person/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_DIR = "person dataset improved/data"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30  # More epochs for better learning
LEARNING_RATE = 0.001
IMAGE_SIZE = 128

# Model save path
MODEL_SAVE_PATH = "models/person_detector_improved.pth"
HISTORY_SAVE_PATH = "models/person_training_history_improved.json"
PLOT_SAVE_PATH = "models/person_training_plot_improved.png"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# IMPROVED CNN MODEL ARCHITECTURE
# ============================================================================

class ImprovedPersonDetectorCNN(nn.Module):
    """
    Improved CNN architecture for person detection
    
    Improvements:
    - Deeper network (4 conv blocks instead of 3)
    - More filters (32→64→128→256)
    - Batch normalization for stability
    - Dropout for regularization
    - Larger fully connected layer
    """
    
    def __init__(self):
        super(ImprovedPersonDetectorCNN, self).__init__()
        
        # Convolutional blocks with increasing filters
        self.features = nn.Sequential(
            # Block 1: 128x128x3 -> 64x64x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2: 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3: 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 4: 16x16x128 -> 8x8x256 (NEW)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),  # Larger hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Output: 2 classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# DATA AUGMENTATION & LOADING
# ============================================================================

# Enhanced data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================================
# MODEL, LOSS, OPTIMIZER
# ============================================================================

print("\nInitializing model...")
model = ImprovedPersonDetectorCNN().to(device)

# Loss function with class weights (if imbalanced)
criterion = nn.CrossEntropyLoss()

# Optimizer with learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
start_time = datetime.now()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Model saved! (Best Val Acc: {best_val_acc:.2f}%)")

training_time = datetime.now() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Training time: {training_time}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")

# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================

with open(HISTORY_SAVE_PATH, 'w') as f:
    json.dump(history, f, indent=4)
print(f"Training history saved to: {HISTORY_SAVE_PATH}")

# ============================================================================
# PLOT TRAINING CURVES
# ============================================================================

plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"Training plot saved to: {PLOT_SAVE_PATH}")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Dataset: {DATASET_DIR}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Training time: {training_time}")
print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print(f"History saved to: {HISTORY_SAVE_PATH}")
print(f"Plot saved to: {PLOT_SAVE_PATH}")
print("="*80)
