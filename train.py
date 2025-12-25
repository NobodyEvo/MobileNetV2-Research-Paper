import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import time

# 1. Setup Device (Force GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Configuration
DATA_DIR = r"Final_Split_Sugarcane\train"
BATCH_SIZE = 16      
IMG_SIZE = 448        
EPOCHS = 30

# 3. Data Transforms
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Load Dataset & Correct Split Strategy
print("\nLoading Dataset...")
try:
    # We load the folder TWICE to allow different transforms for Train vs Val
    full_data_train = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    full_data_val = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
    
    class_names = full_data_train.classes
    print(f"Detected Classes: {class_names}")
    print(f"Total Images: {len(full_data_train)}")

    # Create Indices for Split
    dataset_size = len(full_data_train)
    indices = list(range(dataset_size))
    split = int(0.8 * dataset_size)
    
    # Shuffle indices manually (since we can't use random_split on two diff objects easily)
    import random
    random.shuffle(indices)
    
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Create Subsets using the correct transforms
    train_dataset = Subset(full_data_train, train_indices)
    val_dataset = Subset(full_data_val, val_indices)

except FileNotFoundError:
    print(f"ERROR: Could not find folder '{DATA_DIR}'. Please check the path.")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. Load Pre-trained MobileNetV2
print("\nInitializing MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze early layers
for param in model.features.parameters():
    param.requires_grad = False 

# Modify the final classification layer
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

# 7. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training Loop
print(f"\nStarting Training for {EPOCHS} Epochs at {IMG_SIZE}x{IMG_SIZE}...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    
    # Optional: Print Validation Accuracy every epoch to track progress
    # (Kept simple here for speed, but you will see Training Accuracy climb)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

total_time = time.time() - start_time
print(f"\nTraining Complete in {total_time//60:.0f}m {total_time%60:.0f}s")

# 9. Save Model
torch.save(model.state_dict(), 'sugarcane_mobilenet_448.pth') # Renamed slightly to avoid overwriting old one
print("Model saved as 'sugarcane_mobilenet_448.pth'")