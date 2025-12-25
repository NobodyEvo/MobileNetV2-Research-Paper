import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Config
TEST_DIR = r"Final_Split_Sugarcane\test"
MODEL_PATH = 'sugarcane_mobilenet_448.pth' 
IMG_SIZE = 448                              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. PLOT TRAINING HISTORY 
print("Generating Training Graphs...")

epochs = range(1, 31)

# UPDATED DATA
train_acc = [
    0.5602, 0.7130, 0.7434, 0.7743, 0.7820, 0.8195, 0.7980, 0.7930, 0.8206, 0.8135,
    0.8184, 0.8317, 0.8460, 0.8195, 0.8251, 0.8389, 0.8460, 0.8372, 0.8394, 0.8355,
    0.8543, 0.8499, 0.8532, 0.8543, 0.8471, 0.8543, 0.8626, 0.8460, 0.8731, 0.8615
]
train_loss = [
    1.2621, 0.9119, 0.8008, 0.7145, 0.6572, 0.6078, 0.6046, 0.5904, 0.5359, 0.5274,
    0.5194, 0.4930, 0.4806, 0.4777, 0.4860, 0.4537, 0.4625, 0.4621, 0.4439, 0.4517,
    0.4232, 0.4273, 0.4193, 0.4251, 0.4299, 0.4072, 0.3920, 0.4279, 0.3799, 0.4101
]

plt.figure(figsize=(14, 5))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'g-o', label='Training Accuracy')
plt.title('Sugarcane Accuracy (448px Resolution)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'r-o', label='Training Loss')
plt.title('Sugarcane Loss (448px Resolution)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('sugarcane_training_graph.png')
print("Saved Graph: sugarcane_training_graph.png")

# 3. EVALUATE ON UNSEEN DATA
print("\n🔍 Testing on Unseen Data (Hidden Split)...")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    # Reduced batch size for testing too
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    print(f"Found {len(test_dataset)} test images.")
except FileNotFoundError:
    print(f"Error: Could not find {TEST_DIR}")
    exit()

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 5) 
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Predict
y_true = []
y_pred = []

print("Running predictions...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 4. REPORT & MATRIX
print("\n" + "="*40)
print("FINAL SUGARCANE TEST REPORT (448px)")
print("="*40)
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Sugarcane Confusion Matrix (448px)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('sugarcane_confusion_matrix.png')
print("Saved Matrix: sugarcane_confusion_matrix.png")

plt.show()
