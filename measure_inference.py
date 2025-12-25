import torch
import time
from torchvision import models
import torch.nn as nn
import os

# CONFIGURATION
# ---------------------------------------------------------
# 1. SET THE IMAGE SIZE 
IMG_SIZE = 224

# 2. SET THE EXACT PATH TO YOUR MODEL FILE
MODEL_FOLDER = r"C:\Project\Research paper\Sugercane\Paper224"

MODEL_FILENAME = "sugarcane_mobilenet.pth" 
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------

def load_model():
    print(f"⏳ Loading Model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ERROR: Model file not found at {MODEL_PATH}\n   -> Check if the filename is correct!")

    # 1. Initialize Architecture
    model = models.mobilenet_v2(weights=None)
    
    # 2. Rebuild the Custom Head
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    
    # 3. Load Weights
    try:
        # Load state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Direct load failed. Trying full model load... Error: {e}")
        
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        
    model.to(DEVICE)
    model.eval()
    return model

def measure_speed(model):
    print(f"Measuring Inference Speed on {DEVICE} at {IMG_SIZE}x{IMG_SIZE}...")
    
    # 1. Create a Dummy Image (Batch Size = 1)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # 2. GPU Warm-up
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # 3. Actual Measurement
    iterations = 100
    print(f"⏱️ Running {iterations} predictions...")
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
                
    end_time = time.time()
    
    # 4. Calculate Stats
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    print("\n" + "="*40)
    print(f"RESULTS FOR {IMG_SIZE}px MODEL")
    print("="*40)
    print(f"Path: {MODEL_PATH}")
    print(f"Avg Inference Time per Image: {avg_time_ms:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    try:
        model = load_model()
        measure_speed(model)
    except Exception as e:

        print(f"\n CRITICAL ERROR: {e}")
