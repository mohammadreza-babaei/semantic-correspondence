import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from safetensors.torch import load_file

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(script_dir, "model.safetensors")
MODEL_NAME = "dinov3_vits16"                   
REPO_SOURCE = "facebookresearch/dinov3" 

def load_dinov3_safetensors(model_name, weights_path):
    print(f"Creating model architecture: {model_name}...")

    model = torch.hub.load(REPO_SOURCE, model_name, pretrained=False)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    print(f"Loading safetensors from: {weights_path}")
    # This is the replacement for torch.load
    state_dict = load_file(weights_path)

    # Safetensors from Hugging Face often wrap the model in "model." or "base_model."
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove Hugging Face specific prefixes
        k = k.replace("model.", "") 
        k = k.replace("base_model.model.", "")
        # Remove standard DINO prefixes
        k = k.replace("teacher.", "")
        k = k.replace("backbone.", "") 
        new_state_dict[k] = v
        
    # 4. Inject weights
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load Status: {msg}") 
    
    model.eval()
    return model

def test_inference(image_path="test_image.jpg"):
    # Create dummy image if needed
    if not os.path.exists(image_path):
        Image.new('RGB', (224, 224), color='blue').save(image_path)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # Load & Run
    model = load_dinov3_safetensors(MODEL_NAME, WEIGHTS_PATH)
    
    with torch.no_grad():
        features = model(input_tensor)

    print("\n--- Success! ---")
    print(f"Feature shape: {features.shape}")
    print("First 5 values:", features[0][:5])

if __name__ == "__main__":
    test_inference()