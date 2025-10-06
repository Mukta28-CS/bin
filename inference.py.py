import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image

#  Load Model 
num_classes = 2  # dry, wet
model = efficientnet_v2_s(weights=None)  # no pretrained classifier
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Loading trained weights
model.load_state_dict(torch.load(r"C:\Users\VC\Desktop\eco_bin\efficientnet_waste.pth", map_location="cpu"))
model.eval()

# Preprocessing 
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # EfficientNetV2-S default size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class 
classes = ["Dry Waste", "Wet Waste"]

# --- 4. Load and preprocess image ---
img = Image.open("test2.jpg").convert("RGB")
img_t = transform(img).unsqueeze(0)  # add batch dimension

# --- 5. Run inference ---
with torch.no_grad():
    outputs = model(img_t)
    predicted_idx = torch.argmax(outputs, dim=1).item()

print(f"Prediction: {classes[predicted_idx]}")