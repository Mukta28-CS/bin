import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import os

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define class names (must match your training dataset structure)
class_names = ['dry', 'wet']  # Adjust if your folder names differ

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model architecture
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = efficientnet_v2_s(weights=weights)

# Modify the classifier for 2 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))

# Load trained weights
model.load_state_dict(torch.load("checkpoints/efficientnetv2s_epoch_50.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded and ready for inference!")

# Function to predict single image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = torch.softmax(outputs, dim=1)[0][class_idx].item()

    print(f"Image: {os.path.basename(image_path)} → Predicted: {class_names[class_idx]} "
          f"(Confidence: {confidence*100:.2f}%)")
    return class_names[class_idx], confidence

# Example usage:
# Predict a single image
image_path = "test2.jpg"  # Replace with your test image path
if os.path.exists(image_path):
    predict_image(image_path)
else:
    print("Please place a test image and update 'image_path' variable.")

# (Optional) Predict all images in a folder
# folder_path = "test_images"
# for img_file in os.listdir(folder_path):
#     if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#         predict_image(os.path.join(folder_path, img_file))
