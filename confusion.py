import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================
# DEVICE CONFIGURATION
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# TRANSFORMS (Same as training)
# ============================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================
# LOAD VALIDATION DATASET
# ============================
val_dataset = datasets.ImageFolder(root='data/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = val_dataset.classes
print("Class Names:", class_names)

# ============================
# LOAD MODEL
# ============================
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = efficientnet_v2_s(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(
    r"C:\Users\VC\Desktop\bin\bin\checkpoints\efficientnetv2s_epoch150.pth", 
    map_location=device
))

model = model.to(device)
model.eval()

# ============================
# INFERENCE + CONFUSION MATRIX
# ============================
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

# ============================
# CLASSIFICATION REPORT
# ============================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ============================
# PLOT CONFUSION MATRIX
# ============================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - EfficientNetV2S Dry/Wet Classification')
plt.tight_layout()
plt.show()

# ============================
# SINGLE IMAGE INFERENCE EXAMPLE
# ============================
from PIL import Image

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
    print(f"Predicted: {predicted_class}")
    return predicted_class

# Example usage
# predict_image("data/val/dry/image1.jpg")
predict_image("test.jpg")
