import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Dry, Wet
# --- 2. Data transforms ---
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 3. Load dataset (make sure you have folders train/dry, train/wet, valid/dry, valid/wet) ---
train_data = datasets.ImageFolder("data/train", transform=transform)
val_data = datasets.ImageFolder("data/valid", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)

# --- 4. Load pretrained model ---
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# --- 5. Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 6. Training loop ---
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# --- 7. Save trained model weights ---
torch.save(model.state_dict(), "efficientnet_waste.pth")
print("✅ Model saved as efficientnet_waste.pth")