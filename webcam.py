import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import cv2
from PIL import Image
import numpy as np

# ============================
# ✅ CONFIGURATION
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['dry_waste', 'wet_waste']  # Make sure order matches training dataset

# ============================
# ✅ LOAD MODEL
# ============================
model = efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(
    r"C:\Users\VC\Desktop\bin\bin\checkpoints\efficientnetv2s_epoch150.pth",
    map_location=device
))
model.to(device)
model.eval()

print("✅ Model loaded successfully on:", device)

# ============================
# ✅ TRANSFORM (same as training)
# ============================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================
# ✅ START WEBCAM
# ============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Press 'q' to quit the webcam.")

# ============================
# ✅ REAL-TIME LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    # Convert frame to RGB and preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_t = transform(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item() * 100

    # ============================
    # ✅ DEBUG PRINT (CONSOLE)
    # ============================
    print(f"Prediction: {label} | Confidence: {confidence:.2f}% | Raw probs: {probs.cpu().numpy()}")

    # ============================
    # ✅ DISPLAY ON FRAME
    # ============================
    cv2.putText(frame, f"{label} ({confidence:.1f}%)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Real-time Waste Classification", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
