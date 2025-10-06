import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

class WasteClassifier:
    def __init__(self, model_path, num_classes=2, classes=None, device="cpu"):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.classes = classes if classes else["Dry Waste", "Wet Waste"]

        # Define preprocessing transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load model
        self.model = efficientnet_v2_s(weights=None)  # architecture only
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_t)
            predicted_idx = torch.argmax(outputs, dim=1).item()

        return self.classes[predicted_idx]


# --- Example usage ---
if __name__ == "__main__":
    classifier = WasteClassifier(
        model_path="efficientnet_waste.pth",
        classes=["Dry Waste", "Wet Waste"]
    )

    result = classifier.predict("test2.jpg")
    print(f"Prediction: {result}")
