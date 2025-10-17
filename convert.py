import torch
from torchvision.models import efficientnet_v2_s
import torch.nn as nn

model = efficientnet_v2_s()
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(r"C:\Users\VC\Desktop\bin\bin\checkpoints\efficientnetv2s_epoch150.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(model, dummy_input, "drywet_model.onnx", opset_version=14)
print("✅ Saved drywet_model.onnx")

