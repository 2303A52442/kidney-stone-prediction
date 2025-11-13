import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

MODEL_PATH = "models/kidney_stone_model.pth"

# 1. Recreate the same architecture used during training
model = models.resnet18(weights=None)   # no pretrained weights
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)   # 2 classes: Normal, Stone

# 2. Load weights properly
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# 3. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Normal", "Stone"]

def predict_image(file):
    image = Image.open(file).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][predicted].item() * 100

    label = classes[predicted.item()]
    return label, prob





