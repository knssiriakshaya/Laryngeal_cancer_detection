import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes in your dataset
num_classes = 4

# Class labels dictionary
class_labels = {
    0: 'Hepatitis B Virus (Related Condition) (Hbv)',
    1: 'He',  # Used to indicate no cancer
    2: 'Inverted Papilloma and Laryngeal Cancer (IPCL)',
    3: 'Laryngeal Cancer (Primary) (Le)'
}


# Load the SqueezeNet 1.0 model (same as training)
model = models.squeezenet1_0(pretrained=False)

# Modify the classifier to match number of classes
model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
model.num_classes = num_classes

# Load the trained weights
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the image transformation pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # Open image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_name = class_labels[predicted.item()]
    
    if class_name == 'He':
        return "No Cancer Detected", class_name
    else:
        return "Laryngeal Cancer Detected", class_name
