import torch
from torchvision import transforms
from PIL import Image
import os
import random
import string
from cnn import CNN

num_classes = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8128030896186829, 0.5365262627601624, 0.5638176798820496], std=[0.1239788755774498, 0.2561289966106415, 0.2529338300228119])
])

model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("modelo.pth", map_location=device))
model.eval()

test_dir = "dataset-test"

image_files = [f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

class_names = ["balao", "colcheias", "floco", "helice", "tv"]

def generate_key(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def transform_image(img_path):
    global device
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def rename_img(predicted, img_path):
    global class_names

    predicted_class = class_names[predicted.item()] if predicted.item() < len(class_names) else str(predicted.item())
    new_name = f"{predicted_class}_{generate_key()}.png"
    new_path = os.path.join(test_dir, new_name)
    os.rename(img_path, new_path)

    return predicted_class

def prediction(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted

def test_images():
    global image_files

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        
        image = transform_image(img_path)
        
        predicted = prediction(image)
        
        predicted_class = rename_img(predicted, img_path)

        print(f"Imagem: {img_name} -> Predição: {predicted_class}")

if __name__ == '__main__':
    test_images()