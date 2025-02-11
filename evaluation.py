import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import CNN

num_classes = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.8128030896186829, 0.5365262627601624, 0.5638176798820496], std=[0.1239788755774498, 0.2561289966106415, 0.2529338300228119])
  # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("modelo.pth", map_location=device))
model.eval()

def evaluation_model(model, dataloader):
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Precisão do modelo: {accuracy:.2f}%")

evaluation_model(model, test_loader)
