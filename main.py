import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from cnn import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformações para normalizar e redimensionar as imagens
transform = transforms.Compose([
  transforms.Resize((64, 64)),  
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.8128030896186829, 0.5365262627601624, 0.5638176798820496], std=[0.1239788755774498, 0.2561289966106415, 0.2529338300228119]) 
  # transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Carregar os dados
train_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Criar modelo, função de perda e otimizador
num_classes = len(train_dataset.classes)  
model = CNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento do modelo
def training(num_epochs=10):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f"Época {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

if __name__ == '__main__':
  training()
  print("Treinamento concluído!")

  torch.save(model.state_dict(), "modelo.pth")
  print("Modelo salvo com sucesso!")