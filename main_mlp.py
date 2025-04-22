import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from mlp import MLP  # Importando a classe MLP do novo arquivo

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Transformações para normalizar e redimensionar as imagens
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8128030896186829, 0.5365262627601624, 0.5638176798820496], 
                       std=[0.1239788755774498, 0.2561289966106415, 0.2529338300228119])
])

# Carregar os dados
print("\nCarregando conjunto de dados de treinamento...")
train_dataset = datasets.ImageFolder(root="dataset-train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print(f"Tamanho do conjunto de treinamento: {len(train_dataset)} imagens")

# Configurações do MLP
input_size = 64 * 64 * 3  # 64x64 pixels, 3 canais RGB
hidden_size = 512
num_classes = len(train_dataset.classes)

print("\nConfigurando parâmetros do modelo MLP:")
print(f"- Tamanho da entrada: {input_size} (64x64x3)")
print(f"- Tamanho da camada oculta: {hidden_size} neurônios")
print(f"- Número de classes: {num_classes}")

# Criar modelo, função de perda e otimizador
model = MLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento do modelo
def training(num_epochs=10):
    print("\nIniciando treinamento do modelo MLP...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f'Época [{epoch+1}/{num_epochs}], Lote [{i+1}/{len(train_loader)}], Perda Atual: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        print(f'Época [{epoch+1}/{num_epochs}] concluída, Perda Média: {epoch_loss:.4f}')

if __name__ == '__main__':
    training()
    print("\nTreinamento concluído!")

    # Salvar o modelo
    torch.save(model.state_dict(), "modelo_mlp.pth")
    print("Modelo MLP salvo como 'modelo_mlp.pth'") 