import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset_path = "dataset-train"
def compute_mean_std(dataset_path):
  transform = transforms.Compose([
      transforms.Resize((64,64)),
      transforms.ToTensor()
  ])

  dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
  loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

  mean = torch.zeros(3)
  std = torch.zeros(3)
  num_samples = 0

  for images, _ in loader:
    batch_samples = images.size(0)
    num_samples += batch_samples
    
    mean += images.mean(dim=[0, 2, 3]) * batch_samples
    std += images.std(dim=[0, 2, 3]) * batch_samples

  mean /= num_samples
  std /= num_samples

  print(f"Mean calculado: {mean.tolist()}")
  print(f"Std calculado: {std.tolist()}")

if __name__ == '__main__':
  compute_mean_std(dataset_path)
