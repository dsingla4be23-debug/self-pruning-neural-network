import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv

from src.model.network import PrunableNet
from src.training.loss import compute_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

def calculate_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates()
    pruned = (gates < threshold).sum().item()
    total = gates.numel()
    return (pruned / total) * 100

lambda_values = [0.0001, 0.001, 0.01]
results = []

for lambda_ in lambda_values:
    print(f"\nTraining with lambda = {lambda_}")

    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = compute_loss(model, outputs, labels, lambda_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)

    results.append([lambda_, accuracy, sparsity])

    print(f"Lambda: {lambda_}, Accuracy: {accuracy:.2f}, Sparsity: {sparsity:.2f}")

with open("results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Lambda", "Accuracy", "Sparsity"])
    writer.writerows(results)
