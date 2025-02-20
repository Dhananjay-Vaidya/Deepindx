import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from search_space import SearchSpace

class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=self.transform)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=64, shuffle=False)

    def train(self, model, epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    def evaluate(self, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy
