import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data2 import get_cifar10_data
from adapters import get_pretrained_model

class TransferOptimizer:
    def __init__(self, lr=0.001, batch_size=32, epochs=10, device='cuda'):
        self.device = device
        self.epochs = epochs
        self.train_loader, self.test_loader = get_cifar10_data(batch_size)
        self.model = get_pretrained_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')
