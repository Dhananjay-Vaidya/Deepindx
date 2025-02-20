import torch
import torch.optim as optim
import torch.nn as nn
from datanas import CIFAR
from search_space import SearchSpace

class NASOptimizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.train_loader, self.test_loader = CIFAR(batch_size=128)  # Load CIFAR-10
        self.model = SearchSpace().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def train(self, epochs=10):
        for epoch in range(epochs):
            loss = self.train_one_epoch()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
