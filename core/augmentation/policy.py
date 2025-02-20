import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class PolicyLearner:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.writer = SummaryWriter()

    def train(self, epochs=20):  # Reduced epochs
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(self.train_loader)}, Accuracy: {accuracy}%')
            self.writer.add_scalar('Training Loss', running_loss / len(self.train_loader), epoch)
            self.writer.add_scalar('Training Accuracy', accuracy, epoch)

            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Validation Accuracy: {accuracy}%')
        self.writer.add_scalar('Validation Accuracy', accuracy, epoch)
        self.model.train()
