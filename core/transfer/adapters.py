import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes=10):
    model = models.resnet50(pretrained=True)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
