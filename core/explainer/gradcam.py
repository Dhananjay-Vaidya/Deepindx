import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Function

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_score = output[:, target_class]
        class_score.backward()
        
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()
