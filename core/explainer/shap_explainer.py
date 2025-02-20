import torch
import shap
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

class ShapExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.GradientExplainer(self.model, torch.zeros((1, 3, 224, 224)))
    
    def explain(self, input_tensor):
        shap_values = self.explainer.shap_values(input_tensor)
        return shap_values