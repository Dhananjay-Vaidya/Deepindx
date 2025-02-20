from tkinter import Image
import lime
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch

class LimeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime.lime_image.LimeImageExplainer()
    
    def explain(self, image, preprocess_fn, target_class):
        def batch_predict(images):
            batch = torch.stack([preprocess_fn(Image.fromarray(img)) for img in images])
            self.model.eval()
            with torch.no_grad():
                preds = self.model(batch)
            return preds.numpy()
        
        explanation = self.explainer.explain_instance(image, batch_predict, top_labels=5, hide_color=0, num_samples=1000)
        return explanation
