import numpy as np
import shap
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gradcam import GradCAM
from lime_explainer import LimeExplainer
from shap_explainer import ShapExplainer

class RunExplainer:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.gradcam = GradCAM(model, target_layer) if target_layer else None
        self.lime = LimeExplainer(model)
        self.shap = ShapExplainer(model)
    
    def run_gradcam(self, input_tensor, target_class):
        if self.gradcam:
            return self.gradcam.generate_heatmap(input_tensor, target_class)
        else:
            raise ValueError("Grad-CAM not initialized. Provide a target layer.")
    
    def run_lime(self, image, preprocess_fn, target_class):
        return self.lime.explain(image, preprocess_fn, target_class)
    
    def run_shap(self, input_tensor):
        return self.shap.explain(input_tensor)

    def visualize_gradcam(self, input_tensor, target_class, original_image):
        heatmap = self.run_gradcam(input_tensor, target_class)
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

    def visualize_lime(self, image, preprocess_fn, target_class):
        explanation = self.run_lime(image, preprocess_fn, target_class)
        temp, mask = explanation.get_image_and_mask(target_class, positive_only=True, num_features=10, hide_rest=True)
        plt.imshow(temp)
        plt.axis('off')
        plt.show()

    def visualize_shap(self, input_tensor):
        shap_values = self.run_shap(input_tensor)
        shap.image_plot(shap_values, input_tensor.numpy())
