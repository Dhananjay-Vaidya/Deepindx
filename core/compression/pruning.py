import torch
import torch.nn.utils.prune as prune
import torchvision.models as models

# Load ResNet18 model with updated weight parameter
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Apply pruning to convolutional and fully connected layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  # Remove pruning reparameterization

# Save pruned model
torch.save(model.state_dict(), "pruned_model.pth")
print("Pruning applied successfully! Pruned model saved as pruned_model.pth.")
