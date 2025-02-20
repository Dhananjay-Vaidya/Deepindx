import torch
import torchvision.models as models

# Set the correct quantization backend for x86-64 CPU
torch.backends.quantized.engine = 'fbgemm'  # Use FBGEMM instead of QNNPACK

# Load pruned model
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.load_state_dict(torch.load("pruned_model.pth"))
model.eval()

# Apply Post-Training Quantization (PTQ)
model_quantized = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.Conv2d},  # Quantize these layers
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_quantized.state_dict(), "quantized_model.pth")
print("Post-training quantization applied! Quantized model saved as quantized_model.pth.")
