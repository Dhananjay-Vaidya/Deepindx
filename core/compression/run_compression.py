import torch
import os

#Set correct quantization engine for x86-64 CPUs
torch.backends.quantized.engine = 'fbgemm'  # Use FBGEMM instead of QNNPACK

# Define paths for pruned and quantized models
pruned_model_path = "pruned_model.pth"
quantized_model_path = "quantized_model.pth"

# Check if pruned model exists
if not os.path.exists(pruned_model_path):
    raise FileNotFoundError(f"Pruned model not found: {pruned_model_path}")

# Load pruned model
import torchvision.models as models
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.load_state_dict(torch.load(pruned_model_path))
model.eval()

# Apply Post-Training Quantization (PTQ)
model_quantized = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.Conv2d},  # Quantizing Linear & Conv layers
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_quantized.state_dict(), quantized_model_path)
print(f" Compression successful! Quantized model saved at {quantized_model_path}.")

