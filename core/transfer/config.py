import torch

# Transfer Learning Configuration
TRANSFER_CONFIG = {
    "pretrained_model": "resnet50",
    "num_classes": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
