import torch

# Compression Configuration
COMPRESSION_CONFIG = {
    "pruning_amount": 0.3,
    "quantization_dtype": torch.qint8,
    "apply_quantization": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
