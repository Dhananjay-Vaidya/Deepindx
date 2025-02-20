import torch
import logging
import os

# Logger for NAS module
def setup_logger(name="nas"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    return logging.getLogger(name)

logger = setup_logger()

# Model saving/loading for NAS
def save_nas_model(model, path="nas_checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"NAS model saved at {path}")

def load_nas_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info(f"NAS model loaded from {path}")
    else:
        logger.warning(f"NAS model file {path} not found!")
    return model
