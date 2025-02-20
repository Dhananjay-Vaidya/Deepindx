import torch
import logging
import os

# Logger for Transfer Learning module
def setup_logger(name="transfer"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    return logging.getLogger(name)

logger = setup_logger()

# Model saving/loading for Transfer Learning
def save_transfer_model(model, path="transfer_checkpoint.pth"):
    torch.save(model.state_dict(), path)
    logger.info(f"Transfer learning model saved at {path}")

def load_transfer_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info(f"Transfer learning model loaded from {path}")
    else:
        logger.warning(f"Transfer learning model file {path} not found!")
    return model
