import torch
import logging

# Logger for Compression module
def setup_logger(name="compression"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    return logging.getLogger(name)

logger = setup_logger()

# Model Compression Helper
def apply_pruning(model, amount=0.3):
    import torch.nn.utils.prune as prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            logger.info(f"Pruned {name} by {amount*100}%")
    return model
