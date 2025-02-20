import logging

# Logger for Explainer module
def setup_logger(name="explainer"):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    return logging.getLogger(name)

logger = setup_logger()

# Explanation Results Saver
def save_explanations(explanation, path="explanations.json"):
    import json
    with open(path, "w") as f:
        json.dump(explanation, f, indent=4)
    logger.info(f"Explanations saved at {path}")
