import torch
from optimizer import NASOptimizer
from evaluator import Evaluator

if __name__ == "__main__":
    print("Starting Neural Architecture Search (NAS) training on CIFAR-10...\n")

    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the best model
    optimizer = NASOptimizer()
    optimizer.model.to(device)  # Move model to device
    optimizer.train(epochs=10)

    # Create evaluator instance
    evaluator = Evaluator(device=device)  # Pass device to evaluator

    # Move model to device and ensure evaluation data is also on the same device
    evaluator.evaluate(optimizer.model)
