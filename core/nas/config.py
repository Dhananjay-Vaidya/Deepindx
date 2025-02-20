import torch

# NAS Search Space Configuration
NAS_CONFIG = {
    "search_algorithm": "reinforcement_learning",
    "num_generations": 50,
    "population_size": 20,
    "mutation_rate": 0.2,
    "selection_criteria": "accuracy",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
