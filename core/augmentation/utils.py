import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

def apply_random_transformation(img, transformations, magnitude=0.5):
    """
    Apply a random transformation from the list of given transformations.
    """
    operation = random.choice(transformations)
    return apply_operation(img, operation, magnitude)

def apply_operation(img, op_name, magnitude):
    """
    Apply a specific transformation operation to the image.
    """
    operations = {
        'ShearX': lambda: F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(magnitude, 0)),
        'ShearY': lambda: F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(0, magnitude)),
        'Rotate': lambda: F.rotate(img, magnitude),
        'Brightness': lambda: F.adjust_brightness(img, max(0, 1 + magnitude)),
        'Contrast': lambda: F.adjust_contrast(img, max(0, 1 + magnitude)),
        'Sharpness': lambda: F.adjust_sharpness(img, max(0, 1 + magnitude)),
        'Posterize': lambda: F.posterize(img, max(1, int(magnitude)))
    }
    return operations.get(op_name, lambda: img)()