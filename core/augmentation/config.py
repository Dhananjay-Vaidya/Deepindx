AUGMENTATION_POLICIES = [
    [('ShearX', 0.5, 0.3), ('Rotate', 0.5, 10)],
    [('Brightness', 0.5, 0.4), ('Contrast', 0.5, 0.5)],
    [('Sharpness', 0.5, 0.5), ('Posterize', 0.5, 4)]
]

DEFAULT_CUTMIX_ALPHA = 1.0
DEFAULT_MIXUP_ALPHA = 1.0