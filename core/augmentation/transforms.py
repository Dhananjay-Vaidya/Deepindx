import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

class AutoAugment:
    def __init__(self, policy):
        self.policy = policy
        
    def __call__(self, imgs):
            # If the input is a batch of images (4D tensor), process each image separately
            if imgs.ndimension() == 4:
                augmented_imgs = []
                for img in imgs:
                    img = F.to_pil_image(img)  # Convert each image to PIL format
                    for sub_policy in self.policy:
                        for op_name, prob, magnitude in sub_policy:
                            if random.random() < prob:
                                img = self.apply_operation(img, op_name, magnitude)
                    img = F.to_tensor(img)  # Convert back to tensor
                    augmented_imgs.append(img)
                return torch.stack(augmented_imgs)  # Combine augmented images into a batch
            
            # If it's a single image (3D tensor), process it directly
            if isinstance(imgs, torch.Tensor):
                imgs = F.to_pil_image(imgs)
            
            for sub_policy in self.policy:
                for op_name, prob, magnitude in sub_policy:
                    if random.random() < prob:
                        imgs = self.apply_operation(imgs, op_name, magnitude)
                        
            return F.to_tensor(imgs)
    
    def apply_operation(self, img, op_name, magnitude):
        operations = {
            'ShearX': lambda: F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(magnitude, 0)),
            'ShearY': lambda: F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(0, magnitude)),
            'TranslateX': lambda: F.affine(img, angle=0, translate=(int(magnitude * img.size[0]), 0), scale=1.0, shear=0),
            'TranslateY': lambda: F.affine(img, angle=0, translate=(0, int(magnitude * img.size[1])), scale=1.0, shear=0),
            'Rotate': lambda: F.rotate(img, magnitude),
            'Brightness': lambda: F.adjust_brightness(img, max(0, 1 + magnitude)),
            'Color': lambda: F.adjust_saturation(img, max(0, 1 + magnitude)),
            'Contrast': lambda: F.adjust_contrast(img, max(0, 1 + magnitude)),
            'Sharpness': lambda: F.adjust_sharpness(img, max(0, 1 + magnitude)),
            'Posterize': lambda: F.posterize(img, max(1, int(magnitude))),
            'Solarize': lambda: F.solarize(img, magnitude),
            'AutoContrast': lambda: F.autocontrast(img),
            'Equalize': lambda: F.equalize(img),
            'Invert': lambda: F.invert(img)
        }
        return operations.get(op_name, lambda: img)()

class RandAugment:
    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.available_ops = [
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
            'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize', 
            'Solarize', 'AutoContrast', 'Equalize', 'Invert'
        ]
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        
        ops = random.sample(self.available_ops, self.num_ops)
        for op_name in ops:
            magnitude = random.uniform(0, self.magnitude)
            img = AutoAugment([]).apply_operation(img, op_name, magnitude)
            
        return F.to_tensor(img)

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        image_a, image_b = images, images[index]
        label_a, label_b = labels, labels[index]
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        image_a[:, :, bbx1:bbx2, bby1:bby2] = image_b[:, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        return image_a, label_a, label_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class Mixup:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        label_a, label_b = labels, labels[index]
        
        return mixed_images, label_a, label_b, lam
