import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_default_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={"image0": "image", "mask": "mask"})
