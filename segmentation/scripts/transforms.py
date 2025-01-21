import albumentations as A
import cv2


def get_train_transform(use_color_transforms=False):
    """
    Returns a composition of data augmentations to be applied to images during training.

    - Random crop to 256x256 size.
    - Randomly applies one of the following augmentations with 75% probability:
        - Horizontal Flip
        - Vertical Flip
        - 90-degree rotation.
    - Optionally adds color transformations:
        - RandomBrightnessContrast
        - HueSaturationValue
        - RandomGamma

    Args:
        use_color_transforms (bool, optional): If True, includes color transformations. Defaults to False.

    Returns:
        A.Compose: A pipeline of augmentations to be applied to training images.
    """
    transforms = [
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=0.75),
    ]

    if use_color_transforms:
        color_transforms = [
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ]
        transforms.extend(color_transforms)

    return A.Compose(transforms)

def get_test_transform():
    """
    Returns a composition of transformations for testing/validation images.

    - Pads images to ensure a minimum size of 1536x1536, adding padding where necessary.

    Returns:
        A.Compose: A pipeline of transformations to be applied to test images.
    """
    return A.Compose([
        A.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
    ])