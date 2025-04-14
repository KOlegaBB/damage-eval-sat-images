import albumentations as A
import cv2


def get_train_transform(
    use_crop=True,
    crop_size=(256, 256),
    use_color_transforms=False,
    flip_prob=0.75,
    brightness_limit=(-0.2, 0.2),
    contrast_limit=(-0.2, 0.2),
    hue_shift_limit=(-20, 20),
    sat_shift_limit=(-30, 30),
    val_shift_limit=(-20, 20),
    gamma_limit=(80, 120)
):
    """
    Returns a composition of data augmentations to be applied to images during training.

    Args:
        use_crop (bool, optional): Whether to apply random cropping. Defaults to True.
        crop_size (tuple, optional): Size of the crop (height, width). Defaults to (256, 256).
        use_color_transforms (bool, optional): Whether to include color transformations. Defaults to False.
        flip_prob (float, optional): Probability of applying flip/rotation. Defaults to 0.75.
        brightness_limit (tuple, optional): Range for brightness adjustment. Defaults to (-0.2, 0.2).
        contrast_limit (tuple, optional): Range for contrast adjustment. Defaults to (-0.2, 0.2).
        hue_shift_limit (tuple, optional): Hue shift range. Defaults to (-20, 20).
        sat_shift_limit (tuple, optional): Saturation shift range. Defaults to (-30, 30).
        val_shift_limit (tuple, optional): Value shift range. Defaults to (-20, 20).
        gamma_limit (tuple, optional): Range for gamma correction. Defaults to (80, 120).

    Returns:
        A.Compose: A pipeline of augmentations to be applied to training images.
    """
    transforms = [
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=flip_prob),
    ]

    if use_crop:
        transforms.append(A.RandomCrop(height=crop_size[0], width=crop_size[1], always_apply=True))

    if use_color_transforms:
        color_transforms = [
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.5),
            A.HueSaturationValue(hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, p=0.5),
            A.RandomGamma(gamma_limit=gamma_limit, p=0.5),
        ]
        transforms.extend(color_transforms)

    return A.Compose(transforms)


def get_test_transform(min_size=(1536, 1536), pad_value=(0, 0, 0)):
    """
    Returns a composition of transformations for testing/validation images.

    Args:
        min_size (tuple, optional): Minimum size (height, width) after padding. Defaults to (1536, 1536).
        pad_value (tuple, optional): RGB value for padding color. Defaults to (0, 0, 0).

    Returns:
        A.Compose: A pipeline of transformations to be applied to test images.
    """
    return A.Compose([
        A.PadIfNeeded(min_height=min_size[0], min_width=min_size[1], always_apply=True,
                      border_mode=cv2.BORDER_CONSTANT, value=pad_value),
    ])
