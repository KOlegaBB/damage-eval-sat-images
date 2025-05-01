import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_default_transforms(image_size=224):
    """
    Create a default set of image and mask transformations for preprocessing and normalization.

    Applies resizing, normalization (using ImageNet statistics), and conversion to PyTorch tensors.
    Also supports additional targets for multi-image inputs and masks.

    Args:
        image_size (int, optional): Target height and width to resize the images and masks to. Default is 224.

    Returns:
        albumentations.Compose: A composed transformation object that applies:
            - Resize to (image_size, image_size)
            - Normalization with ImageNet mean and std
            - Conversion to PyTorch tensors (for both image and additional targets like image0 and mask)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={"image0": "image", "mask": "mask"})
