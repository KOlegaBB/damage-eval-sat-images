import albumentations as A
import cv2

def get_train_transform():
    return A.Compose([
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=0.75),
    ])

def get_test_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
    ])