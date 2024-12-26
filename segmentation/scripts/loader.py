import os
import cv2
from torch.utils.data import Dataset, DataLoader
from .utils import one_hot_encode, to_tensor
from .transforms import get_train_transform, get_test_transform

class BuildingsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_rgb_values=None, augmentation=None):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return to_tensor(image), to_tensor(mask)


def get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir, class_rgb_values, batch_size=16):
    train_dataset = BuildingsDataset(
        x_train_dir, y_train_dir,
        augmentation=get_train_transform(),
        class_rgb_values=class_rgb_values,
    )

    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_test_transform(),
        class_rgb_values=class_rgb_values,
    )

    test_dataset = BuildingsDataset(
        x_test_dir, y_test_dir,
        augmentation=get_test_transform(),
        class_rgb_values=class_rgb_values,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, valid_loader, test_loader
