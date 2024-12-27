import os
import cv2
from torch.utils.data import Dataset, DataLoader
from .utils import one_hot_encode, to_tensor
from .transforms import get_train_transform, get_test_transform

class BuildingsDataset(Dataset):
    """
    A PyTorch Dataset for loading images and their corresponding masks
    for semantic segmentation tasks.

    Args:
        images_dir (str): Path to the directory containing input images.
        masks_dir (str): Path to the directory containing corresponding masks.
        class_rgb_values (list, optional): A list of RGB values representing
            the classes in the segmentation task. Defaults to None.
        augmentation (callable, optional): A function or object for applying
            augmentations to the images and masks. Defaults to None.

    Attributes:
        image_paths (list): List of paths to the input images.
        mask_paths (list): List of paths to the corresponding masks.
        class_rgb_values (list): The provided list of RGB values for classes.
        augmentation (callable): Augmentation function applied to samples.
    """
    def __init__(self, images_dir, masks_dir, class_rgb_values=None, augmentation=None):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, i):
        """
        Fetches the image and mask at the specified index.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input image tensor.
                - torch.Tensor: The corresponding mask tensor.
        """
        # Load and convert the image and mask
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # One-hot encode the mask using the provided RGB values
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # Apply augmentations if specified
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Convert image and mask to PyTorch tensors
        return to_tensor(image), to_tensor(mask)



def get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir, class_rgb_values, batch_size=16):
    """
    Create and return PyTorch dataloaders for training, validation, and testing datasets.

    Args:
        x_train_dir (str): Directory path containing training images.
        y_train_dir (str): Directory path containing corresponding training masks.
        x_valid_dir (str): Directory path containing validation images.
        y_valid_dir (str): Directory path containing corresponding validation masks.
        x_test_dir (str): Directory path containing test images.
        y_test_dir (str): Directory path containing corresponding test masks.
        class_rgb_values (list): List of RGB values representing the classes in the segmentation task.
        batch_size (int, optional): Batch size for the training dataloader. Defaults to 16.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - valid_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    """
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
