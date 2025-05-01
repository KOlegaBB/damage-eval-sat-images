import os
import cv2
from torch.utils.data import Dataset, DataLoader
from .utils import one_hot_encode, to_tensor


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
        num_crops (int, optional): Number of crops to generate per image. Defaults to 1.
    """

    def __init__(self, images_dir, masks_dir, class_rgb_values=None,
                 augmentation=None, num_crops=1):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in
                            sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in
                           sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.num_crops = num_crops

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_paths) * self.num_crops

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
        # Map the flat index back to an image index
        image_index = i // self.num_crops

        # Load and convert the image and mask
        image = cv2.cvtColor(cv2.imread(self.image_paths[image_index]),
                             cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[image_index]),
                            cv2.COLOR_BGR2RGB)

        # One-hot encode the mask using the provided RGB values
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # Apply augmentations if specified
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Convert image and mask to PyTorch tensors
        return to_tensor(image), to_tensor(mask)


def get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir,
                    x_test_dir, y_test_dir, class_rgb_values, batch_size=16,
                    train_transform=None, test_transform=None,
                    num_crops=1):
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
        train_transform (callable, optional): Transformations to apply to training images and masks. Defaults to None.
        test_transform (callable, optional): Transformations to apply to validation and test images and masks. Defaults to None.
        num_crops (int, optional): Number of random crops to generate per image in the training dataset. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - valid_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    """

    train_dataset = BuildingsDataset(
        x_train_dir, y_train_dir,
        augmentation=train_transform,
        class_rgb_values=class_rgb_values, num_crops=num_crops
    )

    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir,
        augmentation=test_transform,
        class_rgb_values=class_rgb_values,
    )

    test_dataset = BuildingsDataset(
        x_test_dir, y_test_dir,
        augmentation=test_transform,
        class_rgb_values=class_rgb_values,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


class ExhaustiveCropsDataset(Dataset):
    """
    A PyTorch Dataset for generating non-overlapping crops from images and their corresponding masks.

    Args:
        images_dir (str): Path to the directory containing input images.
        masks_dir (str): Path to the directory containing corresponding masks.
        class_rgb_values (list): A list of RGB values representing the classes in the segmentation task.
        crop_size (tuple): The size (height, width) of each crop.
        augmentation (callable, optional): A function/transform to apply to the image.
    """

    def __init__(self, images_dir, masks_dir, class_rgb_values,
                 crop_size=(256, 256), augmentation=None):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in
                            sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in
                           sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.crop_positions = []

        self._calculate_crops()

    def _calculate_crops(self):
        """Precompute crop positions for all images."""
        for i, image_path in enumerate(self.image_paths):
            # Load the image to calculate its size
            image = cv2.imread(image_path)
            h, w, _ = image.shape

            # Calculate how many non-overlapping crops we can have
            num_crops_y = h // self.crop_size[0]
            num_crops_x = w // self.crop_size[1]

            for y in range(num_crops_y):
                for x in range(num_crops_x):
                    self.crop_positions.append((image_path, self.mask_paths[i],
                                                y * self.crop_size[0],
                                                x * self.crop_size[1]))

    def __len__(self):
        """Return the total number of crops."""
        return len(self.crop_positions)

    def __getitem__(self, idx):
        """
        Fetches a crop from an image and its corresponding mask using pre-calculated positions.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input image tensor.
                - torch.Tensor: The corresponding mask tensor.
        """
        # Get the image path and crop position
        image_path, mask_path, y_start, x_start = self.crop_positions[idx]

        # Load the image and mask
        image = cv2.cvtColor(cv2.imread(image_path),
                           cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path),
                            cv2.COLOR_BGR2RGB)

        # Crop the image and mask (non-overlapping)
        crop_image = image[y_start:y_start + self.crop_size[0],
                     x_start:x_start + self.crop_size[1]]
        crop_mask = mask[y_start:y_start + self.crop_size[0],
                    x_start:x_start + self.crop_size[1]]

        crop_mask = one_hot_encode(crop_mask, self.class_rgb_values).astype('float')

        # Apply transformations if provided
        if self.augmentation:
            sample = self.augmentation(image=crop_image, mask=crop_mask)
            crop_image, crop_mask = sample['image'], sample['mask']

        return to_tensor(crop_image), to_tensor(crop_mask)


def get_exhaustive_crops_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir,
                                     x_test_dir, y_test_dir, class_rgb_values, batch_size=16,
                                     crop_size=(256, 256),
                                     train_transform=None, test_transform=None):
    """
    Create and return PyTorch dataloaders for training, validation, and testing datasets.
    The training dataset is created using all non-overlapping crops from the input images.

    Args:
        x_train_dir (str): Directory path containing training images.
        y_train_dir (str): Directory path containing corresponding training masks.
        x_valid_dir (str): Directory path containing validation images.
        y_valid_dir (str): Directory path containing corresponding validation masks.
        x_test_dir (str): Directory path containing test images.
        y_test_dir (str): Directory path containing corresponding test masks.
        class_rgb_values (list): List of RGB values representing the classes in the segmentation task.
        batch_size (int, optional): Batch size for the training dataloader. Defaults to 16.
        crop_size (tuple, optional): Tuple (H, W) specifying the height and width of each crop. Defaults to (256, 256).
        train_transform (callable, optional): Transformations to apply to training images and masks. Defaults to None.
        test_transform (callable, optional): Transformations to apply to validation and test images and masks. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - valid_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    """

    train_dataset = ExhaustiveCropsDataset(
        x_train_dir, y_train_dir,
        class_rgb_values=class_rgb_values, crop_size=crop_size,
        augmentation=train_transform,
    )

    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir,
        class_rgb_values=class_rgb_values,
        augmentation=test_transform,
    )

    test_dataset = BuildingsDataset(
        x_test_dir, y_test_dir,
        class_rgb_values=class_rgb_values,
        augmentation=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

