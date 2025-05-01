import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DisasterDataset(Dataset):
    """
    A PyTorch Dataset for loading pre- and post-disaster images along with damage labels.

    Args:
        root_dir (str): Path to the root dataset directory.
        split (str): Dataset split to load ('train', 'val', or 'test').
        transform (callable, optional): Transformations to apply to the image pair.

    Returns:
        tuple: (pre_image, post_image, label), where label is an integer in [0, 3].
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.csv_file = os.path.join(root_dir, split, f"{split}.csv")
        self.df = pd.read_csv(self.csv_file)  # Load metadata CSV
        self.transform = transform
        self.label_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read file names from CSV
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post", row["post_image"])

        # Load and convert images to RGB
        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)

        # Apply augmentations, if any
        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]

        # Convert damage label to integer
        label = self.label_map[row["damage"]]
        return pre_image, post_image, label


class DisasterMasksDataset(Dataset):
    """
    A PyTorch Dataset for loading pre- and post-disaster images, segmentation masks, and labels.

    Args:
        root_dir (str): Path to the root dataset directory.
        split (str): Dataset split to load ('train', 'val', or 'test').
        transform (callable, optional): Transformations to apply to images and masks.

    Returns:
        tuple: (pre_image, post_image, mask, label)
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.csv_file = os.path.join(root_dir, split, f"{split}.csv")
        self.df = pd.read_csv(self.csv_file)
        self.transform = transform
        self.label_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read file paths
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post", row["post_image"])
        mask_path = os.path.join(self.root_dir, self.split, "mask", row["mask_image"])

        # Load and convert images
        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)

        # Load mask in grayscale
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply transforms to both images and mask
        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image, mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        # Normalize mask to [0, 1] float
        mask_image = mask_image.float() / 255.0

        # Convert textual label to numeric
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label


class DisasterMasksDatasetSize(Dataset):
    """
    A PyTorch Dataset similar to DisasterMasksDataset, with an additional
    output for the original mask area before transformations.

    Args:
        root_dir (str): Path to the root dataset directory.
        split (str): Dataset split to load ('test' by default).
        transform (callable, optional): Transformations to apply to images and masks.

    Returns:
        tuple: (pre_image, post_image, mask, label, original_mask_area)
    """

    def __init__(self, root_dir, split="test", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.csv_file = os.path.join(root_dir, split, f"{split}.csv")
        self.df = pd.read_csv(self.csv_file)
        self.transform = transform
        self.label_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read metadata
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post", row["post_image"])
        mask_path = os.path.join(self.root_dir, self.split, "mask", row["mask_image"])

        # Load and convert images
        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Compute original area of the mask before transforms
        original_mask_area = (mask_image > 0).sum()

        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image, mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        # Normalize mask
        mask_image = mask_image.float() / 255.0
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label, original_mask_area


def get_dataloaders(root_dir, dataset_type="disaster", batch_size=32,
                    transform=None, num_workers=2, test_only=False):
    """
    Returns PyTorch DataLoaders for the specified disaster dataset type.

    Args:
        root_dir (str): Path to the dataset root directory.
        dataset_type (str): One of 'disaster', 'masks', or 'masks_size'.
        batch_size (int): Batch size for loading data.
        transform (callable, optional): Transformations to apply to data.
        num_workers (int): Number of worker threads for data loading.
        test_only (bool): If True, only returns the test loader.

    Returns:
        tuple: (train_loader, val_loader, test_loader) or (None, None, test_loader)
    """
    # Select dataset class based on type
    if dataset_type == "disaster":
        DatasetClass = DisasterDataset
    elif dataset_type == "masks":
        DatasetClass = DisasterMasksDataset
    elif dataset_type == "masks_size":
        DatasetClass = DisasterMasksDatasetSize
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load train and validation loaders if not test_only
    if not test_only:
        train_dataset = DatasetClass(root_dir=root_dir, split="train", transform=transform)
        val_dataset = DatasetClass(root_dir=root_dir, split="val", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    else:
        train_loader, val_loader = None, None

    # Always load test loader
    test_dataset = DatasetClass(root_dir=root_dir, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
