import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class DisasterDataset(Dataset):
    """
    A PyTorch Dataset for loading pre- and post-disaster images with damage labels.

    Supports loading from a single split or a list of splits (e.g., for folds).

    Args:
        root_dir (str): Path to the root dataset directory.
        split (str or list): Dataset split(s) to load ('train', 'val', 'test', or list thereof).
        transform (callable, optional): Transformations to apply to the image pair.

    Returns:
        tuple: (pre_image, post_image, label), where label is an integer in [0, 3].
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.splits = [split] if isinstance(split, str) else split
        self.transform = transform
        self.label_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }

        dfs = []
        for s in self.splits:
            csv_file = os.path.join(root_dir, s, f"{s}.csv")
            df = pd.read_csv(csv_file)
            df["split_folder"] = s
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        split_folder = row["split_folder"]

        pre_path = os.path.join(self.root_dir, split_folder, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, split_folder, "post", row["post_image"])

        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]

        label = self.label_map[row["damage"]]
        return pre_image, post_image, label


class DisasterMasksDataset(DisasterDataset):
    """
    Dataset that includes segmentation masks along with images and labels.

    Returns:
        tuple: (pre_image, post_image, mask, label)
    """

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        split_folder = row["split_folder"]

        pre_path = os.path.join(self.root_dir, split_folder, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, split_folder, "post", row["post_image"])
        mask_path = os.path.join(self.root_dir, split_folder, "mask", row["mask_image"])

        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image, mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        mask_image = torch.tensor(mask_image, dtype=torch.float32) / 255.0
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label


class DisasterMasksDatasetSize(DisasterMasksDataset):
    """
    Dataset that also returns the original mask area before transformation.

    Returns:
        tuple: (pre_image, post_image, mask, label, original_mask_area)
    """

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        split_folder = row["split_folder"]

        pre_path = os.path.join(self.root_dir, split_folder, "pre", row["pre_image"])
        post_path = os.path.join(self.root_dir, split_folder, "post", row["post_image"])
        mask_path = os.path.join(self.root_dir, split_folder, "mask", row["mask_image"])

        pre_image = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        original_mask_area = (mask_image > 0).sum()

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image, mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        mask_image = torch.tensor(mask_image, dtype=torch.float32) / 255.0
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label, original_mask_area


def get_dataloaders(root_dir, dataset_type="disaster", batch_size=32,
                    transform=None, num_workers=2, test_only=False,
                    train_split="train", val_split="val", test_split="test"):
    """
    Returns PyTorch DataLoaders for the specified disaster dataset type.

    Args:
        root_dir (str): Path to the dataset root directory.
        dataset_type (str): One of ['disaster', 'masks', 'masks_size'].
        batch_size (int): Batch size for loading data.
        transform (callable, optional): Transformations to apply.
        num_workers (int): Number of subprocesses to use for data loading.
        test_only (bool): If True, only the test loader is returned.
        train_split (str or list): Name(s) of training split folder(s).
        val_split (str or list): Name(s) of validation split folder(s).
        test_split (str or list): Name(s) of test split folder(s).

    Returns:
        tuple: (train_loader, val_loader, test_loader) or (None, None, test_loader)
    """
    if dataset_type == "disaster":
        DatasetClass = DisasterDataset
    elif dataset_type == "masks":
        DatasetClass = DisasterMasksDataset
    elif dataset_type == "masks_size":
        DatasetClass = DisasterMasksDatasetSize
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not test_only:
        train_dataset = DatasetClass(root_dir=root_dir, split=train_split, transform=transform)
        val_dataset = DatasetClass(root_dir=root_dir, split=val_split, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    else:
        train_loader = val_loader = None

    test_dataset = DatasetClass(root_dir=root_dir, split=test_split, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
