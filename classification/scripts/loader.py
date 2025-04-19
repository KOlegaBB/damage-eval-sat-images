# loader.py
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DisasterDataset(Dataset):
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
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre",
                                row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post",
                                 row["post_image"])

        pre_image = cv2.imread(pre_path)
        post_image = cv2.imread(post_path)
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]

        label = self.label_map[row["damage"]]
        return pre_image, post_image, label


class DisasterMasksDataset(Dataset):
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
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre",
                                row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post",
                                 row["post_image"])
        mask_path = os.path.join(self.root_dir, self.split, "mask",
                                 row["mask_image"])

        pre_image = cv2.imread(pre_path)
        post_image = cv2.imread(post_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image,
                                         mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        mask_image = mask_image.float() / 255.0
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label


class DisasterMasksDatasetSize(Dataset):
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
        row = self.df.iloc[idx]
        pre_path = os.path.join(self.root_dir, self.split, "pre",
                                row["pre_image"])
        post_path = os.path.join(self.root_dir, self.split, "post",
                                 row["post_image"])
        mask_path = os.path.join(self.root_dir, self.split, "mask",
                                 row["mask_image"])

        pre_image = cv2.imread(pre_path)
        post_image = cv2.imread(post_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        original_mask_area = (mask_image > 0).sum()

        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=pre_image, image0=post_image,
                                         mask=mask_image)
            pre_image = transformed["image"]
            post_image = transformed["image0"]
            mask_image = transformed["mask"]

        mask_image = mask_image.float() / 255.0
        label = self.label_map[row["damage"]]
        return pre_image, post_image, mask_image, label, original_mask_area


def get_dataloaders(root_dir, dataset_type="disaster", batch_size=32,
                    transform=None, num_workers=2, test_only=False):
    """
    dataset_type: str - one of 'disaster', 'masks', 'masks_size'
    transform: albumentations.Compose
    Returns: train_loader, val_loader, test_loader
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
        train_dataset = DatasetClass(root_dir=root_dir, split="train",
                                     transform=transform)
        val_dataset = DatasetClass(root_dir=root_dir, split="val",
                                   transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    else:
        train_loader = val_loader = None

    test_dataset = DatasetClass(root_dir=root_dir, split="test",
                                transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
