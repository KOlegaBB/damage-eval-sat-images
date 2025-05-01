import os
import torch
import pandas as pd
from torch import optim

from .evaluation import evaluate_damagenet
from .loader import get_dataloaders
from .models import DamageNet
from .train import train_damagenet_model


def run_kfold_damagenet_experiments(
        DATA_DIR,
        transform,
        checkpoint_path,
        test_fold_combinations=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        epochs=30,
        mode="classification",
        freeze_modules=["stem", "res1"],
        batch_size=32,
        num_workers=2
):
    """
    Runs k-fold cross-validation experiments using DamageNet.

    Args:
        DATA_DIR (str): Root directory of the dataset.
        transform (callable): Transformations applied to input data.
        checkpoint_path (str): Path to pretrained model checkpoint.
        test_fold_combinations (list): List of fold index pairs used for testing.
        epochs (int): Number of training epochs.
        mode (str): "classification" or "regression".
        freeze_modules (list): Names of model submodules to freeze.
        batch_size (int): Dataloader batch size.
        num_workers (int): Dataloader worker threads.

    Returns:
        pd.DataFrame: Aggregated test metrics for all folds.
    """

    def freeze_module(module):
        """Freezes all parameters in a given module."""
        for param in module.parameters():
            param.requires_grad = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [f"fold_{i}" for i in range(10)]
    all_test_metrics = []

    for fold_idx, test_fold_ids in enumerate(test_fold_combinations):
        print(f"\n=== Fold {fold_idx + 1} | Test folds: {test_fold_ids} ===")

        test_folds = [f"fold_{i}" for i in test_fold_ids]
        train_folds = [f for f in folds if f not in test_folds]

        print(f"Train folds: {train_folds}")
        print(f"Test folds: {test_folds}")

        # --- Data Loaders ---
        train_loader, val_loader, test_loader = get_dataloaders(
            root_dir=DATA_DIR,
            dataset_type="masks",
            batch_size=batch_size,
            transform=transform,
            num_workers=num_workers,
            test_only=False,
            train_split=train_folds,
            val_split=test_folds,
            test_split=test_folds
        )

        # --- Model Setup ---
        model = DamageNet(mode=mode).to(DEVICE)

        if mode == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        # --- Load Pretrained Weights ---
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])

        # --- Freeze selected modules ---
        for name in freeze_modules:
            if hasattr(model, name):
                freeze_module(getattr(model, name))
            else:
                print(f"Warning: model has no attribute '{name}' to freeze.")

        # --- Training ---
        print("Training...")
        train_damagenet_model(
            model, train_loader, val_loader,
            loss_fn, optimizer,
            epochs=epochs,
            device=DEVICE,
            mode=mode
        )

        # --- Evaluation ---
        print("Evaluating...")
        test_metrics = evaluate_damagenet(model, test_loader, DEVICE,
                                          mode=mode)

        test_metrics["fold"] = fold_idx
        test_metrics["test_folds"] = test_fold_ids
        all_test_metrics.append(test_metrics)

    # --- Results Summary ---
    df = pd.DataFrame(all_test_metrics)
    print("\n=== All Fold Test Metrics ===")
    print(df)

    return df
