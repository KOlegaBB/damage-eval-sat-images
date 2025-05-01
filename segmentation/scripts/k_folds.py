import torch
import os
import pandas as pd

from .train import train_model
from .evaluation import evaluate_model
from .loader import get_exhaustive_crops_dataloaders


def run_kfold_segmentation_experiments(
        DATA_DIR,
        train_transform,
        test_transform,
        model_class,
        loss_fn,
        checkpoint_path,
        class_rgb_values=[(0, 0, 0), (255, 255, 255)],
        test_fold_combinations=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        lr=1e-4,
        epochs=10,
        num_pred=20,
        layers_to_freeze=["down_conv1", "down_conv2"]
):
    """
    Runs k-fold cross-validation experiments for semantic segmentation using a pretrained model checkpoint.

    For each fold combination, the model is loaded, optionally has layers frozen, trained on training folds,
    validated, and evaluated on test folds. Performance metrics and losses are collected and returned.

    Args:
        DATA_DIR (str): Root directory containing the fold subdirectories with 'images' and 'targets'.
        train_transform (callable): Data augmentation / preprocessing for training data.
        test_transform (callable): Preprocessing for validation and test data.
        model_class (Callable): A class constructor for the model architecture (e.g., UNet).
        loss_fn (callable): Loss function used during training and evaluation.
        checkpoint_path (str): Path to the pretrained model checkpoint file (.pth).
        class_rgb_values (list, optional): List of RGB tuples representing the class labels. Defaults to binary.
        test_fold_combinations (list of lists, optional): Fold combinations to use for testing. Defaults to 5 splits.
        lr (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        num_pred (int, optional): Number of predictions to generate during evaluation. Defaults to 20.
        layers_to_freeze (list of str, optional): Names of model layers to freeze before training. Defaults to first two down conv blocks.

    Returns:
        pd.DataFrame: DataFrame containing loss and evaluation metrics for each fold.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_test_metrics = []
    folds = [f'fold_{i}' for i in range(10)]

    def collect_fold_dirs(fold_list):
        images_dirs = [os.path.join(DATA_DIR, f, 'images') for f in fold_list]
        targets_dirs = [os.path.join(DATA_DIR, f, 'targets') for f in
                        fold_list]
        return images_dirs, targets_dirs

    def setup_model():
        model = model_class().to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])

        if layers_to_freeze:
            for layer_name in layers_to_freeze:
                layer = getattr(model, layer_name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    print(
                        f"[Warning] Layer '{layer_name}' not found in model.")

        return model

    for fold_idx, test_fold_ids in enumerate(test_fold_combinations):
        print(f"\n=== Fold {fold_idx + 1} | Test folds: {test_fold_ids} ===")

        test_folds = [f'fold_{i}' for i in test_fold_ids]
        train_folds = [f for f in folds if f not in test_folds]

        # Collect dataset paths
        x_train_dir, y_train_dir = collect_fold_dirs(train_folds)
        x_valid_dir, y_valid_dir = collect_fold_dirs(test_folds)
        x_test_dir, y_test_dir = collect_fold_dirs(test_folds)

        # Load data
        train_loader, valid_loader, test_loader = get_exhaustive_crops_dataloaders(
            x_train_dir, y_train_dir,
            x_valid_dir, y_valid_dir,
            x_test_dir, y_test_dir,
            class_rgb_values=class_rgb_values,
            train_transform=train_transform,
            test_transform=test_transform
        )

        # Model setup
        model = setup_model()
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])

        # Train
        print("Training...")
        _ = train_model(model, train_loader, valid_loader, loss_fn, optimizer,
                        epochs=epochs, device=DEVICE)

        # Evaluate
        print("Evaluating...")
        predictions, test_loss, test_metrics = evaluate_model(
            model, test_loader, loss_fn, DEVICE, class_rgb_values,
            num_pred=num_pred
        )

        # Store metrics
        test_metrics["fold"] = fold_idx
        test_metrics["test_folds"] = test_fold_ids
        test_metrics["loss"] = test_loss
        all_test_metrics.append(test_metrics)

    # Results summary
    df = pd.DataFrame(all_test_metrics)
    print("\n=== All Fold Test Metrics ===")
    print(df)

    return df
