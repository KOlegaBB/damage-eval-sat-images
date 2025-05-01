import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# ------------- General Utilities -------------
CLASS_NAMES = ["No Damage", "Minor", "Major", "Destroyed"]
SIZE_GROUP_ORDER = ["Very Small(<100)", "Small(<200)", "Average(<500)",
                    "Large(<1000)", "Very Large(1000+)"]


def categorize_size(area):
    """
    Categorizes the size of an object based on its area.

    Args:
    - area (float): The area of the object.

    Returns:
    - str: A category representing the size of the object.
    """
    if area < 100:
        return "Very Small(<100)"
    elif area < 200:
        return "Small(<200)"
    elif area < 500:
        return "Average(<500)"
    elif area < 1000:
        return "Large(<1000)"
    else:
        return "Very Large(1000+)"


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots the confusion matrix using a heatmap for better visualization.

    Args:
    - cm (np.array): The confusion matrix to plot.
    - title (str): The title for the plot.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


# ------------- Main Evaluation Functions -------------
def evaluate_siamese(model, test_loader, device, mode='classification'):
    """
    Evaluates the performance of the Siamese network model.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - test_loader (DataLoader): The DataLoader for the test dataset.
    - device (torch.device): The device (CPU or GPU) to run the model on.
    - mode (str): The mode of evaluation, either 'classification' or 'regression'.

    Returns:
    - dict: A dictionary containing various evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    unprocessed_preds = []  # For raw predictions (before rounding or argmax)
    losses = []

    loss_fn = torch.nn.CrossEntropyLoss() if mode == 'classification' else torch.nn.MSELoss()

    with torch.no_grad():
        for images_1, images_2, labels in tqdm(test_loader,
                                               desc="Evaluating Siamese"):
            images_1, images_2, labels = images_1.to(device), images_2.to(
                device), labels.to(device)

            outputs = model(images_1, images_2)

            if mode == 'regression':
                labels = labels.float()
                preds = outputs.squeeze().round().clamp(0,
                                                        3).long()  # Round predictions for regression
                unprocessed_preds.extend(
                    outputs.squeeze().cpu().numpy())  # Store raw regression outputs

            elif mode == 'classification':
                preds = torch.argmax(outputs,
                                     dim=1)  # Classification mode (argmax)
                unprocessed_preds.extend(
                    preds.cpu().numpy())  # Store raw logits/probabilities for classification

            losses.append(loss_fn(outputs, labels).item())  # Add loss
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert unprocessed_preds and all_labels to numpy arrays
    unprocessed_preds = np.array(unprocessed_preds)
    all_labels = np.array(all_labels)
    # Calculate MSE using raw predictions and true labels
    mse = np.mean((unprocessed_preds - all_labels) ** 2)
    mae = np.mean(np.abs(unprocessed_preds - all_labels))

    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\nSiamese Evaluation ({mode.title()}):")
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, title=f"Siamese - {mode.title()}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mse": mse,
        "mae": mae
    }


def evaluate_damagenet(model, test_loader, device, mode='classification'):
    """
    Evaluates the performance of the DamageNet model on the test data.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - test_loader (DataLoader): The DataLoader for the test dataset.
    - device (torch.device): The device (CPU or GPU) to run the model on.
    - mode (str): The mode of evaluation, either 'classification' or 'regression'.

    Returns:
    - dict: A dictionary containing various evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    unprocessed_preds = []  # For raw predictions (before rounding or argmax)
    losses = []

    loss_fn = torch.nn.CrossEntropyLoss() if mode == 'classification' else torch.nn.MSELoss()

    with torch.no_grad():
        for pre_images, post_images, mask_images, labels in tqdm(test_loader,
                                                                 desc="Evaluating DamageNet"):
            pre_images, post_images, mask_images, labels = (
                pre_images.to(device),
                post_images.to(device),
                mask_images.unsqueeze(1).to(device),
                labels.to(device)
            )

            outputs = model(pre_images, post_images, mask_images)

            if mode == 'regression':
                labels = labels.float()
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()
                unprocessed_preds.extend(
                    outputs.squeeze().cpu().numpy())  # Store raw regression outputs

            elif mode == 'regression_sigmoid':
                labels = labels.float()
                outputs = torch.sigmoid(
                    outputs) * 3  # scale sigmoid output to [0, 3]
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()
                unprocessed_preds.extend(
                    outputs.squeeze().cpu().numpy())  # Store raw regression outputs

            else:  # classification
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(1)
                unprocessed_preds.extend(
                    preds.cpu().numpy())  # Store raw classification outputs

            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)

    # Calculate MSE using raw predictions and true labels
    unprocessed_preds = np.array(unprocessed_preds)
    all_labels = np.array(all_labels)
    mse = np.mean((unprocessed_preds - all_labels) ** 2)
    mae = np.mean(np.abs(unprocessed_preds - all_labels))

    # Calculate accuracy for regression (using rounded predictions) and for classification
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate precision, recall, and F1 score for classification only
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Print results
    print(f"\nDamageNet Evaluation ({mode.title()}):")
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  MAE:       {mae:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm,
                          title=f"Confusion Matrix (DamageNet - {mode.title()})")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mse": mse,
        "mae": mae
    }


# ------------- Error Analysis by Size -------------
def analyze_model_errors_by_size(model, test_loader, device,
                                 mode='classification'):
    """
    Analyzes the model's performance by calculating prediction errors and categorizing them by object size.

    Args:
    - model (torch.nn.Module): The model to evaluate. It expects pre-images, post-images, and mask images as inputs.
    - test_loader (DataLoader): The DataLoader for the test dataset. Each batch contains pre-images, post-images,
      mask images, labels, and object areas.
    - device (torch.device): The device (CPU or GPU) to run the model on.
    - mode (str): The evaluation mode. Can be 'classification', 'regression', or 'regression_sigmoid'. Defaults to 'classification'.

    Returns:
    - pd.DataFrame: A DataFrame containing the following columns:
      - "pred": The predicted values.
      - "label": The true labels.
      - "area": The object sizes in pixels.
      - "error_gap": The absolute error between predicted and true values.
      - "correct": A boolean indicating if the prediction is correct based on a threshold or equality.
      - "size_group": A categorized size group based on the object area.

    Example:
        df = analyze_model_errors_by_size(model, test_loader, device, mode='classification')
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_areas = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pre_images, post_images, mask_images, labels, areas = batch
            pre_images = pre_images.to(device)
            post_images = post_images.to(device)
            mask_images = mask_images.to(device).unsqueeze(1)
            labels = labels.to(device)

            outputs = model(pre_images, post_images, mask_images)

            if mode == 'regression':
                preds = outputs.squeeze().cpu().numpy()
                labels = labels.cpu().numpy()

            elif mode == 'regression_sigmoid':
                outputs = torch.sigmoid(outputs) * 3
                preds = outputs.squeeze().cpu().numpy()
                labels = labels.cpu().numpy()

            else:  # classification
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_areas.extend(areas.numpy())

    df = pd.DataFrame({
        "pred": all_preds,
        "label": all_labels,
        "area": all_areas
    })

    df["error_gap"] = np.abs(df["pred"] - df["label"]) if mode.startswith(
        "regression") else np.abs(df["pred"] - df["label"]).astype(int)
    df["correct"] = (df["error_gap"] < 0.5) if mode.startswith(
        "regression") else (df["pred"] == df["label"])
    df["size_group"] = pd.Categorical(df["area"].apply(categorize_size),
                                      categories=SIZE_GROUP_ORDER,
                                      ordered=True)

    return df


def plot_error_rate_by_size_group(df):
    """
    Plots the error rate by size group.

    Args:
    - df (pd.DataFrame): A DataFrame containing the evaluation results with columns 'size_group' and 'correct'.

    Returns:
    - None: Displays a bar plot showing the error rate by size group.
    """
    grouped = df.groupby("size_group", observed=True)["correct"]
    error_rates = grouped.apply(lambda x: 1 - x.mean()).reset_index(name="error_rate")
    error_rates["count"] = grouped.count().values

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=error_rates, x="size_group", y="error_rate", palette="viridis")
    for i, row in error_rates.iterrows():
        ax.text(i, row["error_rate"] + 0.02, f'n={row["count"]}', ha='center', fontsize=9)

    plt.title("Error Rate by Size Group")
    plt.xlabel("Size Group")
    plt.ylabel("Error Rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_gap_vs_size(df, mode='classification'):
    """
    Plots the relationship between the error gap and object size.

    Args:
    - df (pd.DataFrame): A DataFrame containing the evaluation results with columns 'error_gap' and 'area'.
    - mode (str): The evaluation mode. Can be 'classification' or 'regression'. Defaults to 'classification'.

    Returns:
    - None: Displays a scatter plot (for regression) or strip plot (for classification) showing error gap vs. object size.
    """
    plt.figure(figsize=(10, 10))
    if mode.startswith("regression"):
        sns.scatterplot(data=df, x="error_gap", y="area", hue="error_gap",
                        palette="coolwarm", size="error_gap", legend=False)
        plt.xlabel("Absolute Error (|Predicted - True|)")
    else:
        sns.stripplot(data=df, x="error_gap", y="area", hue="error_gap",
                      jitter=0.3, palette="coolwarm", dodge=True)
        plt.xlabel("|Predicted - True Class|")
        plt.legend(title="Error Gap")
    plt.title("Prediction Gap vs Object Size")
    plt.ylabel("Object Size (Pixels)")
    plt.ylim(0, 5000)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_gap_distribution_by_size_group(df):
    """
    Plots the distribution of error gaps by size group.

    Args:
    - df (pd.DataFrame): A DataFrame containing the evaluation results with columns 'size_group' and 'error_gap'.

    Returns:
    - None: Displays a line plot showing the distribution of error gaps by size group.
    """
    total_per_group = df.groupby("size_group", observed=True).size().rename("total").reset_index()
    gap_counts = df.groupby(["size_group", "error_gap"], observed=True).size().rename("count").reset_index()
    merged = gap_counts.merge(total_per_group, on="size_group")
    merged["percent"] = 100 * merged["count"] / merged["total"]

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=merged, x="size_group", y="percent", hue="error_gap", marker='o', palette="coolwarm")

    for _, row in total_per_group.iterrows():
        x = row["size_group"]
        y = merged[merged["size_group"] == x]["percent"].max()
        ax.text(x, y + 2, f'n={int(row["total"])}', ha='center', fontsize=9)

    plt.title("Distribution of Error Gaps by Size Group")
    plt.xlabel("Size Group")
    plt.ylabel("Error Gap Percentage (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Error Gap |Pred - True|")
    plt.show()
