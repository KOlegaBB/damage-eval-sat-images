import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --------- Utility Functions ---------
def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F1 score, and accuracy.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

# --------- Siamese ResNet Training ---------
def train_siamese_model(
    model, train_loader, valid_loader, loss_fn, optimizer, lr_scheduler,
    epochs=12, device="cuda", start_save_epoch=0, save_path="", save_interval=10
):
    train_logs_list = []
    val_logs_list = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()

        train_losses = []
        train_true_labels, train_pred_labels = [], []

        for pre_images, post_images, labels in tqdm(train_loader, desc="Training"):
            pre_images, post_images, labels = pre_images.to(device), post_images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(pre_images, post_images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, preds = torch.max(outputs, 1)
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        train_loss = np.mean(train_losses)
        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(train_true_labels, train_pred_labels)

        # --- Validation ---
        model.eval()
        val_losses, val_true_labels, val_pred_labels = [], [], []

        with torch.no_grad():
            for pre_images, post_images, labels in tqdm(valid_loader, desc="Validation"):
                pre_images, post_images, labels = pre_images.to(device), post_images.to(device), labels.to(device)

                outputs = model(pre_images, post_images)
                loss = loss_fn(outputs, labels)

                val_losses.append(loss.item())
                _, preds = torch.max(outputs, 1)
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(val_true_labels, val_pred_labels)

        train_logs_list.extend(train_losses)
        val_logs_list.extend(val_losses)

        print(f"Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        if save_path and (epoch + 1 + start_save_epoch) % save_interval == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, f"{save_path}/model_SiameseResNet_epoch{epoch + 1 + start_save_epoch}.pt")

        lr_scheduler.step(val_loss)

    return train_logs_list, val_logs_list


# --------- DamageNet Training ---------
def train_damagenet_model(
    model, train_loader, valid_loader, loss_fn, optimizer, lr_scheduler,
    epochs=12, device="cuda", mode='classification',
    start_save_epoch=0, save_path="", save_interval=10
):
    train_logs_list, val_logs_list = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_losses, train_true_labels, train_pred_labels = [], [], []

        for pre_images, post_images, mask_images, labels in tqdm(train_loader, desc="Training"):
            pre_images, post_images, mask_images, labels = (
                pre_images.to(device), post_images.to(device),
                mask_images.unsqueeze(1).to(device), labels.to(device)
            )

            optimizer.zero_grad()
            outputs = model(pre_images, post_images, mask_images)

            if mode == 'regression':
                labels = labels.float()
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()

            elif mode == 'regression_sigmoid':
                labels = labels.float()
                outputs = torch.sigmoid(outputs) * 3  # scale sigmoid output to [0, 3]
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()

            else:  # classification
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(1)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        train_loss = np.mean(train_losses)
        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(train_true_labels, train_pred_labels)

        # --- Validation ---
        model.eval()
        val_losses, val_true_labels, val_pred_labels = [], [], []

        with torch.no_grad():
            for pre_images, post_images, mask_images, labels in tqdm(valid_loader, desc="Validation"):
                pre_images, post_images, mask_images, labels = (
                    pre_images.to(device), post_images.to(device),
                    mask_images.unsqueeze(1).to(device), labels.to(device)
                )

                outputs = model(pre_images, post_images, mask_images)

                if mode == 'regression':
                    labels = labels.float()
                    loss = loss_fn(outputs, labels)
                    preds = outputs.round().clamp(0, 3).long()

                elif mode == 'regression_sigmoid':
                    labels = labels.float()
                    outputs = torch.sigmoid(outputs) * 3
                    loss = loss_fn(outputs, labels)
                    preds = outputs.round().clamp(0, 3).long()

                else:  # classification
                    loss = loss_fn(outputs, labels)
                    preds = outputs.argmax(1)

                val_losses.append(loss.item())
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(val_true_labels, val_pred_labels)

        train_logs_list.extend(train_losses)
        val_logs_list.extend(val_losses)

        print(f"Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        if save_path and (epoch + 1 + start_save_epoch) % save_interval == 0:
            checkpoint = {
                "epoch": epoch + 1 + start_save_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, f"{save_path}/model_DamageNet_{mode}_epoch{epoch + 1 + start_save_epoch}.pt")

        lr_scheduler.step(val_loss)

    return train_logs_list, val_logs_list

