import numpy as np

import matplotlib.pyplot as plt


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    x = image.transpose(1, 2, 0).astype('float32')
    x = np.argmax(x, axis=-1)
    return x


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def label_to_rgb(label, class_rgb_values):
    label = reverse_one_hot(label)
    rgb_image = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(class_rgb_values):
        rgb_image[label == i] = color
    return rgb_image


def visualize_train_samples(dataloader, class_rgb_values, num_samples=5):
    """
    Visualize training samples with image, mask, and overlap.

    Args:
        dataloader: DataLoader for the training dataset.
        class_rgb_values: List of RGB values for each class.
        num_samples: Number of samples to visualize.
    """
    images, masks, overlaps = [], [], []
    sample_count = 0

    for batch_images, batch_masks in dataloader:
        # Break if we've already collected enough samples
        if sample_count >= num_samples:
            break

        batch_size = batch_images.size(0)

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            image = batch_images[i].permute(1, 2, 0).cpu().numpy()
            mask = batch_masks[i].cpu().numpy()

            # Convert the mask to RGB using the provided class RGB values
            mask_rgb = label_to_rgb(mask, class_rgb_values)

            # Create an overlap image
            overlap = (0.5 * image / 255 + 0.5 * mask_rgb / 255).astype("float32")

            images.append(image / 255)
            masks.append(mask_rgb)
            overlaps.append(overlap)

            sample_count += 1

    # Display results
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(num_samples):
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(masks[i])
        axs[i, 1].set_title("Ground Truth Mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(overlaps[i])
        axs[i, 2].set_title("Image with Mask Overlay")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_test_predictions(predictions):
    num_samples = len(predictions)
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(num_samples):
        image, expected_mask, predicted_mask = predictions[i]

        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(expected_mask)
        axs[i, 1].set_title("Expected Mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(predicted_mask)
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")

    plt.show()
