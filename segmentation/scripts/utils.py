import numpy as np

import matplotlib.pyplot as plt


def one_hot_encode(label, label_values):
    """
    Converts a semantic segmentation mask into a one-hot encoded format.

    This function takes a label (segmentation mask) and converts it into a one-hot encoded representation
    where each pixel has a separate channel corresponding to each class label.

    Args:
        label (numpy.ndarray): The input label image of shape (height, width, 3), where each pixel contains
                               the RGB value representing a class.
        label_values (list of tuple): List of RGB values corresponding to the class labels.

    Returns:
        numpy.ndarray: A one-hot encoded version of the input label with shape (height, width, num_classes).
    """
    semantic_map = []
    for colour in label_values:
        # Check for equality between each pixel and the color of the current class
        equality = np.equal(label, colour)
        # Combine the equality for all RGB channels to form a binary map for the class
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    # Stack all class maps into a multi-channel output
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    """
    Converts a one-hot encoded image back into class indices.

    Given an image in one-hot encoded format, this function converts it back to a 2D array of class indices,
    where each pixel is assigned the index of the class with the highest value.

    Args:
        image (numpy.ndarray): The one-hot encoded image of shape (height, width, num_classes).

    Returns:
        numpy.ndarray: A 2D array of shape (height, width), where each pixel contains the index of the class.
    """
    x = image.transpose(1, 2, 0).astype('float32')
    x = np.argmax(x, axis=-1)
    return x


def to_tensor(x, **kwargs):
    """
    Converts an image from HWC format (Height, Width, Channels) to CHW format (Channels, Height, Width),
    and ensures it has a float32 data type.

    Args:
        x (numpy.ndarray): The input image of shape (height, width, channels).

    Returns:
        numpy.ndarray: The converted image of shape (channels, height, width) with dtype 'float32'.
    """
    return x.transpose(2, 0, 1).astype('float32')


def label_to_rgb(label, class_rgb_values):
    """
    Converts a label map (class indices) into an RGB image.

    Args:
        label (numpy.ndarray): The input label map of shape (height, width), where each pixel contains a class index.
        class_rgb_values (list of tuple): List of RGB tuples representing the color for each class label.

    Returns:
        numpy.ndarray: An RGB image of shape (height, width, 3) where each pixel contains the RGB value
                        corresponding to its class index.
    """
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


def visualize_predictions(predictions, num_images=10):
    """
    Visualizes predictions by displaying the input image, true mask, predicted mask,
    and optionally the regularized predicted mask.

    Args:
        predictions (list of tuples): Each tuple should be of the form:
            (image, true_mask, pred_mask) or
            (image, true_mask, pred_mask, reg_pred_mask)
        num_images (int, optional): Number of predictions to visualize. Defaults to all.
    """
    num_images = num_images or len(predictions)
    for i in range(min(num_images, len(predictions))):
        sample = predictions[i]
        has_regularization = len(sample) == 4

        fig, axes = plt.subplots(1, 4 if has_regularization else 3, figsize=(16, 6))

        axes[0].imshow(sample[0])
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(sample[1])
        axes[1].set_title("True Mask")
        axes[1].axis("off")

        axes[2].imshow(sample[2])
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        if has_regularization:
            axes[3].imshow(sample[3])
            axes[3].set_title("Pred Mask (After Reg.)")
            axes[3].axis("off")

        plt.tight_layout()
        plt.show()

