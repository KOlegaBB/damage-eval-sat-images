from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def semantic_to_instance_mask(mask_rgb, target_rgb=(255, 255, 255)):
    """
    Convert an RGB mask to an instance segmentation mask.

    Args:
        mask_rgb (numpy.ndarray): RGB mask (H, W, 3).
        target_rgb (tuple): Target RGB value to isolate (e.g., (255, 255, 255)).

    Returns:
        instance_mask (numpy.ndarray): Instance segmentation mask with unique labels for each instance.
    """
    # Create a binary mask for the target class
    binary_mask = np.all(mask_rgb == np.array(target_rgb), axis=-1).astype(
        np.uint8)  # Shape: (H, W)

    # Initialize the instance mask
    instance_mask = np.zeros_like(binary_mask, dtype=np.int32)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Assign unique IDs to each contour
    for instance_id, contour in enumerate(contours, start=1):
        cv2.drawContours(instance_mask, [contour], -1, instance_id,
                         thickness=cv2.FILLED)

    return instance_mask, contours


def visualize_instance_mask(instance_mask):
    """
    Visualize the instance segmentation mask

    Args:
        instance_mask (numpy.ndarray): Instance segmentation mask (H, W) with unique labels.
    """
    # Generate a color map for all possible labels
    unique_ids = np.unique(instance_mask)
    color_map = np.zeros((unique_ids.max() + 1, 3), dtype=np.uint8)
    for i in unique_ids:
        if i == 0:  # Background
            color_map[i] = [0, 0, 0]
        else:
            color_map[i] = np.random.randint(0, 255, size=3)

    # Apply the color map to the instance mask
    instance_rgb = color_map[instance_mask]

    return instance_rgb


def process_predictions_to_instance_masks(predictions,
                                          target_rgb=(255, 255, 255)):
    """
    Process predictions to generate instance segmentation masks for all images.

    Args:
        predictions (list): List of predictions, where each element contains
                            [initial_image, true_mask, predicted_mask].
        target_rgb (tuple): RGB color value representing the class to segment (default is (255, 255, 255)).

    Returns:
        results (list): List of dictionaries with keys 'image', 'true_instance_mask', and 'predicted_instance_mask'.
    """
    results = []
    for prediction in predictions:
        initial_image = prediction[0]
        true_mask, _ = semantic_to_instance_mask(prediction[1],
                                                 target_rgb=target_rgb)
        predicted_mask, _ = semantic_to_instance_mask(prediction[2],
                                                      target_rgb=target_rgb)

        results.append({
            "image": initial_image,
            "true_instance_mask": true_mask,
            "predicted_instance_mask": predicted_mask,
        })

    return results


def visualize_instance_results(results):
    """
    Visualize all results with initial image, true instance mask, and predicted instance mask.

    Args:
        results (list): List of dictionaries with keys 'image', 'true_instance_mask', and 'predicted_instance_mask'.
    """
    for result in results:
        image = result["image"]
        true_mask_rgb = visualize_instance_mask(result["true_instance_mask"])
        predicted_mask_rgb = visualize_instance_mask(
            result["predicted_instance_mask"])

        # Plot the results
        plt.figure(figsize=(15, 5))

        # Initial image
        plt.subplot(1, 3, 1)
        plt.title("Initial Image")
        plt.imshow(image)
        plt.axis("off")

        # True instance mask
        plt.subplot(1, 3, 2)
        plt.title("True Instance Mask")
        plt.imshow(true_mask_rgb)
        plt.axis("off")

        # Predicted instance mask
        plt.subplot(1, 3, 3)
        plt.title("Predicted Instance Mask")
        plt.imshow(predicted_mask_rgb)
        plt.axis("off")

        plt.show()


def iou(true_mask, pred_mask):
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Args:
        true_mask (numpy.ndarray): True binary mask.
        pred_mask (numpy.ndarray): Predicted binary mask.

    Returns:
        float: IoU value between 0 and 1.
    """
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0


def get_contour_bounding_boxes(contours):
    """
    Get bounding boxes for a list of contours.

    Args:
        contours (list): List of contours from cv2.findContours.

    Returns:
        list: List of bounding boxes [(x, y, w, h)].
    """
    return [cv2.boundingRect(contour) for contour in contours]


def filter_contours_by_bbox(true_contours, pred_contours, true_bboxes,
                            pred_bboxes):
    """
    Filter contours by checking bounding box intersections.

    Args:
        true_contours (list): True contours.
        pred_contours (list): Predicted contours.
        true_bboxes (list): Bounding boxes for true contours.
        pred_bboxes (list): Bounding boxes for predicted contours.

    Returns:
        list: Filtered true and predicted contour pairs.
    """
    filtered_pairs = []
    for i, true_bbox in enumerate(true_bboxes):
        filtered_buildings = []
        for j, pred_bbox in enumerate(pred_bboxes):
            # Check if bounding boxes overlap
            if (
                    true_bbox[0] < pred_bbox[0] + pred_bbox[2] and
                    true_bbox[0] + true_bbox[2] > pred_bbox[0] and
                    true_bbox[1] < pred_bbox[1] + pred_bbox[3] and
                    true_bbox[1] + true_bbox[3] > pred_bbox[1]
            ):
                filtered_buildings.append((true_contours[i], pred_contours[j]))
        filtered_pairs.append(filtered_buildings)
    return filtered_pairs


def evaluate_instance_masks(predictions, target_rgb=(255, 255, 255),
                            iou_threshold=0.5):
    """
    Process predictions to generate instance segmentation masks and calculate metrics for all images.

    Args:
        predictions (list): List of predictions, where each element contains
                            [initial_image, true_mask, predicted_mask].
        target_rgb (tuple): RGB color value representing the class to segment.
        iou_threshold (float): IoU threshold to consider a building successfully identified.

    Returns:
        metrics (dict): Aggregated metrics for all images.
        results (list): Detailed results per image with errors per building.
    """
    total_target_buildings = 0
    total_predicted_buildings = 0
    total_successful_buildings = 0
    total_iou_sum = 0
    total_iou_count = 0

    results = []
    for prediction in tqdm(predictions):
        initial_image = prediction[0]
        true_mask, true_contours = semantic_to_instance_mask(prediction[1],
                                                             target_rgb=target_rgb)
        predicted_mask, pred_contours = semantic_to_instance_mask(
            prediction[2], target_rgb=target_rgb)

        # Get bounding boxes for filtering
        true_bboxes = get_contour_bounding_boxes(true_contours)
        pred_bboxes = get_contour_bounding_boxes(pred_contours)

        # Filter contours using bounding box intersection
        filtered_pairs = filter_contours_by_bbox(true_contours, pred_contours,
                                                 true_bboxes, pred_bboxes)

        # Metrics initialization
        target_buildings = len(true_contours)
        predicted_buildings = len(pred_contours)
        successful_buildings = 0
        iou_per_building = []

        for filtered_buildings in filtered_pairs:
            ious = []
            for true_contour, pred_contour in filtered_buildings:
                # Create masks for the filtered contours
                true_building_mask = np.zeros_like(true_mask, dtype=np.uint8)
                pred_building_mask = np.zeros_like(predicted_mask,
                                                   dtype=np.uint8)

                cv2.drawContours(true_building_mask, [true_contour], -1, 255,
                                 thickness=cv2.FILLED)
                cv2.drawContours(pred_building_mask, [pred_contour], -1, 255,
                                 thickness=cv2.FILLED)

                # Calculate IoU
                iou_value = iou(true_building_mask, pred_building_mask)
                ious.append(iou_value)

            if ious:
                max_iou = max(ious)
                iou_per_building.append(max_iou)
                if max_iou >= iou_threshold:
                    successful_buildings += 1

        # Aggregate metrics
        total_target_buildings += target_buildings
        total_predicted_buildings += predicted_buildings
        total_successful_buildings += successful_buildings
        total_iou_sum += sum(iou_per_building)
        total_iou_count += len(iou_per_building)

        # Append results for this image
        results.append({
            "image": initial_image,
            "true_instance_mask": true_mask,
            "predicted_instance_mask": predicted_mask,
            "errors_per_building": iou_per_building,
            "successful_buildings": successful_buildings,
            "total_target_buildings": target_buildings,
            "total_predicted_buildings": predicted_buildings,
        })

    # Compute overall metrics
    average_iou_per_building = total_iou_sum / total_iou_count if total_iou_count > 0 else 0
    metrics = {
        "total_target_buildings": total_target_buildings,
        "total_predicted_buildings": total_predicted_buildings,
        "total_successful_buildings": total_successful_buildings,
        "average_iou_per_building": average_iou_per_building,
        "overall_success_rate": total_successful_buildings / total_target_buildings if total_target_buildings > 0 else 0,
        "overall_error_rate": (
                                          total_predicted_buildings - total_successful_buildings) / total_predicted_buildings if total_predicted_buildings > 0 else 0,
    }

    return metrics, results
