import numpy as np


def parse_coordinates(file_path):
    """
    Parses a text file containing labels and coordinates.

    Each line in the file should contain a label followed by four integers representing
    the coordinates of a bounding box. The format for each line is as follows:
    
    ```
    Label x_min y_min x_max y_max
    ```
    
    Parameters:
    -----------
    file_path : str
        The path to the text file containing the coordinates to be parsed.

    Returns:
    --------
    list of tuples
        A list where each element is a tuple containing:
        - label (str): The label (e.g., 'Car').
        - coordinates (list of int): A list of four integers [x_min, y_min, x_max, y_max]
          representing the bounding box coordinates.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = []
    for line in lines:
        # Split the line into parts
        parts = line.strip().split()

        # The first part is the label (e.g., "Car")
        label = parts[0]

        # The remaining parts are the coordinates
        coordinates = list(map(int, parts[1:]))

        # Append the parsed data as a tuple (label, coordinates)
        parsed_data.append(coordinates)
    return parsed_data


def compute_iou_matrix(pred_boxes, gt_boxes):
    """
    Computes the Intersection over Union (IoU) matrix between multiple 2D bounding boxes.

    Parameters:
    pred_boxes (list of lists or tuples): List of predicted bounding boxes, where each box is [x1, y1, x2, y2].
    gt_boxes (list of lists or tuples): List of ground truth bounding boxes, where each box is [x1, y1, x2, y2].

    Returns:
    np.ndarray: A matrix of IoU scores where each element [i, j] corresponds to the IoU between
                the i-th prediction box and the j-th ground truth box.
    """
    
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    # Initialize an empty IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    # Iterate over each prediction and ground truth pair
    for i in range(num_preds):
        for j in range(num_gts):
            iou_matrix[i, j] = compute_iou(pred_boxes[i], gt_boxes[j])
    
    return iou_matrix

def compute_iou(pred_box, gt_box):
    """
    Computes the Intersection over Union (IoU) between two 2D bounding boxes.

    Parameters:
    pred_box (list, tuple): Predicted bounding box as [x1, y1, x2, y2]
                            where (x1, y1) is the top-left corner, and
                            (x2, y2) is the bottom-right corner.
    gt_box (list, tuple): Ground truth bounding box as [x1, y1, x2, y2]
                          where (x1, y1) is the top-left corner, and
                          (x2, y2) is the bottom-right corner.

    Returns:
    float: Intersection over Union (IoU) score.
    """
    
    # Unpack the coordinates
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box
    
    # Calculate the intersection coordinates
    x1_inter = max(x1_pred, x1_gt)
    y1_inter = max(y1_pred, y1_gt)
    x2_inter = min(x2_pred, x2_gt)
    y2_inter = min(y2_pred, y2_gt)
    
    # Calculate the area of intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Calculate the area of both the prediction and ground truth boxes
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Calculate the union area
    union_area = pred_area + gt_area - inter_area
    
    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou


def get_scalar_metrics_with_precision_recall(iou_matrix, threshold=0.5):
    """
    Computes scalar metrics, precision, and recall from an IoU matrix.

    Parameters:
    iou_matrix (np.ndarray): A 2D array where each element [i, j] corresponds to the IoU
                             between the i-th prediction box and the j-th ground truth box.
    threshold (float): A threshold for considering a detection as a match (default is 0.5).

    Returns:
    dict: A dictionary with the following metrics:
        - mean_iou: The mean IoU across all pairs.
        - mean_max_iou_per_pred: The mean of the maximum IoU for each prediction.
        - mean_max_iou_per_gt: The mean of the maximum IoU for each ground truth.
        - precision: The ratio of true positives to the total number of predictions.
        - recall: The ratio of true positives to the total number of ground truths.
        - matches_above_threshold: The number of IoUs above the threshold, normalized by the number of predictions.
    """

    # No predictions in GT and Pred
    if iou_matrix.shape[0] == 0 and iou_matrix.shape[1] == 0:
        return {
            "mean_iou": 1,
            "mean_max_iou_per_pred": 1,
            "mean_max_iou_per_gt": 1,
            "precision": 1,
            "recall": 1,
        }
    # No predictions
    elif iou_matrix.shape[0] == 0 and iou_matrix.shape[1] > 0:
        return {
            "mean_iou": 0,
            "mean_max_iou_per_pred": 0,
            "mean_max_iou_per_gt": 0,
            "precision": 1,
            "recall": 0,
        }
    # Predictions but no GT
    elif iou_matrix.shape[1] == 0 and iou_matrix.shape[0] > 0:
        return {
            "mean_iou": 0,
            "mean_max_iou_per_pred": 0,
            "mean_max_iou_per_gt": 0,
            "precision": 0,
            "recall": 1,
        }

    # Mean IoU across all pairs
    mean_iou = np.mean(iou_matrix)
    
    # Mean of the maximum IoU for each prediction
    max_iou_per_pred = np.max(iou_matrix, axis=1)
    mean_max_iou_per_pred = np.mean(max_iou_per_pred)
    
    # Mean of the maximum IoU for each ground truth
    max_iou_per_gt = np.max(iou_matrix, axis=0)
    mean_max_iou_per_gt = np.mean(max_iou_per_gt)
    
    # True positives: Predictions that match any ground truth with IoU above the threshold
    true_positives = np.sum(max_iou_per_pred >= threshold)
    
    # Precision: True positives / total number of predictions
    precision = true_positives / iou_matrix.shape[0]
    
    # Recall: True positives / total number of ground truths
    recall = true_positives / iou_matrix.shape[1]
    
    return {
        "mean_iou": mean_iou,
        "mean_max_iou_per_pred": mean_max_iou_per_pred,
        "mean_max_iou_per_gt": mean_max_iou_per_gt,
        "precision": precision,
        "recall": recall,
    }