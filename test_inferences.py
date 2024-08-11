import os
import time
import fire

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

import utils.detect_utils
from metrics import compute_iou_matrix, get_scalar_metrics_with_precision_recall, parse_coordinates


def inference_inputs(model, path_img, path_gt, dump_path, N_max=None):
    """
    Perform inference on input images and evaluate metrics against ground truth.

    Parameters:
    -----------
    model : torch.nn.Module
        The model used for inference.
    path_img : str
        Path to the directory containing the images.
    path_gt : str
        Path to the directory containing the ground truth labels.
    dump_path : str
        Path to save the images with predictions and ground truth bounding boxes.
    N_max : int, optional
        Maximum number of images to process. If None, process all images.

    Returns:
    --------
    dict
        A dictionary containing the computed metrics for each image.
    """
    metrics = {
        "mean_iou": [],
        "mean_max_iou_per_pred": [],
        "mean_max_iou_per_gt": [],
        "precision": [],
        "recall": [],
    }
    os.makedirs(dump_path, exist_ok=True)

    N = 0

    # Read the images and run inference for detections
    for file in os.listdir(path_img):
        input_image = Image.open(os.path.join(path_img, file))
        boxes, classes, labels = utils.detect_utils.predict(input_image, model, "cpu", 0.7)

        # Filter results to keep only the cars
        len_boxes = boxes.shape[0]
        classes_array = np.array(classes[:len_boxes], dtype=str)
        idx_preds_car = np.where(classes_array == "car")[0]
        boxes = boxes[idx_preds_car]
        labels = np.array(labels[:len_boxes])[idx_preds_car]
        classes = ["car_pred" for _ in range(len(labels))]

        # Load ground truth (GT)
        gt_boxes = parse_coordinates(os.path.join(path_gt, file.split(".png")[0] + ".txt"))
        classes_gt = ["car_gt" for _ in range(len(gt_boxes))]
        labels_gt = 10 * np.ones(len(gt_boxes), dtype=int)

        # Dump prediction images
        image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_BGR2RGB)
        image = utils.detect_utils.draw_boxes(boxes, classes, labels, image)
        image = utils.detect_utils.draw_boxes(np.array(gt_boxes), classes_gt, labels_gt, image)
        save_name = file.split(".png")[0]
        cv2.imwrite(os.path.join(dump_path, f"pred_{save_name}.jpg"), image)

        # Compute metrics
        iou_matrix = compute_iou_matrix(boxes, gt_boxes)
        metric_img = get_scalar_metrics_with_precision_recall(iou_matrix, 0.65)
        for key in metric_img:
            metrics[key].append(metric_img[key])

        N += 1
        if N_max is not None and N >= N_max:
            break

    return metrics


def main(path_img, path_gt, dump_path_quantized="predictions_quantized", 
         dump_path_non_quantized="predictions", 
         path_quantized_model='quantized_model_full.pth', 
         path_non_quantized_model="non_quantized_model_full.pth"):
    """
    Main function to run inference with and without quantization, and print metrics.

    Parameters:
    -----------
    path_img : str
        Path to the directory containing the images.
    path_gt : str
        Path to the directory containing the ground truth labels.
    dump_path_quantized : str, optional
        Path to save the images for quantized model predictions.
    dump_path_non_quantized : str, optional
        Path to save the images for non-quantized model predictions.
    path_quantized_model : str, optional
        Path to the quantized model file.
    path_non_quantized_model : str, optional
        Path to the non-quantized model file.
    """
    # Load the models
    model_quantized = torch.load(path_quantized_model)
    model_quantized.eval()

    model_non_quantized = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model_non_quantized.eval()

    # Inference with quantized model
    print("Start inference with quantization")
    start = time.time()
    metrics = inference_inputs(model_quantized, path_img, path_gt, dump_path_quantized, N_max=15)
    print("Time to compute:", time.time() - start)
    print("Quantized metrics:")
    for metric in metrics.keys():
        print(metric, np.mean(metrics[metric]))

    # Inference with non-quantized model
    print("Start inference without quantization")
    start = time.time()
    metrics = inference_inputs(model_non_quantized, path_img, path_gt, dump_path_non_quantized, N_max=15)
    print("Time to compute:", time.time() - start)
    print("Non-quantized metrics:")
    for metric in metrics.keys():
        print(metric, np.mean(metrics[metric]))

"""
if __name__ == "__main__":
    # Define paths (these can be changed to command-line arguments or configuration files)
    path_img = "dataset/dataset/images/"
    path_gt = "dataset/dataset/labels/"
    main(path_img, path_gt)
"""
if __name__ == "__main__":
    fire.Fire(main)