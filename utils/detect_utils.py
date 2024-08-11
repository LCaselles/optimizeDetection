import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Random colors for each class for visualization
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

def draw_boxes(boxes, classes, labels, image):
    """
    Draw bounding boxes and class labels on the image.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        classes (list): List of class names.
        labels (torch.Tensor): Tensor of class labels.
        image (np.ndarray): The image to draw on.

    Returns:
        np.ndarray: The image with bounding boxes and labels drawn.
    """
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color.tolist(),  # Convert color to list for cv2
            2
        )
        cv2.putText(
            image,
            classes[i],
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color.tolist(),  # Convert color to list for cv2
            2,
            lineType=cv2.LINE_AA
        )
    return image

def predict(image, model, device, detection_threshold):
    """
    Predict bounding boxes and class labels for the given image.

    Args:
        image (PIL.Image.Image or np.ndarray): The input image.
        model (torch.nn.Module): The object detection model.
        device (torch.device): The device to run the model on.
        detection_threshold (float): Score threshold for detections.

    Returns:
        tuple: A tuple containing:
            - boxes (np.ndarray): Array of bounding boxes.
            - pred_classes (list): List of predicted class names.
            - labels (torch.Tensor): Tensor of predicted class labels.
    """
    # Convert image to tensor and add batch dimension
    image = transform(image).to(device).unsqueeze(0)
    
    # Get model predictions
    outputs = model(image)
    
    # Extract class names, scores, and bounding boxes
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    # Filter boxes by detection threshold
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    
    return boxes, pred_classes, outputs[0]['labels']