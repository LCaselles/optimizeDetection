### Project Name: Time inference of car detection model

### Description:

This project optimizes object detection, specifically car detection, through model quantization. It comprises two primary components:

1. **Model Quantization (`optimize_model_quantization.py`):**
   - Creates a quantized version of the Faster R-CNN MobileNet V3 Large FPN model.
   - Applies quantization (specifically using INT8 data type) to Linear and Conv2D layers.
   - Aims to significantly accelerate inference speed while maintaining accuracy.

2. **Inference and Evaluation (`test_inferences.py`):**
   - Performs inference on a complete dataset using both the original and quantized models.
   - Calculates and compares various detection metrics (Recall, Precision, Mean IoU) to assess the impact of quantization on performance.
   - Measures and reports inference time for both models, quantifying the speedup achieved through quantization.

### Features:
* Quantized Faster R-CNN MobileNet V3 Large FPN model (`optimize_model_quantization.py`).
* Inference and evaluation on a full dataset (`test_inferences.py`).
* Detection metric computation (Recall, Precision, Mean IoU).
* Inference time measurement and comparison.

### Prerequisites
The prerequisites to be able to launch the scripts to optimize the detection model and run the inferences is a conda environnement with Python 3.9

### Installation:
The installation can be done using the pip requirement file provided:
```
pip install -r requirements.txt
```

### Usage:
To run the model quantization, run the following scripts that will save the quantized model in the repertory ```models```:
```
python optimize_model_quantization.py --path_img path_to_images
```
With ```path/to/images/``` being the the path to the input images

Then you can run the inference on the provided dataset with the following script:
```
python test_inferences.py --path_img path/to/images/ --path_gt path/to/ground_truth/ --path_models path/to/models --dump_path_quantized predictions_quantized --dump_path_non_quantized predictions --path_quantized_model quantized_model_full.pth --path_non_quantized_model non_quantized_model_full.pth
```
With ```path/to/images/``` being the the path to the input images of the dataset and ```path/to/ground_truth/``` the path the GT labels of the dataset.

It will display the metrics and the metrics for the quantized/non quantized model.



