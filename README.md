# APS360-Group35
# Mask Detection Dataset

This repository contains a dataset categorized into two classes: 'with_mask' and 'without_mask'. The initial dataset consists of 4,316 'with_mask' images and 4,507 'without_mask' images. These images were sourced from publicly accessible datasets on Kaggle and Roboflow Universe, with a diverse representation of age, ethnicity, and lighting conditions.

## Dataset Description

The dataset is split into the following directories:

- `with_mask`: Contains images of individuals wearing masks.
- `without_mask`: Contains images of individuals not wearing masks.
- `with_mask_processed`: Contains 5,000 processed images from the 'with_mask' directory.
- `without_mask_processed`: Contains 5,000 processed images from the 'without_mask' directory.

After processing, the images are further split into training, validation, and test sets, located in the `train`, `val`, and `test` directories, respectively.

## Scripts

There are two main scripts used to process and split the dataset:

- `Data_processing.py`: Processes the original images to generate 5,000 images for each class, ensuring a balanced and varied dataset.
- `Data_set_split.py`: Splits the processed images into training, validation, and test sets with the following distribution:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

## Usage

To process the images, run:

```bash
python Data_processing.py
```
This will create two folders, with_mask_processed and without_mask_processed, each containing 5,000 processed images.

To split the dataset, run:
```bash
python Data_set_split.py
```

This will distribute the processed images into train, val, and test folders for model training and evaluation purposes.

## Acknowledgements
We would like to acknowledge the creators of the original datasets on Kaggle and Roboflow Universe for providing the images used in this collection. Special thanks to the diverse group of individuals who are represented in these images, helping to develop robust machine learning models.

[1] Kaggle Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/

[2] Roboflow Universe Dataset: https://universe.roboflow.com/pyimagesearch/covid-19-pis

# Architecture
## Model
A lite Single Shot MultiBox Detector (SSD) model with approximately 1,043,822 parameters. The model architecture includes several major components to enable accurate object detection and bounding box prediction.
## Components
- `CNN Model`: The model incorporates a Convolutional Neural Network (CNN) as its backbone to extract meaningful features from input images.
- `ResNet Block`: A Residual Network (ResNet) block is integrated into the architecture to facilitate better gradient flow and enable the model to handle deeper networks more effectively.
- `CBAM`: Convolutional Block Attention Module (CBAM) is employed to enhance the model's attention mechanism, allowing it to focus on salient regions and improve feature representation.
- `Classification and Bounding Box Prediction Layers`: The model consists of specific layers dedicated to classifying objects and predicting bounding boxes. These layers aid in identifying different object categories and accurately localizing them within the image.
- `Batch Normalization`: Batch Normalization is applied throughout the model to improve training stability and accelerate convergence.
## Output
The output of this network includes:
- `Anchors`: The model generates anchor boxes at different scales and aspect ratios, which serve as reference points for object detection.
- `cls_preds`: These are the predictions for the class labels of the detected objects.
- `bbox_preds`: These are the predictions for the bounding boxes that enclose the detected objects.
## Loss Function
To train the model, a loss function is utilized that considers both the classification and bounding box predictions. In this case, the chosen loss functions are CrossEntropyLoss for the classification task and L1Loss (mean absolute error) for the bounding box prediction task. These loss functions enable the model to optimize both the accuracy of object classification and the precision of bounding box localization.
## Perfomance
Currently, our model has achieved a low classification error (2.86e-03) and bounding box Mean Absolute Error (MAE) (3.95e-03) on a dataset of only 1000 samples. 


