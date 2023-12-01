# APS360-Group35
# Mask Detection Dataset
The project utilizes data from three significant datasets:

- MAFA Dataset
- WIDER FACE Dataset
- Kaggle Face Mask Detection Dataset

The training set includes 6,819 images, while the validation set comprises 1,989 images.

### Data Sources and Dataset Composition
- Training Set: 6,819 images (3,006 from MAFA, 3,114 from WIDER Face, 709 from Kaggle)
- Validation Set: 1,989 images (1,059 from MAFA, 780 from WIDER Face, 150 from Kaggle)

### XML File Annotations
Each image in these datasets is accompanied by an XML file. These files include details such as the image's file name, path, size, and bounding box coordinates for faces.

### Preprocessing and Augmentation Techniques
Several preprocessing steps are implemented to enhance the model's performance:
- Photometric Distortions
- Random Cropping
- Horizontal Flipping
- Standardizing Image Sizes
- Normalization using ImageNet values

## Data Splitting
The datasets are split as follows:
- Training: ~70%
- Validation: ~15%
- Testing: ~15%

The split is index-based due to varying face counts in images. Index 900 in the validation dataset marks the start of the test dataset.

## Acknowledgements
This project utilizes datasets from the following sources:
- MAFA Dataset: [MAFA Dataset](http://www.escience.cn/people/geshiming/mafa.html)
- WIDER FACE Dataset: [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)
- Kaggle Face Mask Detection Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/discussion)

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


