# Impact of Segmentation Preprocessing on Food Image Classification

This project evaluates whether semantic segmentation preprocessing improves the performance of convolutional neural networks (CNNs) for food image classification. We compare two identical ResNet-50 models trained on the Food-101 dataset:
1. Raw RGB images
2. Segmented images with background removed via Google’s Mobile Food Segmenter (DeepLab-v3 + MobileNetV2).

## Dataset
- **Food-101**: 101,000 images across 101 food categories
- Official dataset: [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

## Method
Both models were trained under identical conditions:
- Architecture: ResNet-50
- Optimizer: Adam (lr=3e-4)
- Epochs: 15
- Batch size: 64
- Data augmentations: random cropping, horizontal flip, color jitter, Gaussian noise, rotations

The segmentation pipeline uses Google’s Mobile Food Segmenter to remove background pixels and isolate the main food object.

## Results
| Model      | Val Loss | Top-1 Accuracy | Top-5 Accuracy |
|------------|----------|----------------|----------------|
| Raw RGB    | 0.7282   | 80.1%          | 95.2%          |
| Segmented  | TODO     | TODO           | TODO           |
