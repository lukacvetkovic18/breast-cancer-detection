# Breast Cancer Detection using Deep Learning

This project focuses on detecting breast cancer by leveraging deep learning techniques. Various deep learning models, including ResNet, MobileNet, and VGG16, are used to classify images as either "healthy" or "sick." The dataset is comprised of breast cancer images that are preprocessed and fed into these models for training, validation, and testing.

## About The Project

Breast cancer detection remains one of the most critical challenges in medical image analysis. With the advancement of deep learning techniques, the accuracy and speed of detection can be significantly improved. This project aims to classify breast cancer images into "healthy" or "sick" categories using convolutional neural networks (CNNs).

### Built With
The following technologies are used in this project:
- **Python**
- **Keras**: For building and training deep learning models
- **Pandas & Numpy**: For data manipulation
- **Matplotlib & Seaborn**: For visualization
- **OpenCV**: For image processing
- **ResNet, MobileNet, VGG16, DenseNet**: Pre-trained deep learning models

## Dataset

The dataset used in this project consists of labeled breast cancer images. The images are categorized into two labels:
- **Healthy**
- **Sick**

The dataset is loaded from a local directory, and images are resized to 128x128 for training.

**Example image format**:
```python
{
  "category": "sick",
  "image_size": "128x128"
}
```

## Usage
Once you have the environment set up and the dataset available, run the following steps to start the model training process:
- Preprocess the images by resizing them to 128x128.
- Split the data into training, validation, and testing sets.
- Train the model using one of the pre-trained architectures (ResNet, VGG16, etc.).
- Evaluate the model using metrics such as confusion matrix, ROC curve, and accuracy.

### Key Model Steps:
- **Image Preprocessing:** Loading and resizing images.
- **Data Augmentation:** Using ImageDataGenerator for augmenting the dataset.
- **Model Training:** Training CNNs like ResNet, VGG16, MobileNet, and DenseNet.
- **Model Evaluation:** Evaluating model performance using confusion matrix, ROC, and accuracy.