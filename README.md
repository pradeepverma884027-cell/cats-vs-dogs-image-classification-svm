# 🐱🐶 Cats vs Dogs Image Classification using HOG + SVM

A Machine Learning project that classifies images of cats and dogs using
Histogram of Oriented Gradients (HOG) feature extraction and a
Support Vector Machine (SVM) classifier.

The project includes:

- HOG Feature Extraction
- StandardScaler
- Pipeline
- GridSearchCV
- Cross Validation
- Confusion Matrix
- Professional Prediction System

## 📖 Project Overview

This project demonstrates a complete image classification pipeline
using classical Machine Learning techniques.

Instead of using Deep Learning, the project extracts HOG features
from images and trains a Support Vector Machine classifier.

The model is optimized using GridSearchCV and evaluated using
cross-validation, confusion matrix, and classification report.

## 🚀 Features

- Image Preprocessing
- HOG Feature Extraction
- Support Vector Machine (SVM)
- StandardScaler
- Pipeline
- GridSearchCV
- Cross Validation
- Confusion Matrix
- Classification Report
- Prediction on Custom Images

## 📂 Dataset

Dataset Used:

Dogs vs Cats (Kaggle)

Contains:

- 25,000 training images
- Cat images
- Dog images

Each image is resized to 64×64 before feature extraction.
## 📁 Project Structure

```text
Cats_Dogs_SVM/
│
├── dataset/
├── models/
├── outputs/
├── screenshots/
├── test_images/
│
├── config.py
├── utils.py
├── train.py
├── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn
- Scikit-Image
- Matplotlib
- Joblib

Dataset
      │
      ▼
Read Images
      │
      ▼
Resize Images
      │
      ▼
Grayscale Conversion
      │
      ▼
HOG Feature Extraction
      │
      ▼
Pipeline
      │
      ├── StandardScaler
      └── SVM
      │
      ▼
GridSearchCV
      │
      ▼
Prediction

## 🏋 Training

Run

```bash
python train.py
```

This will

- preprocess images
- train the pipeline
- tune hyperparameters
- save the trained model

## 🔍 Prediction

Run

```bash
python predict.py
```

The application will

- load the trained pipeline
- preprocess the input image
- classify it as Cat or Dog

## 📊 Results

The model is evaluated using

- Accuracy
- Cross Validation
- Classification Report
- Confusion Matrix

Example

Accuracy : 92%

Cross Validation Accuracy : 91%

Best Parameters

Kernel : RBF

C : 10

Gamma : 0.01

## 👨‍💻 Author

Pradeep Kumar

Machine Learning | Artificial Intelligence | Python