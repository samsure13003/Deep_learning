# 🧠 Deep Learning Projects

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

A collection of deep learning projects built with **TensorFlow** and **Keras**, covering core neural network architectures — from tabular classification with ANNs to image recognition with CNNs.

---

## 📂 Projects Overview

| # | Project | Type | Dataset | Key Result |
|---|---------|------|---------|------------|
| 1 | [Artificial Neural Network — Bank Churn Prediction](#1-artificial-neural-network--bank-churn-prediction) | Binary Classification | Churn Modelling | Predicts if a customer will leave the bank |
| 2 | [Convolutional Neural Network — Cat vs Dog Classifier](#2-convolutional-neural-network--cat-vs-dog-classifier) | Image Classification | Custom Image Dataset | Classifies animal images as cat or dog |

---

## 1. Artificial Neural Network — Bank Churn Prediction

### 📌 Problem Statement
A bank wants to identify customers who are likely to churn (leave). Using customer demographic and account data, an ANN is trained to predict whether a customer will exit or stay.

### 📊 Dataset
- **File:** `Churn_Modelling.csv`
- **Features:** Geography, Gender, Age, Tenure, Balance, Number of Products, Credit Card status, Active Member status, Estimated Salary
- **Target:** `Exited` (1 = churned, 0 = stayed)

### 🔧 Workflow

```
Data Preprocessing  →  Build ANN  →  Train  →  Evaluate  →  Predict
```

**Preprocessing:**
- Label Encoding for `Gender` column
- One-Hot Encoding for `Geography` column (ColumnTransformer)
- Train/Test Split (80/20)
- Feature Scaling with `StandardScaler`

**Model Architecture:**

```
Input Layer   →  11 features
Hidden Layer 1  →  6 neurons, ReLU
Hidden Layer 2  →  6 neurons, ReLU
Output Layer  →  1 neuron, Sigmoid
```

**Training:**
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Batch Size: `32`
- Epochs: `100`

### 📈 Evaluation
- Confusion Matrix generated on test set
- Accuracy Score reported

### 🔍 Sample Prediction
```python
# Predict for: France | Score:600 | Male | Age:40 | Tenure:3 | Balance:60000 | Products:2 | HasCard:Yes | Active:Yes | Salary:50000
ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
# Output → Customer STAYS in the bank ✅
```

### 🛠️ Libraries Used
`numpy` · `pandas` · `tensorflow` · `sklearn`

---

## 2. Convolutional Neural Network — Cat vs Dog Classifier

### 📌 Problem Statement
Build an image classifier that can distinguish between photos of cats and dogs, using a CNN trained on a custom dataset with data augmentation.

### 📊 Dataset
- **Structure:**
  ```
  dataset/
  ├── training_set/
  │   ├── cats/
  │   └── dogs/
  ├── test_set/
  │   ├── cats/
  │   └── dogs/
  └── single_prediction/
  ```
- Images resized to `64 × 64` pixels

### 🔧 Workflow

```
Image Augmentation  →  Build CNN  →  Train + Validate  →  Single Image Prediction
```

**Preprocessing:**
- Training: Rescaling (1/255) + Shear, Zoom, Horizontal Flip (augmentation)
- Testing: Rescaling only (no augmentation)
- Uses `ImageDataGenerator` + `flow_from_directory`

**Model Architecture:**

```
Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Flatten
Dense (128 neurons, ReLU)
Output Dense (1 neuron, Sigmoid)
```

**Training:**
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Epochs: `25`
- Validated against test set each epoch

### 🔍 Single Image Prediction
```python
# Load and predict on a custom image
result = cnn.predict(test_image)
prediction = 'dog' if result[0][0] > 0.5 else 'cat'
# Displays image with predicted label using matplotlib
```

### 🛠️ Libraries Used
`tensorflow` · `keras` · `numpy` · `matplotlib`

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Run the Notebooks
```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd deep-learning

# Launch Jupyter
jupyter notebook
```

Then open either:
- `artificial_neural_network.ipynb`
- `convolutional_neural_network.ipynb`

---

## 📁 Folder Structure

```
deep-learning/
│
├── artificial_neural_network.ipynb   # ANN — Bank Churn Prediction
├── convolutional_neural_network.ipynb  # CNN — Cat vs Dog Classifier
├── Churn_Modelling.csv               # Dataset for ANN
├── dataset/                          # Image dataset for CNN
│   ├── training_set/
│   ├── test_set/
│   └── single_prediction/
└── README.md
```

---

## 👤 Author

**Samsur**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
