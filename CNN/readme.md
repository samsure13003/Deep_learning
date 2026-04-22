# 🐱🐶 Cat vs Dog Image Classifier — Convolutional Neural Network

A binary image classifier built with TensorFlow/Keras that distinguishes between cats and dogs using a Convolutional Neural Network (CNN).

---

## 📁 Dataset

| Split | Cats | Dogs | Total |
|-------|------|------|-------|
| Training | 4,000 | 4,000 | 8,000 |
| Testing | 1,000 | 1,000 | 2,000 |

Images are organized in the following directory structure:

```
dataset/
├── training_set/
│   ├── cats/        # 4,000 images
│   └── dogs/        # 4,000 images
├── test_set/
│   ├── cats/        # 1,000 images
│   └── dogs/        # 1,000 images
└── single_prediction/
    └── your_image.jpeg
```

---

## 🧠 Model Architecture

The CNN is built using a `Sequential` model with the following layers:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU activation, input shape (64, 64, 3) |
| MaxPooling2D | Pool size 2×2, stride 2 |
| Conv2D | 32 filters, 3×3 kernel, ReLU activation |
| MaxPooling2D | Pool size 2×2, stride 2 |
| Flatten | — |
| Dense | 128 units, ReLU activation |
| Dense (Output) | 1 unit, Sigmoid activation |

- **Loss function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  
- **Epochs:** 25

---

## ⚙️ Data Preprocessing

**Training set** — augmented to improve generalization:
- Rescaled pixel values to `[0, 1]`
- Shear range: `0.2`
- Zoom range: `0.2`
- Horizontal flip: enabled

**Test set** — only rescaled (no augmentation):
- Rescaled pixel values to `[0, 1]`

All images are resized to **64×64 pixels** with a batch size of **32**.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cat-dog-cnn.git
cd cat-dog-cnn
```

### 2. Install dependencies

```bash
pip install tensorflow keras numpy matplotlib
```

### 3. Prepare the dataset

Place your images into the directory structure shown above. The training and test sets should follow the same folder layout.

### 4. Run the notebook

Open `convolutional_neural_network.ipynb` in Jupyter Notebook or JupyterLab and run all cells from top to bottom.

---

## 🔍 Making a Single Prediction

To classify a new image, update the `img_path` variable in the last section of the notebook:

```python
img_path = 'dataset/single_prediction/your_image.jpeg'
```

The notebook will display the image with the predicted label (`cat` or `dog`) as the title.

**Decision threshold:** `> 0.5` → Dog, `≤ 0.5` → Cat (Sigmoid output).

---

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

---

## 📄 License

This project is open source and free to use for educational purposes.
