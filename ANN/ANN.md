# 🧠 Bank Customer Churn Prediction — Artificial Neural Network

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Sequential_API-D00000?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This project builds an **Artificial Neural Network (ANN)** from scratch using **TensorFlow/Keras** to predict whether a bank customer will **churn (leave the bank)** based on their profile and account information.

The model is trained on a dataset of **10,000 bank customers** with features like credit score, geography, age, balance, and activity status. The goal is to classify each customer as either staying (`0`) or leaving (`1`).

---

## 📂 Dataset

**File:** `Churn_Modelling.csv`

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Spain, Germany) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card? (1/0) |
| `IsActiveMember` | Active member? (1/0) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target** — Did the customer leave? (1 = Yes, 0 = No) |

> 📊 **10,000 rows** | **14 columns** | Binary Classification Task

---

## 🔧 Workflow

### Part 1 — Data Preprocessing
- Imported the dataset using **Pandas**
- Applied **Label Encoding** on the `Gender` column (Male/Female → 0/1)
- Applied **One-Hot Encoding** on the `Geography` column (France, Spain, Germany → dummy variables)
- Split the data into **Training (80%)** and **Test (20%)** sets using `train_test_split`
- Applied **Feature Scaling** using `StandardScaler` (fit on train, transform both)

### Part 2 — Building the ANN
- Initialized a **Sequential** model with `tf.keras`
- Added **2 hidden layers**, each with 6 neurons and `ReLU` activation
- Added **1 output layer** with 1 neuron and `Sigmoid` activation (binary output)

```
Input Layer  →  Hidden Layer 1 (6, ReLU)  →  Hidden Layer 2 (6, ReLU)  →  Output Layer (1, Sigmoid)
```

### Part 3 — Training the ANN
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metric:** Accuracy
- **Batch Size:** 32
- **Epochs:** 100

### Part 4 — Evaluation & Prediction
- Predicted churn probability for individual customers
- Evaluated on the full test set using a **Confusion Matrix** and **Accuracy Score**

---

## 🧪 Single Customer Prediction — Example

The model was used to predict churn for the following customer profile:

| Feature | Value |
|---|---|
| Geography | France |
| Credit Score | 600 |
| Gender | Male |
| Age | 40 |
| Tenure | 3 years |
| Balance | $60,000 |
| Num of Products | 2 |
| Has Credit Card | Yes |
| Is Active Member | Yes |
| Estimated Salary | $50,000 |

> ✅ **Prediction: Customer stays in the bank.**

---

## 🏗️ Model Architecture

```python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))   # Hidden Layer 1
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))   # Hidden Layer 2
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Output Layer

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Array operations |
| `pandas` | Data loading and manipulation |
| `tensorflow` / `keras` | Building and training the ANN |
| `scikit-learn` | Encoding, splitting, scaling, evaluation |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ann-churn-prediction.git
   cd ann-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas tensorflow scikit-learn
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook artificial_neural_network.ipynb
   ```

> Make sure `Churn_Modelling.csv` is in the same directory as the notebook.

---

## 📁 Repository Structure

```
ann-churn-prediction/
│
├── artificial_neural_network.ipynb   # Main notebook
├── Churn_Modelling.csv               # Dataset
└── README.md                         # Project documentation
```

---

## 🙋 Author

**Samsur**
Aspiring ML Engineer | Data Science & Deep Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/your-username)

---

> ⭐ If you found this project helpful, feel free to star the repository!
