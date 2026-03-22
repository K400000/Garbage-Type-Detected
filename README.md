# 🗑️ Garbage Image Classification

## Project Overview

This project builds **AI models for garbage image classification**, classifying waste images into 4 main categories using 3 different machine learning models: **KNN**, **SVM**, and **CNN**.

The goal is to compare model performance using **Accuracy**, **Precision**, and **Recall** metrics and determine 
which model works best for this task.

## Dataset
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data

---

## Category Mapping

The dataset contains **10 sub-categories** of waste, mapped into **4 main groups**:

| Sub-category | Main Group |
|---|---|
| Battery | **Hazardous** |
| Biological | **Organic** |
| Cardboard, Glass, Metal, Paper, Plastic | **Recyclable** |
| Clothes, Shoes, Trash | **Others** |

---

## Dataset Structure

```
DetectGarbage/
├── original/              # Raw image dataset
│   ├── battery/
│   ├── biological/
│   ├── cardboard/
│   ├── clothes/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   └── trash/
├── data_preparation.ipynb # Data preparation notebook
├── prepared_data.npz      # Saved preprocessed data (generated after running notebook)
└── README.md
```

---

## Data Preparation Pipeline

The `data_preparation.ipynb` notebook performs the following steps:

1. **Load images** from `original/` using OpenCV
2. **Convert** BGR → RGB color space
3. **Resize** all images to **128×128** pixels
4. **Normalize** pixel values from `[0, 255]` → `[0, 1]` (float32)
5. **Encode labels** using `LabelEncoder` (text → integer)
6. **Split** data into **80% Train / 20% Test** (stratified)
7. **Prepare** model-specific formats (flattened for KNN/SVM, 3D for CNN)
8. **Save** all processed data to `prepared_data.npz`

---

## Prepared Data Variables

After running the notebook, the following variables are available (also saved in `prepared_data.npz`):

### Image Data

| Variable | Shape | Description |
|---|---|---|
| `X_train` | `(n_train, 128, 128, 3)` | Training images — normalized RGB, for **CNN** |
| `X_test` | `(n_test, 128, 128, 3)` | Test images — normalized RGB, for **CNN** |
| `X_train_flat` | `(n_train, 49152)` | Flattened training images (128×128×3), for **KNN & SVM** |
| `X_test_flat` | `(n_test, 49152)` | Flattened test images (128×128×3), for **KNN & SVM** |

- Pixel values: `float32` in range `[0.0, 1.0]`
- 49,152 features = 128 × 128 × 3 (width × height × channels)

### Labels

| Variable | Shape | Description |
|---|---|---|
| `y_train` | `(n_train,)` | Encoded training labels (integers) |
| `y_test` | `(n_test,)` | Encoded test labels (integers) |
| `classes` | `(4,)` | Class name array: `['Hazardous', 'Organic', 'Others', 'Recyclable']` |

### Label Encoding

| Class Name | Encoded Value |
|---|---|
| Hazardous | 0 |
| Organic | 1 |
| Others | 2 |
| Recyclable | 3 |

---

## Loading Prepared Data

```python
import numpy as np

data = np.load('prepared_data.npz')
X_train      = data['X_train']       # For CNN
X_test       = data['X_test']        # For CNN
X_train_flat = data['X_train_flat']  # For KNN & SVM
X_test_flat  = data['X_test_flat']   # For KNN & SVM
y_train      = data['y_train']       # Training labels
y_test       = data['y_test']        # Test labels
classes      = data['classes']       # Class names
```

---

## Models (Next Step)

| Model | Type | Input Format |
|---|---|---|
| **KNN** | K-Nearest Neighbors | `X_train_flat` / `X_test_flat` |
| **SVM** | Support Vector Machine | `X_train_flat` / `X_test_flat` |
| **CNN** | Convolutional Neural Network | `X_train` / `X_test` |

---

## Tech Stack

- **Python 3**
- **OpenCV** — Image loading and preprocessing
- **NumPy** — Array operations
- **Pandas** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **scikit-learn** — KNN, SVM, LabelEncoder, train_test_split, metrics
- **TensorFlow / Keras** — CNN model (upcoming)


## 👥 Team Members
* **Mr. Supasin Khamphayae** - [GitHub Profile](https://github.com/K400000)
* **Mr. Piyakorn Kacharnont** - [GitHub Profile](https://github.com/Shiseko)
* **Mr. Jakkapat Srimek** - [GitHub Profile](https://github.com/jakkapat1513)
* **Mr. Kritamet Manakit** - [GitHub Profile](https://github.com/kritametm-gif)
* **Mr. Kittichai Kuljaruhiran** - [GitHub Profile](https://github.com/kitti1223)
* **Mr. Mawin Boonsri** - [GitHub Profile](https://github.com/Mawinbosri)