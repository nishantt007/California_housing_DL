# California Housing Price Predictor using Deep Learning

> Wide & Deep neural network with multi-input/multi-output architecture for predicting California district median house values using TensorFlow/Keras.

---

## Overview

This project implements a **Wide & Deep regression network** using the Keras Functional API to predict California district median house values. The architecture routes different feature subsets through separate wide (linear) and deep (nonlinear) paths, then merges them before producing a primary prediction. An auxiliary output on the deep path acts as a regularizer during training.


**Key capabilities:**
- Wide & Deep architecture via the Keras Functional API
- Dual-input split: wide path (features 0–4) and deep path (features 2–7)
- Auxiliary output with weighted loss for improved gradient flow
- ModelCheckpoint callback to persist the best epoch automatically
- EarlyStopping callback with best-weight restoration to prevent overfitting

---

## Tech Stack

| Category | Tool / Library |
|----------|----------------|
| Language | Python 3.x |
| Deep learning framework | TensorFlow 2.x, Keras (bundled) |
| Data preprocessing | scikit-learn |
| Notebook environment | Jupyter Notebook |

---

> **Note:** The trained model is saved to `my_keras_model.h5` in the working directory at runtime.

---

**Loss configuration:**

| Output | Loss | Weight |
|--------|------|--------|
| `output` (primary) | MSE | 0.9 |
| `aux_output` (auxiliary) | MSE | 0.1 |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Jupyter Notebook or JupyterLab

### Dependency Installation

```bash
pip install tensorflow scikit-learn notebook
```

Or with conda:

```bash
conda install -c conda-forge tensorflow scikit-learn notebook
```

### Environment Setup (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

pip install tensorflow scikit-learn notebook
```
