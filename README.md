# California Housing Price Predictor — Deep Learning

> Wide & Deep neural network with multi-input/multi-output architecture for predicting California district median house values using TensorFlow/Keras.

---

## Overview

This project implements a **Wide & Deep regression network** using the Keras Functional API to predict California district median house values. The architecture routes different feature subsets through separate wide (linear) and deep (nonlinear) paths, then merges them before producing a primary prediction. An auxiliary output on the deep path acts as a regularizer during training.


**Key capabilities:**
- Wide & Deep architecture via the Keras Functional API
- Dual-input split: wide path (features 0–4) and deep path (features 2–7)
- Auxiliary output with weighted loss for improved gradient flow
- ModelCheckpoint callback to persist the best epoch automatically

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Jupyter Notebook or JupyterLab
