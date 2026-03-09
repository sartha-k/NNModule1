# Neural Network with nn.Module — PyTorch

A hands-on implementation of a binary classification neural network built from scratch using PyTorch's `nn.Module` API.

---

## What This Notebook Covers

- Building a custom model class by inheriting from `nn.Module`
- Defining layers using `nn.Linear` and `nn.Sigmoid`
- Understanding weight and bias shapes
- Running a forward pass
- Inspecting model parameters
- Visualizing model architecture using `torchinfo`

---

## Model Architecture

```
Input (batch_size, num_features)
        ↓
Linear Layer → (batch_size, 1)
        ↓
Sigmoid Activation → (batch_size, 1)
        ↓
Output: probability between 0 and 1
```

| Layer   | Input Shape            | Output Shape  | Parameters        |
|---------|------------------------|---------------|-------------------|
| Linear  | (batch_size, features) | (batch_size, 1) | weight: (1, features), bias: (1,) |
| Sigmoid | (batch_size, 1)        | (batch_size, 1) | None              |

---

## Key Concepts

**`nn.Module`**
All PyTorch models must inherit from `nn.Module`. It enables PyTorch to automatically track parameters, manage gradients, and handle device transfers.

**`__init__`**
Defines the layers of the network. Called once when the model is created.

**`forward()`**
Defines how data flows through the network during a forward pass.

**`super().__init__()`**
Initializes the parent `nn.Module` class — always required.

---

## How to Run

```python
import torch

# Create dummy data
feature = torch.rand(10, 5)   # 10 samples, 5 features

# Create model
model = model(feature.shape[1])

# Forward pass
output = model(feature)       # shape: (10, 1)

# Inspect weights
print(model.linear.weight)    # shape: (1, 5)
print(model.linear.bias)      # shape: (1,)
```

---

## Requirements

```bash
pip install torch torchinfo
```

---

## Part of

This notebook is part of my deep learning self-study series while building toward a career in AI/ML engineering.

| Notebook | Topic |
|---|---|
| `nnModule1.ipynb` | Binary classification with `nn.Module` |
| More coming soon... | CNNs, Datasets, DataLoaders |

---

## Author

**Sarthak** 
Self-studying deep learning with PyTorch
