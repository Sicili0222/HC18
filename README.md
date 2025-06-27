# HC18

# Fetal Head Circumference Segmentation with ResUNet-MHBR and Quad-HC Loss

This repository contains the implementation of a deep learning-based framework for automatic fetal head circumference (HC) segmentation from ultrasound images. The proposed model combines a ResUNet backbone with Multi-branch Hybrid Attention (MBHA) and Boundary Refinement Block (BRB) modules, trained using a novel Quadruple Composite Loss (Quad-HC Loss) to enhance segmentation accuracy and robustness.

## 📁 File Structure

- `train.py`  
  Main script for training the ResUNet-MHBR model with the proposed Quad-HC loss function.

- `test.py`  
  Script for evaluating the trained model on the test dataset.

- `ResUNet_MHBR.py`  
  Implementation of the ResUNet-MHBR architecture, incorporating MBHA and BRB modules.

- `Quad-HC Loss.py`  
  Definition of the Quadruple Composite Loss function, combining four loss components for robust learning.

- `Data_Processing.py`  
  Data preprocessing and augmentation functions for loading and preparing ultrasound images and segmentation masks.

- `package.py`  
  Utility functions or packaging-related code (e.g., metrics, visualization, or model saving).

- `README.md`  
  Project overview and usage instructions.

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- OpenCV
- tqdm
- albumentations

## 🚀 Usage

### Train the Model

```bash
python train.py
```

### Test the Model

```bash
python test.py
```

Data paths and training settings can be configured directly in the `train.py` and `test.py` files.

## 📊 Loss Function – Quad-HC

The Quadruple Composite Loss combines:

* **Dice Loss**
* **Binary Cross-Entropy (BCE) Loss**
* **Ellipse Shape Constraint Loss**
* **Curvature Loss**

This design improves the model’s ability to capture smooth and anatomically realistic head boundaries.


