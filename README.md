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


## 🗂 Dataset

This work is based on the **HC18 Challenge Dataset**, originally provided for the MICCAI 2018 Grand Challenge: Automatic Fetal Biometry from Ultrasound Images.

* 📎 **Official Website:**
 https://hc18.grand-challenge.org/

* 📥 **Access Instructions:**
  To obtain the dataset, please register on the challenge website and request access. The dataset is for **research use only**.

### 📁 Directory Structure (after preprocessing)

Once downloaded and preprocessed, the dataset should be organized as follows:

```
data/
├── train/
│   ├── images/              # Training ultrasound images (e.g., 123_HC.png)
│   └── labels/              # Corresponding binary masks (e.g., 123_HC_Annotation.png)
├── test/
│   └── images/              # Test ultrasound images (no labels)
```

* All images and masks are in `.png` format, and masks are binary with values `[0, 255]`.
* You can modify `Data_Processing.py` to adapt to your own path or mask format.


## 📊 Loss Function – Quad-HC

The Quadruple Composite Loss combines:

* **Dice Loss**
* **Binary Cross-Entropy (BCE) Loss**
* **Ellipse Shape Constraint Loss**
* **Curvature Loss**

This design improves the model’s ability to capture smooth and anatomically realistic head boundaries.


## 📎 Citation
If you use this codebase in your work, please cite the following related manuscript (currently under submission to The Visual Computer)

