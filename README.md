# Colon Disease Segmentation

A deep learning project for multiclass semantic segmentation of colon pathology images using medical imaging datasets and U-Net based architectures.

This repository focuses on detecting and segmenting multiple colon disease categories from histopathology images using PyTorch-based workflows.

---

# Project Overview

The goal of this project is to build a robust semantic segmentation pipeline capable of:

* Identifying diseased regions in colon histopathology images
* Performing pixel-wise segmentation
* Handling multiclass medical image segmentation
* Evaluating segmentation performance using medical imaging metrics
* Managing dataset inconsistencies and preprocessing automatically

The project currently uses the EBHI-SEG dataset structure and is designed for experimentation, training, evaluation, and future model improvements.

---

# Supported Classes

The dataset currently contains the following pathology classes:

| Class ID | Disease Class    |
| -------- | ---------------- |
| 0        | Background       |
| 1        | Adenocarcinoma   |
| 2        | High-grade IN    |
| 3        | Low-grade IN     |
| 4        | Normal           |
| 5        | Polyp            |
| 6        | Serrated adenoma |

---

# Repository Structure

```text
colon-disease-segmentation/
│
├── docs/
│   └── basic info.md
│
├── output/
│   └── evaluation_artifacts/
│       ├── classification_confusion_matrix.csv
│       ├── segmentation_confusion_matrix.csv
│       ├── metrics_summary.json
│       └── metrics_summary.txt
│
├── scripts/
│   └── clean_dataset.py
│
├── src/
│   └── Unet_segmenation_Model.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Features

* Multiclass semantic segmentation
* Medical image preprocessing
* Dataset cleaning and validation
* Automatic image-label pair verification
* U-Net based segmentation workflow
* Evaluation metric generation
* Confusion matrix generation
* Ready for Kaggle or local GPU training

---

# Dataset Information

Dataset characteristics identified during preprocessing:

| Property                | Value                            |
| ----------------------- | -------------------------------- |
| Total image files       | 2254                             |
| Total label files       | 2234                             |
| Valid image-label pairs | 2226                             |
| Image resolution        | 224 × 224                        |
| Mask format             | Grayscale                        |
| Task type               | Multiclass semantic segmentation |

The dataset contains class imbalance, with Adenocarcinoma being the largest class and Serrated adenoma being the smallest.

---

# Dataset Cleaning

The repository includes preprocessing utilities to fix dataset inconsistencies.

Current checks include:

* Missing image detection
* Missing label detection
* Filename consistency validation
* Pair filtering
* Mask preprocessing

Example issues detected:

* 28 images without matching labels
* 8 labels without matching images

Only valid image-label pairs should be used during training.

---

# Model Architecture

Current baseline plan:

| Component         | Configuration   |
| ----------------- | --------------- |
| Framework         | PyTorch         |
| Model             | U-Net++         |
| Encoder           | EfficientNet-B3 |
| Segmentation Type | Multiclass      |
| Output Channels   | 7               |

Why this architecture:

* Strong medical segmentation baseline
* Good balance between performance and compute usage
* Suitable for Kaggle GPU environments
* Easy to extend with advanced augmentation and attention mechanisms

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/colon-disease-segmentation.git
cd colon-disease-segmentation
```

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Current Dependencies

```text
numpy
pillow
```

Additional packages that will likely be required for training:

```text
torch
torchvision
segmentation-models-pytorch
opencv-python
matplotlib
scikit-learn
albumentations
pandas
```

---

# Usage

## Run Dataset Cleaning

```bash
python scripts/clean_dataset.py
```

## Open Training Notebook

```bash
jupyter notebook src/Unet_segmenation_Model.ipynb
```

---

# Training Workflow

Recommended workflow:

1. Clean and validate dataset
2. Create train/validation/test split
3. Convert masks to foreground segmentation masks
4. Train baseline U-Net++ model
5. Evaluate Dice score and mIoU
6. Analyze confusion matrices
7. Improve augmentation and architecture

---

# Evaluation Metrics

The project stores evaluation artifacts inside:

```text
output/evaluation_artifacts/
```

Metrics currently tracked:

* Dice Score
* Mean IoU
* Segmentation confusion matrix
* Classification confusion matrix
* JSON metric summaries

---

# Recommended Improvements

Future improvements planned:

* Advanced augmentation pipeline
* Attention U-Net variants
* DeepLabV3+ experiments
* Transformer-based segmentation
* Mixed precision training
* Cross-validation support
* TensorBoard logging
* WandB integration
* Docker support
* Full training scripts
* ONNX export
* Real-time inference pipeline

---

# Known Issues

* Dataset imbalance between classes
* Some filename mismatches exist
* Current requirements file is incomplete for full training
* Notebook naming contains a typo: `Unet_segmenation_Model.ipynb`

---

# Results

Evaluation summaries and confusion matrices are available inside:

```text
output/evaluation_artifacts/
```

These files can be used to:

* Compare model versions
* Track segmentation performance
* Analyze class-wise behavior
* Debug minority class predictions

---

# Contributing

Contributions are welcome.

Possible contribution areas:

* Model optimization
* Better preprocessing
* Additional architectures
* Training automation
* Visualization tools
* Inference APIs
* Documentation improvements

---

# License

Add your preferred license here.

Example:

```text
MIT License
```

---

# Acknowledgements

Dataset inspiration and segmentation workflow are based on colon histopathology semantic segmentation research datasets and medical imaging segmentation methodologies.

Libraries commonly used in this workflow:

* PyTorch
* NumPy
* Pillow
* segmentation-models-pytorch
* OpenCV

---

# Author

Developed for experimentation and research in medical image segmentation and deep learning.

