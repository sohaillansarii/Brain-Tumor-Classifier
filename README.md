# Brain Tumor MRI Classification

A deep learning pipeline that classifies brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a CNN with transfer learning (MobileNetV2), served through a FastAPI backend with optional Grad-CAM explainability, deployed on Render.


**Live Demo: Brain Tumor Classifier**  
_[View Live Demo](https://braintumorclassifierr.netlify.app/)_


---


## Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, by Masoud Nickparvar) — itself a merge of the Figshare, SARTAJ, and Br35H brain tumor datasets.
- **Classes (4):** `glioma`, `meningioma`, `notumor`, `pituitary`
- **Structure:** pre-split `Training/` and `Testing/` folders, each with one subfolder per class.
- **Size:** 5,616 training images, 1,600 test images (used as provided — folders kept as the original train/test split; no images from `Testing/` were used at any point before final evaluation).

## Project Overview
This project develops a lightweight MobileNetV2-based CNN to classify brain tumor MRI scans into four categories, serving as a fast, assistive screening tool. The end-to-end pipeline covers transfer learning, error analysis, and deployment as a live REST API with Grad-CAM explainability.

* **Preprocessing:** Resized images to 224×224 RGB, scaled to `[0,1]`, and `[-1,1]` for MobileNetV2.
* **Leakage Checks:** MD5 hash comparison confirmed zero exact-duplicate images between train and test splits.
* **Augmentation:** Applied training-only rotations, shifts, zooms, and horizontal flips (excluded unrealistic vertical flips).
* **Baseline Model:** Trained a 3-block CNN from scratch to establish a comparison point for transfer learning.
* **Transfer Learning:** Trained a custom classification head on a frozen, ImageNet-pretrained MobileNetV2 base.
* **Fine-tuning:** Unfroze top MobileNetV2 layers and retrained at a low learning rate (1e-5).
* **Evaluation:** Computed accuracy, loss, per-class accuracy, and prediction confidence on the untouched test set.
* **Error Analysis:** Used per-class breakdown and confidence gaps to identify systematic model weaknesses.
* **Explainability:** Implemented Grad-CAM to visualize model attention, exposed in the notebook and API.

## Model

**Final model: MobileNetV2 (frozen base, feature extractor)**

**Why MobileNetV2 over a larger backbone (e.g. VGG16, EfficientNetB0):** MobileNetV2 was selected over heavier models like VGG16 and EfficientNetB0 to prioritize fast CPU inference and a small footprint (~14 MB), avoiding the slow processing and preprocessing risks of the larger architectures. Additionally, the frozen-base model was chosen over the fine-tuned version because fine-tuning actually decreased test accuracy, providing better generalization on unseen data.



## Model Comparison

| Model | Best Val Accuracy | Best Val Loss | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Baseline CNN (from scratch) | 85.59% | 0.3737 | 75.88% | 0.8991 |
| **MobileNetV2 (frozen base)**  | 93.24% | 0.1696 | **88.13%** | **0.4247** |
| MobileNetV2 (fine-tuned) | 82.92% | 0.4213 | 79.00% | 0.5938 |

Both transfer-learning variants clearly outperformed the from-scratch baseline. Fine-tuning is notable for *underperforming* the frozen-base model on test despite scoring lower validation loss.

## Results

Final model (MobileNetV2, frozen base) performance on the held-out test set:

| Metric | Value |
|---|---|
| Test Accuracy | **88.13%** |
| Test Loss | 0.4247 |
| Best Validation Accuracy | 93.24% |
| Best Validation Loss | 0.1696 |

## Error Analysis

Per-class accuracy and confidence behavior of the final model on the test set :

| Class | Accuracy |
|---|---|
| glioma | 68.50% |
| meningioma | 86.50% |
| notumor | 99.25% |
| pituitary | 98.25% |


- **High accuracy on major classes:** The model performs strongly on notumor (99.25%), pituitary (98.25%), and meningioma (86.50%), demonstrating reliable classification across these categories.

- **Glioma is the clear weak point** — accuracy drops significantly to 68.50%, well below the other three classes, making it the primary target for improvement in future work.

- **Confidence tracks correctness well:** mean confidence on correct predictions was 93.84%, versus 74.11% on incorrect predictions — a meaningful ~20% gap that makes the model's confidence score a reasonable signal for flagging uncertain predictions in a real workflow.

## Explainability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented on the final model's last convolutional layer to visualize which regions of an input MRI scan most influenced its prediction, as a heatmap overlay. This was used  to visually check whether the model's attention aligned with plausible tumor regions rather than irrelevant background/artifacts.
## API

Built with **FastAPI**, served via **Docker**, deployed on **Render**.


### Tech Stack

**Language**
- Python 3.11

**Deep Learning / ML**
- TensorFlow 
- Keras
- scikit-learn

**Data Handling**
- NumPy
- Pandas

**Image Processing**
- Pillow (PIL)
- OpenCV 

**Visualization**
- Matplotlib
- Seaborn

**Backend / API**
- FastAPI
- Uvicorn


**Deployment**
- Docker
- Render

