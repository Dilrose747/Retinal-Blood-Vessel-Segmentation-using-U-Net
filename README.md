# Retinal Blood Vessel Segmentation using U-Net

This project performs **automatic segmentation of retinal blood vessels** from fundus images using a **U-Net** deep learning model.  
The goal is to assist in early detection of diseases such as **Diabetic Retinopathy, Glaucoma, and Hypertension**, where blood vessel structure plays a key diagnostic role.

---

## 🚀 Project Highlights

- **Model:** Original U-Net (encoder-decoder with skip connections)
- **Framework:** PyTorch
- **Dataset Used:** DRIVE (Digital Retinal Images for Vessel Extraction)
- **Task:** Binary Semantic Segmentation (vessel vs non-vessel)
- **Deployment:** Hugging Face Spaces (Interactive Web App)


---

## 🧠 Dataset Overview

**Dataset:** DRIVE  
- **Training Images:** 40 (Split → 30 train, 10 validation)  
- **Image Type:** Retinal Fundus  
- **Ground Truth:** Manual expert segmentation masks  

---

## 🔧 Preprocessing & Augmentation

| Step | Description |
|-----|-------------|
| Resizing | All images + masks resized to **512×512** |
| Normalization | Pixel values scaled to model-friendly range |
| Augmentation | Applied only to training data |

Augmentation performed:
- Horizontal & Vertical flips
- Random rotation (±45°)
- Random cropping (400×400, 50% probability)

**Final Training Samples:**  
30 original → **150 augmented total**  
Validation set remained unaugmented (10 images)

---

## 🏗 Model Architecture

**U-Net** (original architecture):

- **Encoder:** Convolution + ReLU + MaxPool (feature extraction)
- **Decoder:** Transposed Conv + Skip Connections (spatial reconstruction)
- **Output:** Sigmoid activation → binary mask

---

## ⚙️ Training Configuration

| Component | Setting |
|---------|---------|
| Loss Function | **Dice + Binary Cross Entropy** (handles class imbalance) |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| LR Scheduler | ReduceLROnPlateau (patience=5) |
| Epochs | 100 (with early stopping) |
| Batch Size | 2 |
| Logging | Weights & Biases (wandb) |

---

## 📈 Evaluation Metrics

### Performance Across Different Thresholds
| Threshold | Precision | Recall | Accuracy | Dice (F1) |
|----------|-----------|--------|----------|------------|
| **0.4** | 0.6946 | 0.8321 | 0.9559 | 0.7571 |
| **0.5** | 0.7073 | 0.8211 | 0.9571 | 0.7600 |
| **0.6** | 0.7197 | 0.8098 | 0.9582 | 0.7621 |
| **0.7** | 0.7336 | 0.7970 | 0.9593 | 0.7640 |

Dice Score was the primary performance indicator due to class imbalance.

### Interpretation
- Precision ↑ as threshold ↑ → fewer false positives.
- Recall ↓ as threshold ↑ → higher chance of missing smaller vessels.
- Dice score is best around **0.6–0.7**, balancing both.
- Overall accuracy remains stable **>95%**.

---

## 🌍 Deployment

The final model is deployed **on Hugging Face Spaces** with a simple web interface to upload and segment fundus images.

**Live Demo:**  
`[https://huggingface.co/spaces/dilrose/fundus-segmentation]`

---

## Author

**Mohamed Dilrose P M**  
M.Sc. Statistics | Data Scientist
[mohameddilrose2018@gmail.com]  
[https://www.linkedin.com/in/mohamed-dilrose-365554230/]  

---



