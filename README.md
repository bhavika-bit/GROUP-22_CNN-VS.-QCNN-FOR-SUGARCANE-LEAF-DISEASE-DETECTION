# 🌿 Classical vs Quantum CNNs for Sugarcane Leaf Disease Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![PennyLane](https://img.shields.io/badge/PennyLane-QCNN-grey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen?style=flat-square)

**A controlled comparative study of Classical CNNs and Quantum Convolutional Neural Networks (QCNNs) for automated sugarcane leaf disease detection.**

*Group 22 · TY AIDS · K.J. Somaiya School of Engineering · Guide: Dr. Suchitra Patil*

</div>

---

## 📌 Overview

Sugarcane is a critical cash crop in Maharashtra and a major contributor to India's sugar economy. Early and accurate detection of leaf diseases — Red Rot, Mosaic, Yellow Leaf, Rust, Smut, and more — is essential for minimising yield loss.

This project implements and compares **three classical deep learning architectures** (Custom CNN, EfficientNet-B0, MobileNetV2) for sugarcane leaf disease classification, with **Quantum CNN (QCNN)** integration planned for Semester 7. The key contribution is a **controlled, apples-to-apples comparison** — identical datasets, identical preprocessing pipelines, and identical evaluation metrics — which is largely absent from existing literature.

---

## 📁 Repository Structure

```
sugarcane-cnn-qcnn/
│
├── dataset/                        # Dataset folder (total 13 classes)
│   ├── Healthy/
│   ├── Mosaic/
│   ├── RedRot/
│   ├── Rust/
│   └── Yellow/
    ...
│
├── models/                         # Saved model checkpoints
│   ├── custom_cnn.pth
│   ├── efficientnet_b0.pth
│   └── mobilenetv2_best.pth
│
├── training/                       # Training scripts
│   ├── training_CustomCNN.py       # Custom CNN — trained from scratch
│   ├── training_EfficientNetB0.py  # EfficientNet-B0 — transfer learning
│   └── training_MobileNetV2.py     # MobileNetV2 — two-phase fine-tuning
│
├── results/                        # Output plots and reports
│   ├── mobilenetv2_training_report.png
│   └── ...
│
├── qcnn/                           # 🔬 Semester 7 — QCNN (coming soon)
│   ├── quantum_circuit.py
│   └── hybrid_model.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Implemented

### 1. Custom CNN (Trained from Scratch)
- 4-block Conv → BatchNorm → ReLU → MaxPool architecture
- Fully connected head: 512 units + 50% Dropout
- Input: 224×224 RGB images
- **Chosen as QCNN integration target** for Semester 7 due to modular architecture

### 2. EfficientNet-B0 (Transfer Learning)
- ImageNet pretrained backbone — feature layers frozen
- Last 2 feature blocks unfrozen in Phase 2 for domain fine-tuning
- Custom classifier head sized to number of disease classes
- Highest test accuracy: **96.79%**

### 3. MobileNetV2 (Two-Phase Transfer Learning)
- Phase 1: Head-only training (20 epochs, LR = 1e-3)
- Phase 2: Last 3 InvertedResidual blocks unfrozen (30 epochs, LR = 1e-5)
- Early stopping (patience = 7), ReduceLROnPlateau scheduler
- Aggressive augmentation pipeline including RandomErasing
- Fastest training: **43.95 minutes**

---

## 📊 Results Summary

### Training Performance

| Model         | Train Acc | Val Acc | Train Loss | Val Loss | Training Time | Model Fit |
|---------------|-----------|---------|------------|----------|---------------|-----------|
| Custom CNN    | 94.90%    | 93.21%  | 60.42      | 33.27    | 2.20 hrs      | Good Fit  |
| EfficientNet-B0 | 93.69%  | 92.80%  | 90.45      | 25.03    | 2.34 hrs      | Good Fit  |
| MobileNetV2   | 91.96%    | 91.07%  | 23.13      | 25.77    | 43.95 min     | Good Fit  |

> Note: Custom CNN and EfficientNet report **cumulative epoch loss** (sum of per-batch losses). MobileNetV2 reports **average per-batch loss**. Normalised, all three converge to comparable magnitudes.

### Test Performance (2,521 images · 5 classes)

| Model           | Test Accuracy | Precision | Recall | F1-Score | Inference Time |
|-----------------|---------------|-----------|--------|----------|----------------|
| Custom CNN      | 84.73%        | 0.85      | 0.85   | 0.85     | 5.14 sec       |
| EfficientNet-B0 | **96.79%**    | **0.97**  | **0.97** | **0.97** | 7.34 sec   |
| MobileNetV2     | 82.43%        | 0.82      | 0.82   | 0.82     | 5.77 sec       |

**Disease classes tested:** Healthy · Mosaic · Red Rot · Rust · Yellow Leaf

---

## 🗂️ Dataset

| Category           | Public Datasets              | Web Scraped                          |
|--------------------|------------------------------|--------------------------------------|
| **Sources**        | Mendeley, Kaggle             | Zenodo, Bing Images, Google Images   |
| **Location**       | Maharashtra, India           | Maharashtra, India                   |
| **Timeline**       | 2022–2023                    | Jan 2026 – Mar 2026                  |
| **Size**           | ~16,000+ images              | ~1,500–2,000 curated images          |
| **Classes**        | Yellow Leaf, Smut, Mosaic, Rust | Red Rot, Smut, Mosaic, Yellow Leaf, Grassy Shoot, Rust, Wilt, Leaf Scald, Pokkah Boeng, Healthy |
| **Resolution**     | ~768×1024 px                 | Variable (>800×600 px)               |
| **Final Input**    | 256×256 px (all models)      | 256×256 px                           |

**Public dataset links:**
- [Kaggle — Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)
- [Mendeley Dataset 1](https://data.mendeley.com/datasets/355y629ynj)
- [Mendeley Dataset 2](https://data.mendeley.com/datasets/9twjtv92vk)

> The `dataset/` folder is **not tracked** in this repository due to size. Download the public datasets from the links above and organise them into per-class subdirectories under `dataset/`.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (CPU training supported but slow)

### Install dependencies

```bash
git clone https://github.com/<bhavika-bit>/sugarcane-cnn-qcnn.git
cd sugarcane-cnn-qcnn
pip install -r requirements.txt
```

### `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
numpy>=1.24.0
pillow>=9.5.0
# Semester 7 — QCNN
pennylane>=0.35.0
qiskit>=1.0.0
```

---

## 🚀 Training

### Custom CNN
```bash
python training/training_CustomCNN.py
```

### EfficientNet-B0
```bash
python training/training_EfficientNetB0.py
```

### MobileNetV2 (two-phase with early stopping)
```bash
python training/training_MobileNetV2.py
```

All scripts auto-detect classes from the `dataset/` folder. Checkpoints and training plots are saved automatically.

---

## 🔬 Technical Approach

```
                     ┌─────────────────────────────────────┐
                     │        Data Preprocessing           │
                     │  Resize · Normalize · Augment       │
                     └────────────┬────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼──────────┐               ┌───────────▼───────────┐
   │  Classical Engine   │               │   Quantum Engine      │
   │  (CNN)              │               │   (QCNN — Sem 7)      │
   │                     │               │                       │
   │  Feature Extraction │               │  CNN Feature Extract  │
   │  Feature Fusion     │               │  PCA Reduction        │
   │  Dropout Reg.       │               │  Quantum Encoding     │
   │  Disease Classify   │               │  VQC (PennyLane)      │
   └──────────┬──────────┘               │  Disease Classify     │
              │                          └───────────┬───────────┘
              └───────────────────┬───────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  Performance Comparison  │
                     │  Accuracy · F1 · Speed  │
                     └─────────────────────────┘
```

**Tech Stack:**
`pandas` · `NumPy` · `OpenCV` · `Albumentations` · `scikit-learn` · `PyTorch` · `Qiskit` · `PennyLane`

---

## 🗓️ Roadmap

### ✅ Semester 6 (Complete)
- [x] Dataset collection and preprocessing pipeline
- [x] Custom CNN — trained from scratch
- [x] EfficientNet-B0 — transfer learning
- [x] MobileNetV2 — two-phase fine-tuning
- [x] Comparative evaluation on 5-class test set

### 🔬 Semester 7 (Aug – Dec 2026)
- [ ] Learn Qiskit & PennyLane fundamentals (Aug)
- [ ] Implement quantum convolutional layers with VQCs (Sep)
- [ ] Integrate QCNN with Custom CNN backbone (Sep)
- [ ] Train and optimize hybrid model (Oct)
- [ ] Comparative benchmarking: CNN vs QCNN (Oct–Nov)
- [ ] Journal paper submission + final presentation (Nov–Dec)

---

## 📚 Key References

1. Sugarcane Leaf Disease Classification using Deep Neural Network — *BMC Plant Biology, 2025*
2. QSVM vs SVM & QCNN vs CNN for Plant Disease Detection — *ISJEM, 2025*
3. Hybrid Quantum-Classical Model for Plant Disease Detection — *IJCPDM, 2025*
4. A Tutorial on Quantum CNNs — *arXiv:2009.09423*
5. Quantum vs Classical CNN Benchmarking in Binary Classification — *IEEE, 2025*
6. Hybrid Classical-to-Quantum Transfer Learning for Image Classification

Full reference list available in the project report.

---

## 👥 Team

| Name | Roll No. | Role |
|------|----------|------|
| Bhavika Jata | 16014233029 | Technical Approach · Model Architecture · Dataset |
| Dhairya Gorasiya | 16014223032 | Problem Statement · Hyperparameters · Results |
| Smit Chavan | 16014223080 | Literature Review · Requirements · Conclusion |

**Project Guide:** Dr. Suchitra Patil  
**Institution:** K.J. Somaiya School of Engineering, Somaiya Vidyavihar University

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<i>Built with 🍃 for Maharashtra's sugarcane farmers</i>
</div>
