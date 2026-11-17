# 🌿 Comparative Analysis of Classical and Quantum CNNs for Sugarcane Leaf Disease Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Classical%20CNNs-EE4C2C?style=flat-square&logo=pytorch)
![Quantum ML](https://img.shields.io/badge/Quantum%20ML-QSVC%20%7C%20VQC-6A5ACD?style=flat-square)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20Circuits-grey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Evaluation-brightgreen?style=flat-square)

**A comparative study of classical deep learning and quantum machine learning methods for sugarcane leaf disease classification.**

*Group 22 · LY AI & DS · K.J. Somaiya School of Engineering · Guide: Dr. Suchitra Patil*

</div>

---

## 📌 Overview

Accurate identification of sugarcane leaf diseases from images can support timely crop monitoring and disease management. This project investigates classical and quantum machine learning approaches for sugarcane leaf disease classification, with a focus on evaluating them through a common experimental framework.

The work includes:

- Classical image classification using a **Custom CNN** and **EfficientNet-B0**.
- Evaluation on a 13-class test set and a separate four-class cross-dataset evaluation.
- Quantum-kernel classification experiments using **Quantum Support Vector Classifiers (QSVC)** with PCA, LDA and NCA feature reduction.
- A **Variational Quantum Classifier (VQC)** using a 12-dimensional NCA feature representation.
- A latent-space comparison between VQC-NCA12 and raw EfficientNet-B0 features.

The experiments investigate the effect of feature dimensionality reduction and quantum model design. The reported results do not establish that quantum models outperform classical models; rather, they show the feasibility and limitations of the evaluated quantum approaches under the tested configurations.

---

## 🎯 Problem Definition

CNNs are widely used for plant disease image classification. Quantum machine learning methods, including quantum kernels and variational circuits, offer alternative approaches to learning decision boundaries from compact feature representations.

However, there is limited comparative work focused specifically on sugarcane leaf diseases. Comparisons are also difficult when models use different datasets, preprocessing pipelines or evaluation metrics.

This project studies classical and quantum approaches for sugarcane leaf disease classification and examines whether quantum methods provide measurable advantages over classical baselines in the evaluated settings.

---

## 🧪 Dataset

The project dataset combines publicly available and curated web-scraped sugarcane leaf images.

| Attribute | Public datasets | Web-scraped datasets |
|---|---|---|
| Sources | Mendeley, Kaggle | Zenodo, Bing Images, Google Images, gardening sites |
| Collection | In-field and post-harvest surveys | Python-based web collection |
| Geographic context | Maharashtra, India | Maharashtra, India |
| Timeline | 2022–2023 | January–March 2026 |
| Approximate size | 16,000+ images | 1,500–2,000 curated images |
| Disease labels | Yellow Leaf, Smut, Mosaic, Rust (source datasets vary) | Red Rot, Smut, Mosaic, Yellow Leaf, Grassy Shoot, Rust, Wilt, Leaf Scald, Pokkah Boeng and Healthy Leaf |
| Image format | RGB | RGB |
| Approximate source resolution | 768 × 1024 pixels | Variable; greater than 800 × 600 pixels |
| Final input resolution | 256 × 256 pixels | 256 × 256 pixels |
| Metadata | Disease labels | 12-attribute metadata CSV |

The project report describes a dataset spanning **13 classes**. The primary classical CNN test evaluation reports 976 test samples across 13 classes. The cross-dataset evaluation uses 2,052 samples across four shared classes:

- Healthy
- Mosaic
- Red Rot
- Yellow Leaf

Dataset links:

- [Kaggle — Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)
- [Mendeley Dataset 1](https://data.mendeley.com/datasets/355y629ynj)
- [Mendeley Dataset 2](https://data.mendeley.com/datasets/9twjtv92vk)

> The complete dataset is not reproduced in this README. Check the repository's dataset instructions and source terms before downloading or redistributing data.

---

## 🧠 Classical Models

### 1. Custom CNN

A CNN architecture trained from scratch and used as the custom classical baseline.

| Parameter | Configuration |
|---|---|
| Batch size | 16 |
| Epochs | 40 |
| Learning rate | 5 × 10⁻⁵ (fixed) |
| Optimizer | Adam |
| Learning-rate scheduler | CosineAnnealingLR, Tmax = 40, ηmin = 1 × 10⁻⁶ |
| Loss | CrossEntropyLoss, without label smoothing |
| Train / validation / test split | 80 / 10 / 10 |
| Input size | 224 × 224 |
| Augmentation | Horizontal flip, rotation up to 20°, brightness and contrast |
| Training approach | From scratch |
| Dropout | 0.5 |

### 2. EfficientNet-B0

An ImageNet-pretrained EfficientNet-B0 model evaluated using transfer learning and fine-tuning.

| Parameter | Configuration |
|---|---|
| Batch size | 32 |
| Epochs | 100 |
| Classifier-head learning rate | 1 × 10⁻³ (fixed) |
| Backbone learning rate | 5 × 10⁻⁵ with decay |
| Optimizer | Adam |
| Learning-rate schedule | Cosine decay on backbone learning rate |
| Weight decay | 1 × 10⁻⁴ |
| Loss | CrossEntropyLoss, label smoothing = 0.1 |
| Train / validation / test split | 80 / 10 / 10 |
| Input size | 224 × 224 |
| Augmentation | Same as Custom CNN, with normalization |
| Training approach | Transfer learning; backbone fine-tuning |
| Dropout | 0.2, with BatchNorm1d before final layer |
| Pretrained weights | ImageNet |

---

## 📊 Classical CNN Results

### Training and validation performance

| Metric | Custom CNN | EfficientNet-B0 |
|---|---:|---:|
| Training accuracy | 90.97% | 98.20% |
| Validation accuracy | 91.81% | 95.84% |
| Training loss | 0.28 | 0.66 |
| Validation loss | 0.27 | 0.69 |
| Training time | 2 hours | 2 hours |
| Model fit (as reported) | Good Fit | Good Fit |

### Primary test-set evaluation

<text>Reported evaluation: 976 test samples and 13 classes.</text>

| Metric | Custom CNN | EfficientNet-B0 |
|---|---:|---:|
| Test accuracy | 90.50% | **95.08%** |
| Precision | 0.91 | **0.95** |
| Recall | 0.90 | **0.95** |
| F1-score | 0.90 | **0.95** |
| Testing runtime | 2 mins | 2 mins |

### Cross-dataset generalization

<text>Reported evaluation: 2,052 test samples across four shared classes — Healthy, Mosaic, Red Rot and Yellow Leaf.</text>

| Metric | Custom CNN | EfficientNet-B0 |
|---|---:|---:|
| Test accuracy | 89.63% | **96.56%** |
| Precision | 0.90 | **0.97** |
| Recall | 0.90 | **0.97** |
| F1-score | 0.90 | **0.97** |

**Observation:** EfficientNet-B0 achieved higher reported test accuracy and F1-score than the Custom CNN in both the primary test evaluation and the four-class cross-dataset evaluation.

---

## ⚛️ Quantum Machine Learning Experiments

The quantum experiments investigate classification using reduced-dimensional feature representations. The report evaluates quantum-kernel methods (QSVC) against classical SVM baselines and also evaluates a variational quantum classifier (VQC).

### 1. QSVC with PCA and LDA

| Metric | PCA QSVC | PCA Classical SVM | LDA-12 QSVC | LDA-12 Classical SVM |
|---|---:|---:|---:|---:|
| Feature dimension | 256 | 256 | 12 | 12 |
| Training accuracy | 69.29% | 70.56% | 99.44% | 99.01% |
| Validation accuracy | 68.96% | 70.56% | 96.32% | 96.18% |
| Test accuracy | 66.30% | 67.47% | 94.11% | 94.45% |
| Test precision | 0.72 | 0.73 | 0.94 | 0.94 |
| Test recall | 0.66 | 0.67 | 0.94 | 0.94 |
| Test F1-score | 0.65 | 0.67 | 0.94 | 0.94 |
| Test MCC | 0.6382 | 0.6504 | 0.9336 | 0.9374 |
| Train–validation gap | 0.33 pts | 0.01 pts | 3.12 pts | 2.83 pts |
| Best reported hyperparameters | bw = 0.1, C = 10.0 | C = 10.0 | bw = 0.08, C = 20.0 | C = 10.0 |

**Observation:** The LDA-12 configurations performed substantially better than the PCA-256 configurations. In this experiment, the classical SVM slightly exceeded QSVC on test accuracy and F1-score for both feature-reduction settings.

### 2. QSVC with NCA and QKA

| Metric | NCA-12 QSVC (fixed kernel) | NCA-12 Classical SVM | NCA-12 QSVC + QKA (aligned) |
|---|---:|---:|---:|
| Feature dimension | 12 | 12 | 12 |
| Training accuracy | 99.57% | 99.57% | 99.93% |
| Validation accuracy | 96.39% | 96.11% | 96.04% |
| Test accuracy | 94.38% | **94.52%** | **94.52%** |
| Test precision | 0.94 | 0.95 | 0.95 |
| Test recall | 0.94 | 0.95 | 0.95 |
| Test F1-score | 0.94 | 0.94 | 0.94 |
| Test MCC | 0.9367 | **0.9383** | 0.9382 |
| Train–validation gap | 3.18 pts | 3.46 pts | 3.89 pts |
| Training time | 42 mins | 15 mins | 42 mins |
| Best reported hyperparameters | bw = 0.08, C = 1.0 | C = 5.0 | bw = 0.08 (fixed), C = 20.0, θ trained |

**Observation:** NCA-12 produced stronger results than the PCA-256 QSVC configuration. The NCA-12 QSVC and QKA-aligned QSVC reached 94.52% test accuracy, matching the classical SVM's reported accuracy. The classical SVM had the highest reported MCC in this comparison (0.9383).

### 3. Variational Quantum Classifier (VQC) with NCA-12

The VQC experiment uses a reduced 12-dimensional feature representation.

| Metric | VQC-NCA12 |
|---|---:|
| Feature dimension | 12 |
| Training accuracy | 99.16% |
| Validation accuracy | 96.04% |
| Test accuracy | **94.32%** |
| Test precision | 0.95 |
| Test recall | 0.95 |
| Test F1-score | 0.95 |
| Test MCC | 0.9383 |
| Train–validation gap | 3.32 pts |
| Training time | 1.29 hours |

**Reported configuration:**

- 4 quantum circuit layers
- 12 qubits
- 793 trainable parameters
- Data re-uploading enabled
- Ensemble of seeds 7 and 123

**Observation:** VQC-NCA12 achieved 94.52% test accuracy and a reported F1-score of 0.95. These results demonstrate the feasibility of the evaluated variational quantum classifier on the reduced feature representation. They do not, by themselves, demonstrate an advantage over the classical models.

---

## 🔍 Latent-Space Comparison

The report compares the 12-dimensional VQC-NCA12 representation with the 1,280-dimensional raw EfficientNet-B0 representation.

| Metric | VQC-NCA12 | EfficientNet-B0 (raw) |
|---|---:|---:|
| Feature dimension | 12 | 1,280 |
| Fisher ratio (train) | 4.872 | 0.203 |
| Fisher ratio (test) | 3.481 | 0.218 |
| Silhouette score | 0.362 | 0.038 |
| kNN accuracy | 94.315% | 94.178% |
| Logistic regression accuracy | 94.384% | 94.726% |
| RBF-SVM accuracy | 94.726% | 95.479% |
| RBF-SVM validation accuracy | 96.181% | 96.806% |
| Effective dimension | 8.025 | 84.338 |
| Redundancy | 0.059 | 0.082 |
| Minimum margin | 0.664 | 0.206 |

**Interpretation:** The VQC-NCA12 representation has higher reported Fisher ratios and silhouette score, while the raw EfficientNet-B0 representation performs slightly better for logistic regression and RBF-SVM accuracy in this comparison. The representations therefore show different characteristics; no single metric establishes overall superiority.

---

## 🧩 Technical Workflow

The experiments follow two broad paths:

1. **Classical image classification:** preprocess sugarcane leaf images and train classical CNN architectures for disease classification.
2. **Quantum / hybrid classification:** obtain compact feature representations, apply quantum-kernel or variational-circuit methods, and compare their results with classical baselines.

```text
                  Sugarcane Leaf Image Dataset
                              |
                    Preprocessing Pipeline
                 Resize · Normalize · Augment
                              |
                +-------------+-------------+
                |                           |
          Classical CNN                Feature Extraction
                |                           |
       +--------+--------+           Dimensionality Reduction
       |                 |             PCA · LDA · NCA
    Custom CNN      EfficientNet-B0            |
       |                 |              +-----+------+
       +--------+--------+              |            |
                |                      QSVC          VQC
         Disease Prediction        Quantum Kernel   Variational
                |                  Classification   Quantum Circuit
                |                      |            |
                +----------------------+------------+
                                       |
                              Performance Evaluation
                         Accuracy · Precision · Recall
                              F1 · MCC · Runtime
```

### Technology stack

- **Programming:** Python
- **Classical deep learning:** PyTorch
- **Numerical computing:** NumPy
- **Machine learning and evaluation:** scikit-learn
- **Quantum machine learning:** QSVC and VQC
- **Quantum framework:** PennyLane (as applicable to the implemented quantum workflow)

---

## 📈 Key Findings

1. **EfficientNet-B0 was the strongest classical CNN in the reported evaluations**, reaching 95.08% test accuracy on the 13-class primary test set and 96.56% on the four-class cross-dataset evaluation.
2. **Feature reduction materially affected quantum-kernel results.** The reported LDA-12 and NCA-12 QSVC configurations performed substantially better than PCA-256 QSVC.
3. **QSVC did not consistently outperform its classical SVM baselines.** The classical SVM matched or slightly exceeded QSVC on several reported test metrics.
4. **VQC-NCA12 reached 94.52% test accuracy and 0.95 F1-score** in the reported experiment.
5. **Latent-space metrics showed different strengths across representations.** VQC-NCA12 had higher Fisher ratios and silhouette score, while raw EfficientNet-B0 features achieved slightly higher logistic-regression and RBF-SVM accuracy in the reported comparison.

---

## ⚠️ Limitations and Scope

- Results are specific to the datasets, splits, preprocessing and configurations described in the project evaluation.
- The cross-dataset evaluation covers four shared classes, not all 13 classes.
- Quantum experiments use reduced-dimensional feature representations; their results should not be interpreted as a direct comparison of end-to-end quantum image models against full classical CNNs.
- QSVC and VQC are quantum machine learning approaches, but they are not the same architecture as a dedicated Quantum Convolutional Neural Network (QCNN).
- The evaluation report does not provide evidence of a completed dedicated QCNN implementation. QCNN development remains future work.
- Quantum simulator results do not establish performance on noisy, large-scale quantum hardware.
- Further repeated-seed experiments, external validation and computational-cost analysis would strengthen the comparison.

---

## 🔭 Future Scope

- Develop and evaluate a dedicated QCNN architecture for sugarcane leaf disease classification.
- Investigate hybrid quantum–classical architectures and quantum convolutional feature learning.
- Explore improved feature representations, quantum feature maps and kernel selection.
- Conduct systematic hyperparameter optimization and repeated-seed evaluation.
- Study data expressibility, circuit depth, qubit requirements and training stability.
- Evaluate quantum approaches on additional external datasets and, where feasible, quantum hardware.
- Explore improvements in classification accuracy, computational efficiency and generalization.

---

## 📚 References

The following references are listed in the project evaluation report:

1. https://arxiv.org/abs/2406.02229
2. https://isjem.com/download/quantum-enhanced-plant-disease-detection-a-comparative-study-of-qsvm-vs-svm-and-qcnn-vs-cnn/
3. https://www.computersciencejournals.com/ijcpdm/archives/2025.v6.i2.B.127
4. https://arxiv.org/abs/2009.09423
5. https://ieeexplore.ieee.org/document/9574030
6. https://www.hrpub.org/journals/article_info.php?aid=12929
7. https://qtanalytics.in/journals/index.php/IJERR/article/view/3658
8. https://www.ijsrst.com/index.php/home/article/view/IJSRST2512352
9. https://www.atlantis-press.com/proceedings/icarecs-25/126012858
10. https://link.springer.com/article/10.1186/s12870-025-06289-0
11. https://ijcrt.org/papers/IJCRT25A6065.pdf
12. https://www.computersciencejournals.com/ijcpdm/archives/2025/vol6issue2/PartB/6-2-9-272.pdf

### Dataset references

- [Kaggle — Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)
- [Mendeley Dataset 1](https://data.mendeley.com/datasets/355y629ynj)
- [Mendeley Dataset 2](https://data.mendeley.com/datasets/9twjtv92vk)

---

## 👥 Team

| Name | Roll No. |
|---|---|
| Bhavika Jata | 16014233029 |
| Dhairya Gorasiya | 16014223032 |
| Smit Chavan | 16014223080 |

**Project Guide:** Dr. Suchitra Patil  
**Institution:** K.J. Somaiya School of Engineering, Somaiya Vidyavihar University

---

<div align="center">

*Research on classical and quantum approaches for sugarcane leaf disease classification.*

</div>
