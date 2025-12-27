
# 🫁 DeepMed: Unveiling Health with Neural Networks

An end-to-end **Deep Learning–based Medical Image Classification System** for detecting **COVID-19, Pneumonia, and Normal cases** from Chest X-ray images.  
This project demonstrates **production-grade ML engineering**, covering **data handling, model training, evaluation, deployment, and documentation**.

---

## 🚀 Project Highlights

- 🧠 **Multi-model Deep Learning pipeline** (VGG16, ResNet18, DenseNet121, GoogLeNet, EfficientNet, MobileNetV3)
- 🏆 **Best Model: ResNet18 (He Initialization + Adam Optimizer)**
- 📊 Achieved **~95% Test Accuracy** with strong **Macro F1-score (~0.94)**
- ⚖️ **Class imbalance handled** using Weighted Sampling
- 📦 **Clean dataset abstraction** (dataset excluded due to size — industry standard)
- 🎥 **Deployment demo included** (real-time image prediction workflow)
- 📄 Detailed **technical report** with analysis and comparisons
- 💼 **Recruiter-ready, production-style project structure**

---

## 🩺 Problem Statement

Chest X-rays are widely used for diagnosing respiratory diseases.  
Manual interpretation is time-consuming and prone to subjectivity.

This project aims to:
- Automatically classify Chest X-ray images into:
  - **COVID-19**
  - **Pneumonia**
  - **Normal**
- Assist clinicians with **fast, reliable, and explainable predictions**

---

## 🧬 Dataset Overview

> ⚠️ **Note:** Dataset is intentionally **not included** in this repository due to size constraints.

### Expected Dataset Structure

```
dataset/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── COVID19/
    ├── NORMAL/
    └── PNEUMONIA/
```

- Total Images: **~6,400+**
- Severe class imbalance addressed using **WeightedRandomSampler**
- Compatible with `torchvision.datasets.ImageFolder`

---

## 🧠 Models Implemented

| Model | Description |
|-----|-------------|
| VGG16 | Transfer Learning Baseline |
| ResNet18 | Residual Learning (Best Model) |
| DenseNet121 | Feature Reuse via Dense Connections |
| GoogLeNet | Inception-based CNN |
| EfficientNet-B0 | Parameter-Efficient Scaling |
| MobileNetV3 | Lightweight Mobile-Oriented Model |

---

## 📊 Test Results Summary (from Notebook Outputs)

The following table summarizes the **final test-set performance** of all evaluated models:

| Model | Test Accuracy (%) | Test Loss | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|------|------------------:|----------:|------------------:|---------------:|-----------------:|
| **ResNet18 + He Init + Adam** | **94.88** | 0.1487 | 0.9399 | 0.9463 | **0.9430** |
| GoogLeNet | 94.57 | 0.1659 | 0.9523 | 0.9241 | 0.9375 |
| EfficientNet-B0 | 94.10 | 0.1585 | 0.9650 | 0.9149 | 0.9366 |
| GoogLeNet (Alt Run) | 93.48 | 0.1693 | 0.9604 | 0.9054 | 0.9286 |
| VGG16 | 92.86 | 0.2093 | 0.9253 | 0.9039 | 0.9129 |
| ResNet18 (Xavier + SGD) | 92.55 | 0.1875 | 0.9104 | 0.9429 | 0.9235 |
| VGG16 + Early Stopping | 92.00 | 0.2076 | 0.9179 | 0.8847 | 0.8979 |
| DenseNet121 | 91.07 | 0.2311 | 0.9010 | 0.8898 | 0.8940 |
| MobileNetV3-Large | 78.49 | 0.5029 | 0.8374 | 0.7046 | 0.6925 |

✔ **Best Overall Performance:** ResNet18 with He Initialization and Adam Optimizer  
✔ Metrics computed using **macro-averaging** to account for class imbalance

---

## 🎥 Deployment & Prediction Workflow (Video Explained)

The included **deployment video (`Deployment.mp4`)** demonstrates the complete inference pipeline:

1. User uploads a Chest X-ray image
2. Image is preprocessed (resize, normalization)
3. Trained ResNet18 model performs inference
4. Model predicts one of:
   - **COVID-19**
   - **Pneumonia**
   - **Normal**
5. Prediction is displayed to the user

🎯 If a COVID-19 image is uploaded → **COVID-19 predicted**  
🎯 If a Normal image is uploaded → **Normal predicted**  
🎯 If a Pneumonia image is uploaded → **Pneumonia predicted**  

This confirms correct **generalization and deployment readiness**.

---

## 🛠️ Technologies Used

- Python
- PyTorch & Torchvision
- NumPy, Pandas
- Matplotlib, Seaborn
- Transfer Learning
- CNN Architectures
- Class Imbalance Handling
- Model Evaluation Metrics
- Deployment Concepts

---

## 📁 Repository Structure

```
.
├── Code.ipynb            # Full training & evaluation pipeline
├── Model Weights.pt      # Best ResNet18 model weights
├── Report.pdf            # Detailed technical report
├── Deployment.mp4        # Deployment & inference demo
└── README.md             # Project documentation
```

---

## ▶️ How to Run

1. Clone the repository
2. Place dataset in the expected directory structure
3. Open `Code.ipynb`
4. Run cells sequentially
5. Load provided model weights for inference

---

## 👥 Team Contributions

This project was **equally designed, implemented, and tested** by both:

- **@VenkataSriSaiSuryaMandava**  
  👉 https://github.com/VenkataSriSaiSuryaMandava

- **@DharmavaramRachana**  
  👉 https://github.com/DharmavaramRachana

Both contributors were involved in:
- Dataset preprocessing
- Model design & training
- Evaluation & analysis
- Deployment workflow
- Documentation & reporting

---

## 📌 Why This Project Matters

✔ End-to-end ML system (data → model → deployment)  
✔ Strong performance on medical imaging task  
✔ Production-aware design choices  
✔ Ideal for **ML Engineer / Data Scientist / Healthcare AI roles**  

---

⭐ If you find this project useful, consider starring the repository!
