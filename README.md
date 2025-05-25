# 🔐 Enhanced Attack Detection in Cognitive Radio Networks Using LIBESN

This repository contains the official implementation of the **Leaky Integrated Bidirectional Echo State Network (LIBESN)** model for real-time attack detection in **Cognitive Radio Networks (CRNs)**. This research has been **published in IEEE** and presented at the **2025 IEEE Global Conference in Emerging Technology (GINOTECH)**.

> 📄 IEEE Paper Title: **Enhanced Attack Detection in Cognitive Radio Using Leaky Integrated Bidirectional Echo State Networks**
> 
> 📍 Conference: **IEEE GINOTECH 2025**
> 
> 📚 DOI: *\[To be added after publication]*
---

## 🧠 Project Highlights

* 🌐 **Domain**: Cognitive Radio Security, Deep Learning, Wireless Communication
* ⚙️ **Model**: LIBESN – Combines Echo State Networks with leaky integration and bidirectional processing
* 📈 **Performance**:

  * Accuracy: 94.67%
  * Precision: 93.33%
  * Recall: 95.89%
  * F1 Score: 94.59%
* 🧪 Robust against:

  * Jamming Attacks
  * Primary User Emulation Attacks (PUEA)
  * Spectrum Sensing Data Falsification (SSDF)
* 🖼️ Includes t-SNE visualizations, ROC & PR curves, and confusion matrix for evaluation

---

## 📁 Repository Structure

```
📦 CRN-LIBESN-Attack-Detection/
├── libesn_model.py       # Implementation of LIBESN
├── dataset/              # Preprocessed CRN dataset (.csv)
├── utils/                # Helper functions for normalization, splitting, metrics
├── results/              # Figures - t-SNE, ROC, PR, Confusion Matrix
└── README.md             # You are here
```

---

## 🚀 Getting Started

### 📌 Prerequisites

* Python 3.8+
* `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* `seaborn`, `tsne`, or similar libraries for visualization

```bash
pip install -r requirements.txt
```

### 🔧 Run the Model

```bash
python train_and_evaluate.py
```

---

## 📊 Dataset Description

* **Format**: Time-series CSV with labels
* **Features**: Signal strength, transmission time, energy usage, cluster role, etc.
* **Labels**: Normal, Jamming, PUEA, SSDF, and Cross-layer attacks
* **Preprocessing**: Normalized using Min-Max Scaling; split using stratified sampling

---

## 🧠 About the Model

### LIBESN Components:

* **Leaky Integration**: Maintains historical state with decay for long-term patterns
* **Bidirectional Reservoir**: Learns past and future context
* **Fixed Internal Weights**: Faster training than typical RNNs
* **Dense Output Layer**: Lightweight classification

---

## 📸 Key Visualizations

* t-SNE plots for feature clustering
* ROC-AUC (AUC: 0.979)
* Precision-Recall curves
* Confusion matrix showing low false positives and false negatives

---

## 🎓 Publication & Credits

This work is a **published IEEE research paper** and was **presented at GINOTECH 2025**.

* **Authors**:

  * Dr. S. Palanivel Rajan
  * R. Rahul
  * T. Jegan
  * S. Yasar Arafath

For citation:

```
@inproceedings{rahul2025libesn,
  title={Enhanced Attack Detection in Cognitive Radio Using Leaky Integrated Bidirectional Echo State Networks},
  author={R. Rahul, T. Jegan, S. Yasar Arafath, and Dr. S. Palanivel Rajan},
  booktitle={IEEE Global Conference in Emerging Technology (GINOTECH)},
  year={2025}
}
```

---

## 💬 Contact

For collaboration, reach out:

* 📧 [rahulkanna170504@gmail.com](mailto:rahulkanna170504@gmail.com)
* 📍 Velammal College of Engineering and Technology, Madurai, India
