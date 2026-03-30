# 🏦 Bank Marketing Classification Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-red?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A machine learning classification project that applies and compares **6 different algorithms** on the Bank Marketing dataset to predict whether a client will subscribe to a term deposit — built for CS675: Introduction to Data Science at Pace University.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Hyperparameter Tuning](#-hyperparameter-tuning-extra-credit)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

This project applies and compares **6 Machine Learning classification algorithms** on the UCI Bank Marketing dataset. The goal is to predict whether a bank client will subscribe to a term deposit based on demographic and campaign data.

The notebook covers:
- ✅ **Exploratory Data Analysis** — distributions, correlations, class imbalance
- ✅ **Data Preprocessing** — encoding, scaling, stratified train/test split
- ✅ **6 Classification Models** — default hyperparameters
- ✅ **Full Evaluation** — Accuracy, Precision, Recall, F1, AUC
- ✅ **Confusion Matrices** — heatmap for every model
- ✅ **Combined ROC Curve** — all 6 models on one plot
- ✅ **Hyperparameter Tuning** — LR, RF, XGBoost (extra credit)
- ✅ **Final Comparison** — summary table and bar chart

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

| Property | Details |
|---|---|
| Records | 41,188 |
| Features | 20 input features + 1 target |
| Target | `y` — term deposit subscription (yes/no) |
| Class Balance | 88.73% No / 11.27% Yes |
| Missing Values | None |

### Feature Categories

| Category | Features |
|---|---|
| Client Info | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Campaign Info | `contact`, `month`, `day_of_week`, `duration`, `campaign` |
| Previous Campaign | `pdays`, `previous`, `poutcome` |
| Economic Indicators | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

---

## 📁 Project Structure
```
bank-marketing-classification/
│
├── Krishna_Maniyar_Classification_Project_2.ipynb  # Main notebook
├── README.md                                        # Documentation
└── requirements.txt                                 # Dependencies
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis
- Target variable distribution and class imbalance visualization
- Age distribution by subscription outcome
- Job type vs subscription rate
- Numerical feature boxplots per class
- Correlation heatmap of numerical features

### Step 2 — Data Preprocessing
- Binary encoding of target variable (`no=0`, `yes=1`)
- One-hot encoding of 10 categorical columns (54 total features)
- Stratified 70/30 train/test split (class balance preserved)
- StandardScaler applied to all features

### Step 3 — Model Training
All 6 models trained with default parameters first:
```python
models = [
    LogisticRegression(max_iter=1000),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    XGBClassifier(eval_metric='logloss')
]
```

### Step 4 — Evaluation
Each model evaluated on:
- **Accuracy** — overall correct predictions
- **Precision** — of predicted positives, how many are correct
- **Recall** — of actual positives, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **AUC** — area under the ROC curve
- **Confusion Matrix** — heatmap visualization
- **Classification Report** — per-class breakdown

---

## 📈 Models & Results

### Default Model Performance

| Rank | Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|---|
| 1 | Random Forest | ~0.91 | ~0.68 | ~0.42 | ~0.52 | ~0.93 |
| 2 | XGBoost | ~0.91 | ~0.67 | ~0.44 | ~0.53 | ~0.93 |
| 3 | Logistic Regression | ~0.91 | ~0.66 | ~0.38 | ~0.48 | ~0.93 |
| 4 | Decision Tree | ~0.88 | ~0.50 | ~0.50 | ~0.50 | ~0.72 |
| 5 | K-Nearest Neighbors | ~0.90 | ~0.62 | ~0.38 | ~0.47 | ~0.87 |
| 6 | Naive Bayes | ~0.85 | ~0.45 | ~0.69 | ~0.54 | ~0.90 |

> **Note:** Exact values will be printed when you run the notebook.
> High accuracy scores are partly due to class imbalance (88.73% "no").
> F1 Score and AUC are more reliable metrics for imbalanced datasets.

---

## 🔧 Hyperparameter Tuning (Extra Credit)

Three models were retuned to improve performance on the minority class:

### Tuned Logistic Regression
```python
LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
```

### Tuned Random Forest
```python
RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, class_weight='balanced'
)
```

### Tuned XGBoost
```python
XGBClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, scale_pos_weight=7
)
```

> `class_weight='balanced'` and `scale_pos_weight=7` address the class imbalance directly, improving Recall and F1 on the minority "yes" class.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Clone & Setup
```bash
# Clone the repository
git clone https://github.com/krishnamaniyar2209/bank-marketing-classification.git

# Navigate to the project folder
cd bank-marketing-classification

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Krishna_Maniyar_Classification_Project_2.ipynb
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
xgboost>=1.7.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 🚀 Usage

1. Open the notebook in Jupyter or Google Colab
2. The dataset downloads **automatically** from UCI in Cell 3 — no manual download needed
3. Run all cells sequentially from top to bottom
4. All plots, metrics, confusion matrices, and ROC curves generate automatically

---

## 💡 Key Findings

- **Class imbalance** (88.73% / 11.27%) is the biggest challenge — accuracy alone is misleading
- **Duration** of the call is the strongest individual predictor of subscription
- **Economic indicators** (`euribor3m`, `emp.var.rate`, `nr.employed`) are highly correlated with each other
- **Random Forest** and **XGBoost** achieve the highest AUC (~0.93) among default models
- **Naive Bayes** has the highest Recall despite lower overall accuracy — it catches more "yes" clients
- **Hyperparameter tuning** with `class_weight='balanced'` significantly improves Recall on the minority class at a slight cost to Precision
- **Students** and **retired** clients have the highest subscription rates relative to their group size

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| [Python](https://python.org) | Core language |
| [scikit-learn](https://scikit-learn.org/) | All ML models and metrics |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting classifier |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [Seaborn](https://seaborn.pydata.org/) | Statistical visualization |
| [Jupyter](https://jupyter.org/) | Development environment |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS675: Introduction to Data Science (Fall 2024)
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
  <br><br>
  <img src="https://img.shields.io/badge/Pace%20University-Seidenberg%20School%20of%20CSIS-blue" />
</p>
