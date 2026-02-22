# 🍄 Mushroom Classification using Machine Learning

> An AI-powered Machine Learning system for automatic mushroom classification into **edible** or **poisonous** categories using multiple supervised learning algorithms.
>
> This project demonstrates a complete ML pipeline including preprocessing, encoding strategies, model comparison, feature importance analysis, hyperparameter tuning, and overfitting detection — delivering a highly interpretable and robust classification system.

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-2ea44f?style=flat-square)
![Models](https://img.shields.io/badge/Models-6-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---


## 📖 Overview

The system classifies mushrooms into:

| Label | Class | Description |
|-------|-------|-------------|
| `e` | **Edible** | Safe for consumption |
| `p` | **Poisonous** | Toxic — must be correctly identified |

The dataset consists entirely of **categorical features**, making this project an excellent case study in:

- Categorical data encoding
- Tree-based vs. distance-based models
- Feature importance & interpretability
- Model validation & generalization

> ✅ This project highlights how machine learning can assist in **toxicology risk assessment** and **decision support systems**.

---

## 🎯 Objectives

### 🔹 1. Data Exploration & Preprocessing

- Analyze dataset structure and class balance
- Check missing values & duplicates
- Detect rare categories (< 1%)
- Perform optional PCA-based outlier inspection
- Apply encoding strategies

### 🔹 2. Encoding Strategies

Two encoding techniques were implemented and compared *(see [Encoding Strategies](#-encoding-strategies) below)*.

### 🔹 3. Model Development

Six supervised learning models were implemented and evaluated across five metrics: **Accuracy · Precision · Recall · F1-Score · Confusion Matrix**

### 🔹 4. Hyperparameter Tuning

Applied `GridSearchCV` to optimize Random Forest with 3-fold cross-validation.

### 🔹 5. Model Validation & Overfitting Detection

Multi-layered validation to ensure generalization — 5-fold CV, learning curves, and permutation importance.

---

## 🏷️ Encoding Strategies

### Label Encoding — Tree-Friendly

Efficient for tree-based models. Used with:

- Decision Tree
- Random Forest
- Categorical Naive Bayes

### One-Hot Encoding — Distance-Friendly

Suitable for distance and margin-based models. Scaled using `StandardScaler(with_mean=False)`. Used with:

- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes

---

## 🤖 Models Implemented

| Model | Encoding | Kernel / Config |
|-------|----------|-----------------|
| Decision Tree | Label | — |
| **Random Forest** | Label | **GridSearchCV tuned** |
| K-Nearest Neighbors (KNN) | One-Hot + Scaled | Euclidean, k=5 |
| Support Vector Machine | One-Hot + Scaled | RBF Kernel |
| Categorical Naive Bayes | Label | Categorical likelihood |
| Gaussian Naive Bayes | One-Hot + Scaled | Gaussian likelihood |

**Evaluation metrics applied to every model:**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positives / predicted positives |
| Recall | True positives / actual positives |
| F1-Score | Harmonic mean of precision & recall |
| Confusion Matrix | Per-class prediction breakdown |

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` applied to Random Forest:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}
# cv=3, scoring='accuracy'
```

**Best Parameters Found:**

```python
{'max_depth': None, 'n_estimators': 100}
```

---

## ✅ Model Validation & Overfitting Detection

| Technique | Purpose |
|-----------|---------|
| **5-Fold Cross-Validation** | Confirms stable performance across all data splits |
| **Train vs. Test Accuracy** | Detects gap between training and generalization |
| **Learning Curve Analysis** | Visualizes bias-variance tradeoff over dataset size |
| **Permutation Importance** | Model-agnostic feature relevance validation |

> ✔ Results show **strong generalization with no significant overfitting.**

---

## 🏆 Best Performing Model

### 🌲 Random Forest Classifier

After hyperparameter tuning — `{'max_depth': None, 'n_estimators': 100}`

| Metric | Score |
|--------|-------|
| **Accuracy** | ✅ 100% |
| **Precision** | ✅ 1.00 |
| **Recall** | ✅ 1.00 |
| **F1-Score** | ✅ 1.00 |
| **CV Stability** | ✅ Stable across all folds |

---

## 🔬 Feature Importance & Ablation Study

Feature importance was analyzed using two methods:

- **Random Forest built-in importance** (Mean Decrease in Impurity)
- **Permutation Importance** — model-agnostic, shuffle-based

### 🚨 Key Insight: `odor` is the Single Most Important Feature

> *"Mushroom smell is the dominant predictive factor for toxicity."*

### Ablation Study — Effect of Removing `odor`

| Condition | Accuracy | Δ Change |
|-----------|----------|----------|
| ✅ **With** `odor` | **100%** | — |
| ⚠️ **Without** `odor` | **88.6%** | ↓ −11.4 pp |

Even without `odor`, the model significantly outperforms random guessing (50%) — confirming that other structural features (gill color, spore print, ring type) still carry **meaningful predictive signal**.

---


## 🧠 ML Concepts Demonstrated

- Categorical Encoding — Label vs. One-Hot
- Tree-based vs. Distance-based model comparison
- Hyperparameter tuning with `GridSearchCV`
- Cross-validation (3-fold tuning, 5-fold evaluation)
- Learning curve analysis
- Feature importance (MDI) & Permutation Importance
- Feature ablation analysis
- Overfitting detection & mitigation

---

## 🚀 Project Highlights

- ✅ Complete end-to-end ML pipeline
- ✅ Multiple model comparison across two encoding strategies
- ✅ Perfect classification performance (100% accuracy)
- ✅ Strong interpretability via feature importance & ablation
- ✅ Robust multi-layered validation strategy

---


## 👩‍💻 Author

<div align="center">

### ✨ *Eng. Paula Hanna Naguib* ✨

</div>

---

<div align="center">
  <sub>📌 <em>"Machine Learning transforms raw categorical data into life-saving insights."</em></sub>
</div>
