# Credit Card Fraud Detection System 🛡️

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions. 

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents a significant class imbalance challenge, where frauds account for only **0.172%** of all transactions.

**Key Features of this Solution:**
*   **Robust Data Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle extreme class imbalance.
*   **Targeted Feature Engineering:** Adds domain-specific features to capture spending patterns and anomalies.
*   **Strict Feature Selection:** Uses Random Forest importance to select the top ~24 most predictive features, reducing noise.
*   **Ensemble Learning:** Combines **Random Forest** and **XGBoost** via a Soft Voting Classifier to maximize detection capability.
*   **Production-Ready Structure:** Organized code into modular scripts for data processing, training, evaluation, and inference.

---

## 📂 Project Structure
The project is organized efficiently for reproducibility and scalability:

```
credit-card-fraud-ml/
├── data/
│   ├── raw/                  # Original input file (creditcard.csv)
│   ├── processed/            # Generated files after feature selection & splitting
│   │   ├── X_train.csv       # SMOTE-balanced Training Features
│   │   ├── y_train.csv       # SMOTE-balanced Training Target
│   │   ├── X_test.csv        # Unseen Test Features
│   │   └── y_test.csv        # Unseen Test Target
│
├── src/
│   ├── feature_selection.py      # 1️⃣ Data Pipeline: Loading, Engineering, SMOTE, Saving
│   ├── train_random_forest.py    # 2️⃣ Model Training: Random Forest
│   ├── train_xgboost.py          # 2️⃣ Model Training: XGBoost
│   ├── train_ensemble.py         # 2️⃣ Model Training: Voting Ensemble
│   ├── test_pipeline.py          # 3️⃣ Inference: Loads models & predicts on test set
│   └── generate_visualizations.py# 4️⃣ Analysis: Generates Confusion Matrices & PR Curves
│
├── models/                   # 💾 Saved Models (.pkl files)
├── metrics/                  # 📊 Text reports containing F1, Precision, Recall scores
├── plots/                    # 📈 Generated Charts (CM, ROC, PR Curves)
├── requirements.txt          # 📦 Python Dependencies
└── README.md                 # 📖 Project Documentation
```

---

## �️ Methodology & Technical Details

### 1. Feature Engineering
Since the original dataset consists mostly of PCA-transformed features (`V1`...`V28`), we focused detailed engineering on the non-transformed `Amount` and `Time` columns:

| Feature Name | Description | Logic / Formula |
| :--- | :--- | :--- |
| **`log_amount`** | Normalizes the highly skewed `Amount` distribution. | `log(Amount + 1)` |
| **`amount_zscore`** | Standardizes amount to detect outliers based on standard deviation. | `(Amount - Mean) / StdDev` |
| **`is_high_amount`** | Binary flag for high-value transactions. | `1` if Amount > 95th Percentile, else `0` |
| **`amount_per_time`** | Captures rate/velocity of spending. | `Amount / (Time + 1)` |

### 2. Feature Selection
We trained a preliminary Random Forest to rank feature importance. To improve model generalization, we kept only the **Top 24 Features**, including our new engineered features and the most relevant PCA components (e.g., V17, V14, V12), discarding noise.

### 3. Handling Imbalance (SMOTE)
*   **Why?** The dataset has 99.8% non-fraud cases. Standard models would just predict "Legit" for everything and achieve 99.8% accuracy but miss every fraud.
*   **Strategy:** We applied **SMOTE** (Synthetic Minority Over-sampling Technique) to create synthetic examples of fraud.
*   **Crucial Detail:** SMOTE was applied **ONLY to the Training Set**. The Test Set remains practically imbalanced to ensure our evaluation metrics reflect real-world performance.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Run this script first. It performs feature engineering, selection, splitting, and SMOTE balancing.
```bash
python src/feature_selection.py
```
*Output: Saves processed CSVs to `data/processed/`.*

### 3. Train Models
Train the models. Each script saves the trained model to `models/` and a metrics report to `metrics/`.

```bash
# Train Random Forest
python src/train_random_forest.py

# Train XGBoost
python src/train_xgboost.py

# Train Ensemble (Best of both worlds)
python src/train_ensemble.py
```

### 4. Evaluate & Visualize
Generate Confusion Matrices and Precision-Recall Curves to visually assess performance.
```bash
python src/generate_visualizations.py
```
*Output: Images saved to `plots/`.*

### 5. Running Inference (Testing)
To see the models in action on the test set (simulating a production environment):
```bash
python src/test_pipeline.py
```

---

## 📊 Model Performance

We optimized for **Recall** (catching as many frauds as possible) while maintaining decent **Precision** (minimizing false alarms).

| Model | Recall | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **85.71%** | **79.25%** | **0.82** | **0.982** |
| **XGBoost** | 85.71% | 42.25% | 0.57 | 0.976 |
| **Ensemble** | 85.71% | 72.41% | 0.79 | 0.983 |

**Conclusion:**
*   **Random Forest** is the best performing single model for this dataset configuration.
*   All models achieved excellent Recall (~86%), meaning they detect the vast majority of fraud attempts.
*   The **Ensemble model** provides a robust alternative, offering slightly higher ROC-AUC stability.

---

## 📦 Dependencies
*   `pandas`, `numpy`: Data Manipulation
*   `scikit-learn`: Modeling & Metrics
*   `xgboost`: Gradient Boosting Model
*   `imbalanced-learn`: SMOTE Implementation
*   `joblib`: Model Persistence
*   `matplotlib`, `seaborn`: Visualization
