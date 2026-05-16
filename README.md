# Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Telco-blue)

> **Can we predict which customers will leave before they do?**  
> This end-to-end machine learning project identifies telecom customers at risk of churning using the IBM Telco dataset, enabling proactive retention strategies that save real revenue.

---

## Problem Statement

Customer churn costs the telecom industry billions annually. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a binary classification pipeline to flag at-risk customers **before** they churn, giving the business a window to intervene.

---

## Project Structure

```
 Telco-Churn-Prediction
 ┣ Churn_prediction_analysis.ipynb   # EDA & Data Cleaning
 ┣ Churn_prediction_model.ipynb      # ML Modeling & Evaluation
 ┣ df_churn.csv                      # Processed feature dataset
 ┣ RFC_model.sav                     # Serialized best model (Random Forest + SMOTEENN)
 ┗ README.md
```

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Records | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` — Yes / No (binary) |
| Class Imbalance | ~26% churners vs ~74% non-churners |

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Fixed data type mismatch: `TotalCharges` stored as string → coerced to numeric
- Handled 11 missing values in `TotalCharges` (new customers with 0 tenure) → filled with `0`
- Engineered a `tenuregroup` feature by discretizing continuous tenure into 6 meaningful cohorts (1–12, 13–24 … 61–72 months)
- One-hot encoded all categorical features via `pd.get_dummies`

**Key Insights from EDA:**

| Finding | Business Implication |
|---|---|
| Month-to-month contract customers churn ~3× more | Offer incentives to switch to annual plans |
| Fiber optic internet users show higher churn | Investigate pricing and service quality |
| Electronic check payment users churn most | Consider autopay incentives |
| No Online Security / Tech Support → higher churn | Bundle these services for at-risk customers |
| Gender & phone service lines → near-zero correlation with churn | Drop from feature set |
| Customers in first 12 months are highest risk | Trigger early retention outreach at month 3 |

---

### 2. Class Imbalance Handling
The dataset is imbalanced (~26% churn). A naive model would simply predict "No Churn" most of the time. To fix this, **SMOTEENN** was applied — a hybrid technique that:
- **SMOTE**: Oversamples minority class (churners) using synthetic interpolation
- **ENN (Edited Nearest Neighbours)**: Cleans noisy borderline samples

This ensures the model learns to *genuinely* identify churners rather than exploiting class frequency.

---

### 3. Models Trained

| Model | Variant | Notes |
|---|---|---|
| Decision Tree | Baseline | `gini`, `max_depth=6`, `min_samples_leaf=8` |
| Decision Tree | + SMOTEENN | Resampled training data |
| Random Forest | Baseline | 100 estimators, same hyperparams |
| **Random Forest** | **+ SMOTEENN** | **Best model — saved to disk** |

---

### 4. Evaluation
Models were evaluated on held-out test data (80/20 split) using:
- **Classification Report** — Precision, Recall, F1-Score per class
- **Confusion Matrix** — True/False Positives & Negatives
- Primary metric: **Recall on churn class** — in a churn problem, a missed churner (False Negative) is more costly than a false alarm

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTEENN) |
| Model Persistence | Pickle |

---

## Getting Started

### Clone the repository
```bash
git clone https://github.com/<your-username>/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
```

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run the notebooks in order
```
1. Churn_prediction_analysis.ipynb   ← Start here (EDA & feature engineering)
2. Churn_prediction_model.ipynb      ← Then run this (model training & evaluation)
```

> **Note:** The analysis notebook exports `df_churn.csv`, which is the input for the model notebook. Run them in sequence.

---

## Results Summary

The **Random Forest Classifier with SMOTEENN** resampling was selected as the final model because it delivered the best recall on the minority (churn) class — the metric that matters most in a retention use case.

The model is serialized as `RFC_model.sav` and can be loaded for inference:

```python
import pickle

model = pickle.load(open('RFC_model.sav', 'rb'))
predictions = model.predict(X_new)
```

---

## Business Impact

If deployed in production, this model enables:
- **Proactive retention campaigns** targeting predicted churners before they leave
- **Resource prioritization** — focus customer success teams on highest-risk accounts
- **Revenue protection** — even a 5% improvement in churn retention can significantly move ARPU

---

## Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` or `Optuna`
- [ ] Try `XGBoost` / `LightGBM` for potentially higher performance
- [ ] Add ROC-AUC curve and Precision-Recall curve for richer evaluation
- [ ] Build a `Streamlit` web app for real-time churn scoring
- [ ] Add SHAP values for model explainability / feature importance visualization

---

## Author

**Yusra Imran Vekriwala**  
[LinkedIn](https://linkedin.com/in/yusra-vekriwala/) · [GitHub](https://github.com/yusravekriwala) · [Email](mailto:yusravekriwala@email.com)

---

## License

This project is open source and available under the [MIT License](LICENSE).
