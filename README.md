# 💳 Credit Risk Analysis
> End-to-end loan default prediction & FICO score bucketing — inspired by the **JPMorgan Chase Quantitative Research** virtual experience.

---

## 📊 Dataset
| Field | Detail |
|---|---|
| Rows | 10,000 loans |
| Features | `fico_score`, `income`, `loan_amt_outstanding`, `total_debt_outstanding`, `credit_lines_outstanding`, `years_employed` |
| Target | `default` (binary) |
| Default Rate | **18.51%** |

---

## ⚙️ Pipeline

**1. 🔍 EDA** — missing value check, class distribution, feature correlations

**2. 🤖 Modelling** — trained & compared:
- Logistic Regression
- Decision Tree
- Random Forest ✅ *(best)*
- Gradient Boosting
- XGBoost *(optional)*

**3. 📏 Evaluation** — ROC-AUC, Precision-Recall, Brier Score, Calibration curves, Permutation Importance

**4. 💰 Portfolio Loss** — Expected Loss = `PD × LGD × EAD` with `LGD = 90%`

**5. 🪣 FICO Bucketing** *(Task 4)* — Dynamic programming quantization into **5 rating bands** using:
- **MSE minimisation**
- **Log-likelihood maximisation**

---

## 🪣 FICO Rating Bands (Log-Likelihood)

| Rating | FICO Range | Default Rate |
|--------|-----------|-------------|
| 1 ⚠️ | 408 – 521 | 65.19% |
| 2 🔴 | 521 – 580 | 38.00% |
| 3 🟠 | 580 – 640 | 20.45% |
| 4 🟡 | 640 – 696 | 10.51% |
| 5 🟢 | 696 – 850 | 4.65% |

---

## 🛠️ Tech Stack
`Python 3.12` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `matplotlib` · `seaborn`

---

## 🚀 Quickstart
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook Credit_risk_analysis.ipynb
```

> 📁 Update the CSV path in cell 2 to point to your local `Loan_Data.csv`
