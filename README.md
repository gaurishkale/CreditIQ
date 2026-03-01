
#  CreditIQ
### AI-Driven Loan Underwriting & Risk Scoring System.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![AUC](https://img.shields.io/badge/ROC--AUC-0.9587-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What is CreditIQ?

CreditIQ is an end-to-end AI-powered loan underwriting system that predicts 
the probability of a borrower defaulting on a loan — in seconds.

Trained on **1.3 million real Lending Club loans** from 2007–2018, the model 
analyzes 114 financial features to generate a **risk score (300–850)**, 
a **risk grade (A–F)**, and an **approval decision** — just like a real bank.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| 🏆 ROC-AUC | **0.9587** |
| ✅ Accuracy | **91%** |
| 🎯 Default Recall | **90%** |
| 📈 F1-Score (weighted) | **0.91** |

> A ROC-AUC of 0.9587 means the model correctly ranks a bad loan above 
> a good loan **95.87% of the time** — competitive with real banking models.

---

## 🖥️ App Screenshot

> *(Add your screenshot here after running the app)*

---

## 🏗️ Project Structure
```
CreditIQ/
│
├── app.py                                         # Streamlit web application
├── booster_only.txt                               # Trained LightGBM model
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project documentation
├── .gitignore                                     # Files excluded from GitHub
│
└── notebook/
    └── AI_Driven_Loan_Underwriting_IMPROVED.ipynb # Full training notebook
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-underwriting-ai.git
cd loan-underwriting-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📋 How It Works

### Step 1 — Enter Loan Details
Fill in the applicant's financial information:
- Loan amount, interest rate, term, purpose
- Annual income, employment length, home ownership
- FICO score, DTI ratio, credit utilization, delinquency history

### Step 2 — AI Processes 114 Features
The model analyzes raw inputs plus engineered features:
- `loan_to_income` — borrowing relative to earnings
- `installment_to_income` — monthly payment affordability
- `credit_util_ratio` — how maxed out are their credit cards
- `fico_avg` — average FICO score
- `delinq_to_open_acc` — late payment rate across accounts

### Step 3 — Get Instant Decision
```
Risk Score  : 742
Risk Grade  : B — Good
Default Prob: 8.2%
Decision    : ✅ APPROVED
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **LightGBM** | Gradient boosting model |
| **Pandas & NumPy** | Data processing |
| **Streamlit** | Web application |
| **SHAP** | Model explainability |
| **Scikit-learn** | Evaluation metrics |
| **Google Colab** | Model training (GPU) |

---

## 📈 ML Pipeline
```
Raw Data (2.2M Lending Club loans)
          ↓
Filter conclusive loans only (1.3M)
          ↓
Remove data leakage columns
          ↓
Handle missing values
          ↓
Feature Engineering (14 new financial ratios)
          ↓
LightGBM training with early stopping
          ↓
Evaluation (AUC = 0.9587)
          ↓
Streamlit app → Risk Score + Decision
```

---

## 📊 Dataset

- **Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Original size:** 2.2M loans, 151 columns
- **After filtering:** 1.3M loans, 114 features
- **Target variable:** `loan_status` → Fully Paid (0) vs Charged Off (1)
- **Class ratio:** ~80% good loans, ~20% defaults

> Dataset not included in this repo due to size (1.5GB).  
> Download from Kaggle and place as `accepted_2007_to_2018Q4.csv` to retrain.

---

## 🧠 Key Concepts Implemented

- ✅ Data leakage prevention
- ✅ Class imbalance handling (`scale_pos_weight`)
- ✅ Feature engineering (financial ratios)
- ✅ Ordinal encoding (grade, sub_grade, emp_length)
- ✅ Early stopping (prevents overfitting)
- ✅ SHAP explainability
- ✅ Risk score calibration (300–850 scale)

---

## 💡 Risk Score Guide

| Score | Grade | Meaning | Decision |
|-------|-------|---------|----------|
| 750–850 | A — Excellent | Very low default risk | ✅ Approve |
| 700–749 | B — Good | Low default risk | ✅ Approve |
| 650–699 | C — Fair | Moderate risk | ✅ Approve |
| 600–649 | D — Poor | Elevated risk | ⚠️ Review |
| 550–599 | E — Very Poor | High risk | ⚠️ Review |
| 300–549 | F — High Risk | Very high risk | ❌ Reject |

---

## 🙋 About

Built by Gaurish as part of an AI-driven fintech project exploring 
machine learning applications in credit risk assessment.

- 💼 [LinkedIn](https://www.linkedin.com/in/gaurishkale16)
- 🐙 [GitHub](https://github.com/gaurishkale)

---

## 📄 License

This project is licensed under the MIT License.  
Dataset is owned by Lending Club / Kaggle — not included in this repo.

---

⭐ If you found this project useful, please give it a star!
