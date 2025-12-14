# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model using transactional and behavioral data.

---

## Project Overview

**Business Need:**
Bati Bank is partnering with an eCommerce platform to offer Buy-Now-Pay-Later (BNPL) services. The goal is to build a credit scoring model to evaluate customers' creditworthiness and assign risk scores.

**Objective:**
- Define high-risk and low-risk customers using a proxy variable.
- Engineer predictive features from transactions.
- Train a model to assign risk probabilities and derive a credit score.
- Predict optimal loan amount and duration.

**Skills Learned:**
- scikit-learn, Feature Engineering, ML Model Building
- MLOps, MLflow, CI/CD, Unit Testing
- REST API Deployment with FastAPI

---

## Folder Structure

```
credit-risk-model/
├── .github/workflows/ci.yml         # CI/CD workflow
├── data/                            # Data folder (.gitignore)
│   ├── raw/                         # Raw data
│   └── processed/                   # Processed data for training
├── notebooks/
│   └── eda.ipynb                     # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering
│   ├── train.py                      # Model training
│   ├── predict.py                    # Inference
│   └── api/
│       ├── main.py                   # FastAPI app
│       └── pydantic_models.py        # API schemas
├── tests/
│   └── test_data_processing.py       # Unit tests
├── Dockerfile                        # Container setup
├── docker-compose.yml                 # Compose file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore sensitive files
└── README.md                          # This file
```

---

## Project Contents

### Task 1: Credit Scoring Business Understanding
- Understand credit risk fundamentals and Basel II.
- Explain why proxy variable is necessary.
- Discuss trade-offs between simple vs. complex models.



## Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for an Interpretable Model

The Basel II Capital Accord establishes a regulatory framework that emphasizes **accurate risk measurement, model validation, and governance**, particularly for credit risk. Under the Internal Ratings-Based (IRB) approach, banks are permitted to use internal models, but these models must be transparent, well-documented, and auditable.

An interpretable model (e.g., Logistic Regression or a scorecard built using Weight of Evidence) is essential because:

- **Regulatory Compliance**: Regulators must be able to understand and validate how risk estimates are produced and ensure they are non-discriminatory.
- **Auditability**: Internal and external auditors must be able to trace decisions from inputs to final risk scores.
- **Explainability**: Clear explanations are required for internal governance and for communicating adverse credit decisions to customers.

Additionally, Basel II requires extensive documentation of:
- Data sources and assumptions  
- Model development methodology  
- Validation and performance monitoring  

This ensures **model stability**, reproducibility, and justifiable capital allocation based on risk estimates.

---

### 2. Proxy Default Variable: Necessity and Business Risks

#### Why a Proxy Variable Is Necessary

Bati Bank is launching a new Buy-Now-Pay-Later (BNPL) product in partnership with an e-commerce platform. Since this product is new, there is **no historical loan repayment or default data** available.

To enable supervised learning, a **proxy default variable** is created using alternative behavioral data. In this project, customer engagement is summarized using **Recency, Frequency, and Monetary (RFM)** metrics. Customers with low engagement (low frequency and low monetary value) are labeled as **high-risk proxies**, based on the assumption that disengaged customers are more likely to default.

This approach allows the bank to:
- Launch the BNPL service with a data-driven risk framework  
- Estimate credit risk before true default outcomes are observed  
- Align with alternative credit scoring practices used in emerging markets  

#### Business Risks of Proxy-Based Modeling

Using a proxy instead of actual default introduces several risks:

- **Misclassification Risk**
  - *False Positives*: Creditworthy customers labeled as high-risk → lost revenue and poor customer experience  
  - *False Negatives*: Risky customers labeled as low-risk → financial loss due to default  

- **Model Drift**
  As real repayment data becomes available, the proxy may no longer reflect true default behavior, requiring model redevelopment.

- **Bias and Fairness Risks**
  Behavioral proxies may unintentionally capture platform-specific or demographic biases, leading to unfair or discriminatory credit decisions.

To mitigate these risks, the proxy-based model should be used as a **decision-support tool**, with continuous monitoring and retraining once true default data is available.

---

### 3. Trade-offs Between Interpretable and Complex Models

| Aspect | Interpretable Model (Logistic Regression + WoE) | Complex Model (Gradient Boosting) |
|------|-----------------------------------------------|----------------------------------|
| Predictive Performance | Moderate but stable | Higher |
| Interpretability | High | Low (black-box) |
| Regulatory Acceptance | Strong | Challenging |
| Explainability | Built-in via coefficients & WoE | Requires SHAP/LIME |
| Development Effort | Higher upfront (binning, WoE) | Lower upfront |
| Stability & Monitoring | High | Lower, more sensitive to drift |

#### Conclusion

In a regulated financial environment such as Bati Bank, **interpretability, auditability, and regulatory compliance often outweigh marginal performance gains**. Logistic Regression with WoE provides transparency, stability, and ease of governance, making it suitable for initial deployment.

Complex models such as Gradient Boosting are best used as **challenger models** or benchmarking tools, particularly when performance optimization is required and sufficient explainability controls are in place.

---

### Task 2: Exploratory Data Analysis (EDA)
- Dataset overview, summary statistics, and distributions.
- Correlation and outlier analysis.
- Missing values detection.
- Key insights for feature engineering.

### Task 3: Feature Engineering
- Aggregate features: total, average, std, count.
- Time-based features: hour, day, month, year.
- Encode categorical variables.
- Handle missing values.
- Normalize/standardize numeric features.
- WoE and IV transformations.

### Task 4: Proxy Target Variable Engineering
- Compute RFM metrics.
- Cluster customers with K-Means.
- Assign `is_high_risk` label.

### Task 5: Model Training and Tracking
- Split dataset, train at least two models.
- Hyperparameter tuning.
- Track experiments with MLflow.
- Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC.
- Unit tests for feature engineering.

### Task 6: Model Deployment and CI/CD
- REST API with FastAPI.
- `/predict` endpoint for new customer data.
- Containerize with Docker.
- CI/CD with GitHub Actions.
- Linting and unit tests integration.

---

## Getting Started

1. Clone the repository:
```bash
git clone <repo-url>
cd credit-risk-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch EDA notebook:
```bash
jupyter notebook notebooks/eda.ipynb
```

4. Run FastAPI locally:
```bash
uvicorn src.api.main:app --reload
```

---

## References

- [Basel II Capital Accord](https://www.bis.org/publ/bcbs107.htm)
- [Alternative Credit Scoring – HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Risk Fundamentals – Investopedia](https://www.investopedia.com/terms/c/creditrisk.asp)
- [Xente Challenge Dataset](https://www.kaggle.com/datasets/atwine/xente-challenge)

---

## License
MIT License
```