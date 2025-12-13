# Credit Risk Probability Model for Alternative Data

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
