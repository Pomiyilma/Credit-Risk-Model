# Credit Scoring Business Understanding

## 1. Basel II and the Need for an Interpretable and Well-Documented Model

The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory compliance in credit risk management. Since credit decisions directly affect customers and financial institutions, banks must be able to explain how risk predictions are generated. As a result, credit scoring models should be interpretable, auditable, and supported by comprehensive documentation.

An interpretable model allows risk analysts, auditors, and regulators to understand the factors influencing a customer's risk score. Well-documented models also facilitate validation, monitoring, maintenance, and regulatory review throughout the model lifecycle. These requirements reduce the risks associated with "black-box" systems, where decisions cannot be easily explained or justified. Therefore, Basel II encourages the use of modeling approaches that balance predictive performance with transparency and accountability.

## 2. The Need for a Proxy Variable and Its Associated Risks

The provided dataset does not contain a direct measure of loan default or repayment behavior. Since supervised machine learning models require a target variable, a proxy variable must be created to represent credit risk.

In this project, customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) metrics will be used to identify customers who appear less engaged with the platform. These customers will be treated as high-risk proxies, while more active customers will be treated as lower-risk proxies.

Although this approach enables model development, it introduces several business risks. First, the proxy variable is an assumption rather than a true measure of default. Customers classified as high risk may not necessarily default on future loans, while some low-risk customers may still fail to repay. Second, any inaccuracies in the proxy definition can propagate into the model, potentially leading to biased predictions and suboptimal lending decisions. Consequently, results produced by the model should be interpreted as estimates of risk based on behavioral patterns rather than actual default outcomes.

## 3. Trade-Offs Between Interpretable and High-Performance Models

A simple and interpretable model such as Logistic Regression combined with Weight of Evidence (WoE) transformation offers several advantages in regulated financial environments. The relationship between features and predictions is transparent, making the model easier to explain, validate, document, and monitor. This level of interpretability aligns well with regulatory expectations and facilitates stakeholder trust.

In contrast, advanced models such as Gradient Boosting often achieve higher predictive performance because they can capture complex and non-linear relationships within the data. However, they are generally more difficult to interpret and explain. This increased complexity can create challenges during regulatory reviews, model validation, and risk governance processes.

Therefore, the choice between the two approaches involves a trade-off between transparency and predictive power. While Gradient Boosting may provide superior accuracy, Logistic Regression with WoE often remains attractive in credit risk applications because its decisions are easier to justify and audit. In practice, financial institutions frequently compare both approaches and select a model that satisfies performance requirements while maintaining an acceptable level of interpretability and regulatory compliance.
