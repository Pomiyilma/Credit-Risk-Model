# Credit Risk Modeling for Buy-Now-Pay-Later Services

Access to credit remains one of the biggest barriers for many customers in emerging digital economies. Traditional credit scoring approaches often depend on repayment histories, credit bureau records, and extensive financial data—resources that may not be available for many potential borrowers.

This project addresses that challenge by building an end-to-end credit risk modeling system for a Buy-Now-Pay-Later (BNPL) service offered by Bati Bank in partnership with an eCommerce platform. Using transaction-level behavioral data from Xente, the project develops a framework capable of identifying potentially risky customers, generating risk scores, and serving predictions through a production-ready API.

Rather than relying on conventional credit histories, the solution leverages customer transaction behavior to uncover meaningful patterns that can be transformed into actionable credit risk signals.

---

## Business Problem

The core objective is to help Bati Bank make informed lending decisions in situations where traditional default labels and credit histories are unavailable. To achieve this, the project focuses on answering a critical question:

> Can customer transaction behavior be transformed into a reliable indicator of future credit risk?

The resulting system is designed to support:

* Loan approval decisions
* Credit limit assignment
* Risk-based loan pricing
* Portfolio risk management
* Real-time credit scoring

---

## Project Journey

### Step 1 — Business Understanding

The project began by exploring the fundamentals of credit risk modeling, alternative credit scoring, and Basel II regulatory principles Particular attention was given to:

* Risk measurement and model governance
* Explainability and interpretability requirements
* Documentation and auditability
* Trade-offs between transparent and high-performing models

This foundation guided every modeling decision made throughout the project.

---

### Step 2 — Exploratory Data Analysis

The Xente transaction dataset was explored to understand customer behavior, data quality, and underlying patterns. The analysis focused on:

* Transaction distributions
* Customer activity patterns
* Categorical feature behavior
* Missing value assessment
* Correlation analysis
* Outlier identification

These insights informed the subsequent feature engineering strategy.

---

### Step 3 — Feature Engineering

A reproducible data processing pipeline was developed to transform raw transaction data into model-ready features. The pipeline combines:

* Behavioral aggregates
* Temporal transaction patterns
* Encoded categorical information
* Scaled numerical variables
* Risk-oriented transformations

The objective was to create features that better represent customer engagement and financial behavior than raw transactions alone.

---

### Step 4 — Behavioral Risk Labeling

Since the dataset does not contain an explicit default indicator, a proxy target variable was engineered. Customer engagement was quantified using:

* Recency
* Frequency
* Monetary value

These behavioral signals were used to segment customers and identify groups exhibiting characteristics commonly associated with elevated credit risk. The resulting risk label provides a practical foundation for supervised machine learning in the absence of historical repayment data.

---

### Step 5 — Model Development and Experiment Tracking

Multiple machine learning models were trained, evaluated, and compared using a structured experimentation workflow. The project incorporates:

* Reproducible train-test evaluation
* Hyperparameter optimization
* Performance benchmarking
* MLflow experiment tracking
* Model versioning and comparison

This approach ensures transparency, repeatability, and informed model selection.

---

### Step 6 — Deployment and Production Readiness

The final stage transformed the model from a research artifact into a deployable service. The solution includes:

* FastAPI-based prediction service
* Docker containerization
* CI/CD automation with GitHub Actions
* Automated testing and validation
* Real-time risk scoring endpoint

This architecture enables the model to be integrated into production environments where lending decisions must be delivered quickly and consistently.

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* MLflow
* FastAPI
* Docker
* GitHub Actions
* Pytest

---

## Key Outcome

This project demonstrates how alternative behavioral data can be transformed into a scalable credit risk assessment framework when traditional credit histories are unavailable.

By combining customer transaction analytics, machine learning, experiment tracking, and modern deployment practices, the solution provides a foundation for responsible, explainable, and production-ready credit decisioning in digital lending environments.

---

## Repository Structure

```text
credit-risk-model/
├── data/
├── notebooks/
├── src/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .github/workflows/
```

---

## Future Improvements

Potential enhancements include:

* Integration of actual repayment and default data
* Advanced gradient boosting models
* Continuous model monitoring and drift detection
* Additional alternative data sources
* Automated retraining pipelines

---



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


**Author:** Pomi Yilma

**Focus Areas:** Credit Risk Modeling • Machine Learning • MLOps • Model Deployment
