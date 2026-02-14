# Credit-Wise: Loan Approval Prediction System

## 📋 Overview

Credit-Wise is an end-to-end machine learning pipeline designed to automate loan approval decisions. This project implements a binary classification system that predicts whether a loan application should be approved or rejected based on applicant financial and personal information.

**🌐 [Try the Live App](https://loanify.streamlit.app/)**

## 🎯 Features

- **Multi-Model Comparison**: Implements and compares three supervised learning algorithms:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Gaussian Naive Bayes

- **Comprehensive Data Pipeline**:
  - Missing value imputation (mean for numerical, mode for categorical)
  - Feature encoding (Label Encoding & One-Hot Encoding)
  - Feature scaling with StandardScaler
  - Feature engineering (polynomial features for Credit Score & DTI Ratio)

- **Interactive Web Application**: Streamlit-based UI for real-time loan approval predictions

- **Model Evaluation**: Complete evaluation metrics including Precision, Recall, F1-Score, and Accuracy

## 📊 Dataset

The system uses a comprehensive loan dataset with 20+ features including:
- **Financial**: Applicant Income, Coapplicant Income, Loan Amount, Savings, Collateral Value
- **Credit**: Credit Score, Debt-to-Income Ratio, Existing Loans
- **Personal**: Age, Education Level, Employment Status, Marital Status, Dependents
- **Loan Details**: Loan Purpose, Loan Term, Property Area

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python |
| ML Framework | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Web Interface | Streamlit |

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Ehsaan08-ai/Credit-Wise-Loan-Approval-System.git

# Install all dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
Credit-Wise-Loan-Approval-System/
├── app.py                    # Streamlit web application
├── main.ipynb                # Jupyter notebook with EDA & model training
├── requirements.txt          # Python dependencies
├── dataset/
│   └── loan_approval_data.csv
└── README.md
```

## 📈 Model Performance

The Logistic Regression model serves as the primary classifier in the web application, providing probability estimates for both approval and rejection outcomes.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Ehsaan08-ai**
- GitHub: [@Ehsaan08-ai](https://github.com/Ehsaan08-ai)
