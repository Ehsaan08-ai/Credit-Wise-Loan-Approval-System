import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

st.title("Loan Approval Predictor 💰")
st.write("Welcome! Enter your details below to check your eligibility.")

# Load dataset for training the model
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("dataset/loan_approval_data.csv")
    
    # Handle missing values
    categorical_cols = df.select_dtypes(include="object").columns
    nums_cols = df.select_dtypes(include="number").columns
    
    num_imputer = SimpleImputer(strategy="mean")
    df[nums_cols] = num_imputer.fit_transform(df[nums_cols])
    
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    # Remove Applicant_ID
    df = df.drop("Applicant_ID", axis=1)
    
    # Label Encoding for Education_Level and Loan_Approved
    le_education = LabelEncoder()
    df["Education_Level"] = le_education.fit_transform(df["Education_Level"])
    
    le_loan = LabelEncoder()
    df["Loan_Approved"] = le_loan.fit_transform(df["Loan_Approved"])
    
    # One-Hot Encoding
    cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
    df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    
    # Feature Engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    
    # Prepare features (drop original DTI_Ratio and Credit_Score, use squared versions)
    X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
    y = df["Loan_Approved"]
    
    # Feature names for prediction
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, le_education, ohe, feature_names

# Load model
model, scaler, le_education, ohe, feature_names = load_and_train_model()

st.markdown("### Enter Your Details")
st.markdown("---")

# Create input fields in a clean single-column layout

# Numerical Inputs
st.markdown("#### Financial Information")
applicant_income = st.number_input("Applicant Income ($)", min_value=0, max_value=1000000, value=5000, step=100)
coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, max_value=1000000, value=0, step=100)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=10000, step=100)
loan_term = st.slider("Loan Term (months)", min_value=6, max_value=84, value=36, step=6)
savings = st.number_input("Savings ($)", min_value=0, max_value=1000000, value=5000, step=100)
collateral_value = st.number_input("Collateral Value ($)", min_value=0, max_value=1000000, value=10000, step=100)

st.markdown("#### Personal Information")
age = st.slider("Age (years)", min_value=18, max_value=80, value=30)
dependents = st.slider("Number of Dependents", min_value=0, max_value=5, value=0)

st.markdown("#### Credit & Loan Details")
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
existing_loans = st.slider("Number of Existing Loans", min_value=0, max_value=10, value=0)

st.markdown("#### Categorical Information")

# Employment Status (One-Hot Encoded - dropped first: Contract)
employment_status = st.selectbox(
    "Employment Status",
    options=["Contract", "Salaried", "Self-employed", "Unemployed"]
)

# Marital Status (One-Hot Encoded - dropped first: Married)
marital_status = st.selectbox(
    "Marital Status",
    options=["Married", "Single"]
)

# Loan Purpose (One-Hot Encoded - dropped first: Business)
loan_purpose = st.selectbox(
    "Loan Purpose",
    options=["Business", "Car", "Education", "Home", "Personal"]
)

# Property Area (One-Hot Encoded - dropped first: Rural)
property_area = st.selectbox(
    "Property Area",
    options=["Rural", "Semiurban", "Urban"]
)

# Gender (One-Hot Encoded - dropped first: Female)
gender = st.selectbox(
    "Gender",
    options=["Female", "Male"]
)

# Employer Category (One-Hot Encoded - dropped first: Business)
employer_category = st.selectbox(
    "Employer Category",
    options=["Business", "Government", "MNC", "Private", "Unemployed"]
)

# Education Level (Label Encoded)
education_level = st.selectbox(
    "Education Level",
    options=["Graduate", "Not Graduate"]
)

st.markdown("---")

# Predict button
if st.button("Predict Loan Approval", type="primary"):
    # Create feature dictionary with all features in correct order
    # Numerical features
    input_data = {
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age': age,
        'Dependents': dependents,
        'Existing_Loans': existing_loans,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
    }
    
    # Label Encode Education Level
    education_encoded = le_education.transform([education_level])[0]
    input_data['Education_Level'] = education_encoded
    
    # One-Hot Encode categorical features
    # Employment Status: Contract is dropped (reference category)
    input_data['Employment_Status_Salaried'] = 1 if employment_status == "Salaried" else 0
    input_data['Employment_Status_Self-employed'] = 1 if employment_status == "Self-employed" else 0
    input_data['Employment_Status_Unemployed'] = 1 if employment_status == "Unemployed" else 0
    
    # Marital Status: Married is dropped (reference category)
    input_data['Marital_Status_Single'] = 1 if marital_status == "Single" else 0
    
    # Loan Purpose: Business is dropped (reference category)
    input_data['Loan_Purpose_Car'] = 1 if loan_purpose == "Car" else 0
    input_data['Loan_Purpose_Education'] = 1 if loan_purpose == "Education" else 0
    input_data['Loan_Purpose_Home'] = 1 if loan_purpose == "Home" else 0
    input_data['Loan_Purpose_Personal'] = 1 if loan_purpose == "Personal" else 0
    
    # Property Area: Rural is dropped (reference category)
    input_data['Property_Area_Semiurban'] = 1 if property_area == "Semiurban" else 0
    input_data['Property_Area_Urban'] = 1 if property_area == "Urban" else 0
    
    # Gender: Female is dropped (reference category)
    input_data['Gender_Male'] = 1 if gender == "Male" else 0
    
    # Employer Category: Business is dropped (reference category)
    input_data['Employer_Category_Government'] = 1 if employer_category == "Government" else 0
    input_data['Employer_Category_MNC'] = 1 if employer_category == "MNC" else 0
    input_data['Employer_Category_Private'] = 1 if employer_category == "Private" else 0
    input_data['Employer_Category_Unemployed'] = 1 if employer_category == "Unemployed" else 0
    
    # Feature Engineering: Squared features
    input_data['DTI_Ratio_sq'] = dti_ratio ** 2
    input_data['Credit_Score_sq'] = credit_score ** 2
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data])
    
    # Ensure columns are in the exact order expected by the model
    input_df = input_df[feature_names]
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display result
    st.markdown("---")
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.success("✅ **APPROVED** - Congratulations! Your loan application is likely to be approved.")
        st.info(f"Confidence: {prediction_proba[1]*100:.1f}%")
    else:
        st.error("❌ **REJECTED** - Unfortunately, your loan application is likely to be rejected.")
        st.info(f"Confidence: {prediction_proba[0]*100:.1f}%")
    
    # Show probability breakdown
    st.markdown("#### Probability Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Probability", f"{prediction_proba[1]*100:.1f}%")
    with col2:
        st.metric("Rejection Probability", f"{prediction_proba[0]*100:.1f}%")
