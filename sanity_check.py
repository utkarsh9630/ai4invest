#!/usr/bin/env python3
import pandas as pd
import joblib

# 1) Load your artifacts
risk_pipe = joblib.load('risk_pipeline.joblib')
label_enc = joblib.load('risk_label_encoder.joblib')

# 2) Pick one of the “clean Low-risk” dicts from extract_low_profiles.py:
#    (copy it exactly, including numeric types)
sample = {
    'Age Group': 1,
    'Education Level': 2,
    'Financially dependent children': 2,
    'Annual Household Income': 2,
    'Spending vs Income Past Year': 3,
    'Difficulty covering expenses': 2,
    'Emergency fund to cover 3 Months expenses': 2,
    'Current financial condition satisfaction': 2,
    'Thinking about FC frequency': 1,
    'Account ownership check': 1,
    'Savings/Money market/CD account ownership': 2,
    'Employer-sponsored retirement plan ownership': 1,
    'Homeownership': 1,
    'Regular contribution to a retirement account': 1,
    'Non-retirement investments in stocks, bonds, mutual funds': 1,
    'Self-efficacy': 1,
    'Self-rated overall financial knowledge': 2,
    'Ethnicity': 1,
    'Marital Status': 2
}

# 3) Build a one-row DataFrame & predict
df = pd.DataFrame([sample])
pred_idx = risk_pipe.predict(df)[0]
pred_label = label_enc.inverse_transform([pred_idx])[0]

print(f"Sanity-check prediction for this row → {pred_label}")
