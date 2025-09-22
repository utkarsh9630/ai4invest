#!/usr/bin/env python3
"""
train_risk_model.py

Loads the cleaned NFCS risk‐profiling dataset, builds a preprocessing + RandomForest pipeline
(with OneHotEncoder(handle_unknown='ignore')), evaluates, and saves it.
"""

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

# 1) Define your feature sets
numeric_feats = [
    'Age Group',
    'Education Level',
    'Financially dependent children',
    'Annual Household Income',
    'Spending vs Income Past Year',
    'Difficulty covering expenses',
    'Emergency fund to cover 3 Months expenses',
    'Current financial condition satisfaction',
    'Thinking about FC frequency',
    'Account ownership check',
    'Savings/Money market/CD account ownership',
    'Employer-sponsored retirement plan ownership',
    'Homeownership',
    'Regular contribution to a retirement account',
    'Non-retirement investments in stocks, bonds, mutual funds',
    'Self-efficacy',
    'Self-rated overall financial knowledge'
]

onehot_feats = [
    'Ethnicity',
    'Marital Status'
]

def map_risk(x):
    if x <= 3:
        return 'Low'
    elif x <= 7:
        return 'Medium'
    else:
        return 'High'

def main():
    # --- LOAD ---
    df = pd.read_excel('risk_profiling_data.xlsx', engine='openpyxl')
    df = df[df['Take Risk'].notnull()].copy()

    # --- TARGET ---
    df['risk_level'] = df['Take Risk'].apply(map_risk)
    le = LabelEncoder()
    df['risk_label'] = le.fit_transform(df['risk_level'])
    print("Classes:", le.classes_)

    # --- COERCE NUMERICS ---
    for col in numeric_feats:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- TRAIN/TEST SPLIT ---
    X = df[numeric_feats + onehot_feats]
    y = df['risk_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"Train/Test samples: {len(X_train)} / {len(X_test)}")

    # --- PIPELINE ---
    preprocessor = ColumnTransformer([
        # 1) Median‐impute numeric columns
        ('num', SimpleImputer(strategy='median'), numeric_feats),

        # 2) One‐hot encode Ethnicity & Marital Status, ignore unseen categories
        ('ohe',
         OneHotEncoder(
             drop='first',
             sparse_output=False,
             handle_unknown='ignore'
         ),
         onehot_feats),
    ], remainder='drop')

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # --- TRAIN ---
    pipeline.fit(X_train, y_train)

    # --- EVALUATE ---
    y_pred = pipeline.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nBalanced accuracy: {bal_acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- SAVE ARTIFACTS ---
    joblib.dump(pipeline, 'risk_pipeline.joblib')
    joblib.dump(le, 'risk_label_encoder.joblib')
    print("\n✅ Saved pipeline → risk_pipeline.joblib")
    print("✅ Saved label encoder → risk_label_encoder.joblib")

if __name__ == "__main__":
    main()
