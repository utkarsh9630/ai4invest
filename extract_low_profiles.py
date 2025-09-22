#!/usr/bin/env python3
import pandas as pd

# 1) load & map risk
def map_risk(x):
    if x <= 3:   return 'Low'
    elif x <= 7: return 'Medium'
    else:        return 'High'

df = pd.read_excel('risk_profiling_data.xlsx', engine='openpyxl')
df = df[df['Take Risk'].notnull()].copy()
df['risk_level'] = df['Take Risk'].apply(map_risk)

# 2) all the features we coerce‐to‐numeric in train_risk_model.py
feats = [
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
    'Self-rated overall financial knowledge',
    'Ethnicity',
    'Marital Status'
]

# 3) coerce & drop any rows with NaN
for c in feats:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=feats + ['risk_level'])

# 4) enforce plausible ranges
ranges = {
    'Age Group':             (1, 6),
    'Education Level':       (1, 4),
    'Financially dependent children': (0, 10),
    'Annual Household Income':        (1, 9),
    'Spending vs Income Past Year':   (1, 3),
    'Difficulty covering expenses':   (1, 3),
    'Emergency fund to cover 3 Months expenses': (0, 3),
    'Current financial condition satisfaction':   (1, 5),
    'Thinking about FC frequency':    (1, 4),
    'Account ownership check':        (0, 10),
    'Savings/Money market/CD account ownership': (0, 10),
    'Employer-sponsored retirement plan ownership': (0, 10),
    'Homeownership': (0, 1),
    'Regular contribution to a retirement account': (0, 1),
    'Non-retirement investments in stocks, bonds, mutual funds': (0, 1),
    'Self-efficacy':                (1, 7),
    'Self-rated overall financial knowledge': (1, 7),
    'Ethnicity': (1, 5),
    'Marital Status': (1, 4),
}

clean = df[df['risk_level']=='Low'].copy()
for col, (lo, hi) in ranges.items():
    clean = clean[ clean[col].between(lo, hi) ]

# 5) grab five examples and print as dicts
sample = clean[feats].head(5).to_dict(orient='records')

print("\nHere are 5 *clean* Low-risk profiles. Copy one dict into your sanity_check or Flask mapping:\n")
for rec in sample:
    print(rec)
