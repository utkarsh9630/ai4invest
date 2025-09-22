#!/usr/bin/env python3
"""
profiles.py

One-stop script to extract Low/edium/High examples from your NFCS data:

  * mapped  → Map codes into your form’s dropdown strings, print and save UTF-8 CSV.
  * show    → Read that CSV and pretty-print each as Field: Value lines.
  * numeric → The original V1 numeric dicts, print and save CSV.
"""

import argparse
import csv
import pandas as pd

# ─── Config & shared lists ────────────────────────────────────────────────────

EXCEL_FILE  = "risk_profiling_data.xlsx"
CSV_MAPPED  = "sample_profiles_by_bucket.csv"
CSV_NUMERIC = "sample_profiles_numeric.csv"

FEATURE_COLS = [
    "Age Group","Ethnicity","Education Level","Marital Status",
    "Financially dependent children","Annual Household Income",
    "Spending vs Income Past Year","Difficulty covering expenses",
    "Emergency fund to cover 3 Months expenses",
    "Current financial condition satisfaction",
    "Thinking about FC frequency","Account ownership check",
    "Savings/Money market/CD account ownership",
    "Employer-sponsored retirement plan ownership","Homeownership",
    "Regular contribution to a retirement account",
    "Non-retirement investments in stocks, bonds, mutual funds",
    "Self-efficacy","Self-rated overall financial knowledge"
]
NUMERIC_FEATS = FEATURE_COLS + ["Ethnicity","Marital Status"]

# ─── Bucket mapping ────────────────────────────────────────────────────────────
def map_risk(x):
    if x <= 3:   return "Low"
    if x <= 7:   return "Medium"
    return "High"

# ─── UI mapping dicts ──────────────────────────────────────────────────────────
age_map       = {1:"18–24",2:"25–34",3:"35–44",4:"45–54",5:"55–64",6:"65+"}
ethnicity_map = {1:"Hispanic",2:"Non-Hispanic White",3:"Non-Hispanic Black",
                 4:"Non-Hispanic Asian",5:"Other"}
education_map = {1:"High School or Less",2:"Some College",
                 3:"Bachelor’s Degree",4:"Postgraduate Degree"}
marital_map   = {1:"Married",2:"Single",3:"Divorced",4:"Widowed"}
spend_map     = {1:"Spend < Income",2:"Spend = Income",3:"Spend > Income"}
diff_map      = {1:"Not at all difficult",2:"Somewhat difficult",3:"Very difficult"}
emerg_map     = {1:"None",2:"1 month",3:"2 months",4:"3+ months"}
sat_map       = {1:"Very dissatisfied",2:"Somewhat dissatisfied",3:"Neutral",
                 4:"Somewhat satisfied",5:"Very satisfied"}
freq_map      = {1:"Never",2:"Yearly",3:"Quarterly",4:"Monthly"}
yesno_map     = {0:"No",1:"Yes"}

def map_value(col, code):
    """Map numeric code to the exact form string."""
    v = int(code)
    mapping = {
      "Age Group": age_map,
      "Ethnicity": ethnicity_map,
      "Education Level": education_map,
      "Marital Status": marital_map,
      "Spending vs Income Past Year": spend_map,
      "Difficulty covering expenses": diff_map,
      "Emergency fund to cover 3 Months expenses": emerg_map,
      "Current financial condition satisfaction": sat_map,
      "Thinking about FC frequency": freq_map,
      "Homeownership": yesno_map,
      "Regular contribution to a retirement account": yesno_map,
      "Non-retirement investments in stocks, bonds, mutual funds": yesno_map
    }.get(col)
    if mapping:
        return mapping.get(v, str(v))
    # numeric-only fields
    return str(v)

# ─── 1) Extract & save UI-mapped examples ──────────────────────────────────────
def extract_mapped(n=5):
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    df = df[df["Take Risk"].notnull()].copy()
    df["risk_level"] = df["Take Risk"].apply(map_risk)

    # coerce & drop missing
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)

    samples = []
    print()
    for bucket in ["Low","Medium","High"]:
        sub = df[df["risk_level"] == bucket]
        sample = sub.sample(n, random_state=42) if len(sub) >= n else sub
        print(f"=== {bucket.upper()}-RISK EXAMPLES ({len(sample)}) ===")
        for _, r in sample.iterrows():
            example = {c: map_value(c, r[c]) for c in FEATURE_COLS}
            print(example)
            example["risk_level"] = bucket
            samples.append(example)
        print()

    # write UTF-8 CSV so later reads work without special encoding
    cols = FEATURE_COLS + ["risk_level"]
    with open(CSV_MAPPED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(samples)
    print(f"✅ Mapped samples written to {CSV_MAPPED}\n")

# ─── 2) Show pretty Field: Value from the mapped CSV ─────────────────────────
def show_profiles():
    df = pd.read_csv(CSV_MAPPED)  # now UTF-8, so no encoding arg needed
    print()
    for idx, row in df.iterrows():
        print(f"=== {row['risk_level'].upper()}-RISK EXAMPLE #{idx+1} ===")
        for col in FEATURE_COLS:
            print(f"{col}: {row[col]}")
        print()

# ─── 3) Extract & save raw numeric examples ──────────────────────────────────
def extract_numeric(n=5):
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    df = df[df["Take Risk"].notnull()].copy()
    df["risk_level"] = df["Take Risk"].apply(map_risk)

    for c in NUMERIC_FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATS)

    samples = []
    print()
    for bucket in ["Low","Medium","High"]:
        sub = df[df["risk_level"] == bucket]
        sample = sub.sample(n, random_state=42) if len(sub) >= n else sub
        print(f"=== {bucket.upper()}-RISK NUMERIC ({len(sample)}) ===")
        for _, r in sample.iterrows():
            d = {c: int(r[c]) for c in NUMERIC_FEATS}
            print(d)
            d["risk_level"] = bucket
            samples.append(d)
        print()

    cols = NUMERIC_FEATS + ["risk_level"]
    with open(CSV_NUMERIC, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(samples)
    print(f"✅ Numeric samples written to {CSV_NUMERIC}\n")

# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract or show risk-profile examples"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["mapped","show","numeric"],
        help="mapped (UI strings), show (print CSV), numeric (raw numbers)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="How many per bucket (default 5)"
    )
    args = parser.parse_args()

    if args.mode == "show":
        show_profiles()
    elif args.mode == "numeric":
        extract_numeric(n=args.n)
    else:
        extract_mapped(n=args.n)
        show_profiles()
