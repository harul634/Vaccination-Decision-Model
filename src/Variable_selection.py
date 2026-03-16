# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:55:46 2026

@author: harul
"""

"""
STEP 2: FLARE Variable Selection + HBM Pattern Classification
Run: python step2_variable_selection.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = r"C:\Users\harul\Documents\Health_Research\adult24csv\nhis2024_vaccination_clean.csv"
OUTPUT_PATH = r"C:\Users\harul\Documents\Health_Research\adult24csv\nhis2024_with_patterns.csv"
IMPORTANCE_PATH = r"C:\Users\harul\Documents\Health_Research\adult24csv\rf_feature_importance.csv"

print("="*60)
print("STEP 2: FLARE Variable Selection - Vaccination Adaptation")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

demographic_vars = ["age","sex","race","hispanic","education","marital_status","region",
                    "us_born","citizenship","income_poverty_ratio","num_children","parent_status",
                    "smoking_status","bmi_category"]
susceptibility_vars = ["health_status","diabetes","copd","cancer_ever","heart_disease",
                       "angina","stroke","hypertension","high_risk_chronic","disability"]
barrier_vars = ["uninsured","has_insurance","insurance_type","medicaid","medicare",
                "cost_barrier_1","cost_barrier_2","reason_no_ins_cost","stopped_care_cost",
                "usual_care_place","any_cost_barrier","access_barrier"]

all_vars = [v for v in demographic_vars+susceptibility_vars+barrier_vars if v in df.columns]
X = df[all_vars].copy()
for col in X.columns:
    X[col] = X[col].replace([7,8,9,97,98,99], np.nan)
    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
y = df["vaccinated"].values

# --- FLARE Weighted Regression ---
print("\n--- Weighted Regression Variable Selection ---")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

np.random.seed(42)
n = len(all_vars)
cov = np.eye(n)*0.05
eps = np.random.multivariate_normal(np.zeros(n), cov, size=len(y))
model = LinearRegression()
model.fit(X_scaled, y + eps[:,0])

w = pd.DataFrame({'Variable':all_vars,'Weight':np.abs(model.coef_)})
w = w.sort_values('Weight',ascending=False).reset_index(drop=True)
w['Cum_Pct'] = w['Weight'].cumsum()/w['Weight'].sum()
selected = w[w['Cum_Pct']<=0.70]
print(f"Top variables (70% weight):")
print(selected[['Variable','Weight','Cum_Pct']].to_string(index=False))

# --- Random Forest Importance ---
print("\n--- Random Forest Feature Importance ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_imp = pd.DataFrame({'Variable':all_vars,'RF_Importance':rf.feature_importances_})
rf_imp = rf_imp.sort_values('RF_Importance',ascending=False)
print(rf_imp.head(15).to_string(index=False))

acc = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()
print(f"\nBaseline RF accuracy: {acc:.3f}")

# --- HBM Construct Mapping ---
print("\n--- HBM Construct Mapping ---")
for _,row in rf_imp.head(15).iterrows():
    v = row['Variable']
    if v in susceptibility_vars: c = "SUSCEPTIBILITY"
    elif v in barrier_vars: c = "BARRIERS"
    else: c = "DEMOGRAPHIC"
    print(f"  {v:25s} -> {c:20s} ({row['RF_Importance']:.4f})")

# --- 4 Reasoning Patterns ---
print("\n--- HBM Reasoning Patterns ---")
df['health_status'] = df['health_status'].replace([7,8,9], 3)
df['susceptibility_score'] = ((df['age']>50).astype(int) + 
    df.get('high_risk_chronic',0) + (df['health_status']>=4).astype(int))
df['barrier_score'] = df.get('access_barrier',0) + df.get('any_cost_barrier',0)

conditions = [
    (df['susceptibility_score']>=2) & (df['barrier_score']==0),
    (df['susceptibility_score']>=2) & (df['barrier_score']>=1),
    (df['susceptibility_score']<2) & (df['barrier_score']==0),
    (df['susceptibility_score']<2) & (df['barrier_score']>=1),
]
df['reasoning_pattern'] = np.select(conditions, [0,1,2,3], default=2)

labels = ['High-Sus/Low-Bar','High-Sus/High-Bar','Low-Sus/Low-Bar','Low-Sus/High-Bar']
print(f"{'Pattern':<30s} {'N':>8s} {'Vax Rate':>10s}")
print("-"*50)
for i,label in enumerate(labels):
    sub = df[df['reasoning_pattern']==i]
    print(f"  {label:<28s} {len(sub):>8d} {sub['vaccinated'].mean():>9.1%}")

df.to_csv(OUTPUT_PATH, index=False)
rf_imp.to_csv(IMPORTANCE_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Saved: {IMPORTANCE_PATH}")
print("DONE! Now run step3_FLARE_VAX.py (requires OpenAI API key)")