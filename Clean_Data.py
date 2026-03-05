# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:54:48 2026

@author: harul
"""

"""
STEP 1: Clean NHIS 2024 Raw Data
Run: python step1_clean_data.py
"""
import pandas as pd
import numpy as np

RAW_DATA_PATH = r"C:\Users\harul\Downloads\adult24csv\adult24.csv"
OUTPUT_PATH = r"C:\Users\harul\Downloads\adult24csv\nhis2024_vaccination_clean.csv"

print("STEP 1: Cleaning NHIS 2024 Data")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Raw: {df.shape[0]} rows, {df.shape[1]} columns")

cols = {
    "SHTFLU12M_A":"flu_shot_12m","SHTCVD191_A":"covid_vac_ever","SHTCVD19NM2_A":"covid_vac_doses",
    "SHTPNUEV_A":"pneumonia_vac_ever","SHTHEPA_A":"hepa_vac","AGEP_A":"age","SEX_A":"sex",
    "RACEALLP_A":"race","HISPALLP_A":"hispanic","EDUCP_A":"education","MARITAL_A":"marital_status",
    "REGION":"region","NATUSBORN_A":"us_born","CITZNSTP_A":"citizenship","RATCAT_A":"income_poverty_ratio",
    "HICOV_A":"has_insurance","COVER_A":"insurance_type","NOTCOV_A":"uninsured","MEDICAID_A":"medicaid",
    "MEDICARE_A":"medicare","HICOSTR1_A":"cost_barrier_1","HICOSTR2_A":"cost_barrier_2",
    "RSNHICOST_A":"reason_no_ins_cost","HISTOPCOST_A":"stopped_care_cost","USUALPL_A":"usual_care_place",
    "PHSTAT_A":"health_status","DIBEV_A":"diabetes","COPDEV_A":"copd","CANEV_A":"cancer_ever",
    "CHDEV_A":"heart_disease","ANGEV_A":"angina","STREV_A":"stroke","HYPEV_A":"hypertension",
    "DISAB3_A":"disability","BMICAT_A":"bmi_category","PCNTKIDS_A":"num_children",
    "SAPARENTSC_A":"parent_status","SMOKELSCR1_A":"smoking_status",
}

avail = {k:v for k,v in cols.items() if k in df.columns}
c = df[list(avail.keys())].copy()
c.rename(columns=avail, inplace=True)
c = c[c["flu_shot_12m"].isin([1,2])].copy()
c["vaccinated"] = (c["flu_shot_12m"]==1).astype(int)

chronic = [x for x in ["diabetes","copd","cancer_ever","heart_disease","angina","stroke","hypertension"] if x in c.columns]
c["high_risk_chronic"] = c[chronic].apply(lambda r: 1 if any(r==1) else 0, axis=1)
barr = [x for x in ["cost_barrier_1","cost_barrier_2","reason_no_ins_cost","stopped_care_cost"] if x in c.columns]
c["any_cost_barrier"] = c[barr].apply(lambda r: 1 if any(r==1) else 0, axis=1)
c["access_barrier"] = (c["uninsured"]==1).astype(int)

print(f"Clean: {len(c)} rows, {len(c.columns)} columns")
print(f"Vaccination rate: {c['vaccinated'].mean():.1%}")
c.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
print("DONE! Now run step2_variable_selection.py")