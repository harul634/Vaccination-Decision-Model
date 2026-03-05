# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:00:02 2026

@author: harul
"""

"""
STEP 3: FLARE-VAX Main Pipeline
Run: python step3_FLARE_VAX.py

BEFORE RUNNING: Set your OpenAI API key below or as environment variable
Estimated cost: ~$2-5 for SAMPLE_SIZE=50, ~$10-15 for SAMPLE_SIZE=300
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json, csv, os
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIG - CHANGE THESE
# ============================================================
OPENAI_API_KEY = "" 
DATA_PATH = r"C:\Users\harul\Documents\Health_Research\adult24csv\nhis2024_with_patterns.csv"
OUTPUT_DIR = r"C:\Users\harul\Documents\Health_Research\adult24csv"

MODEL_NAME = "gpt-4o-2024-11-20"
SAMPLE_SIZE = 50  # Start small! Change to 300 for full run
TRAIN_RATIO = 0.7

# Output files
PRED_FILE = os.path.join(OUTPUT_DIR, "vax_predictions.csv")
REFLECT_FILE = os.path.join(OUTPUT_DIR, "vax_reflection_base.json")
SUS_FILE = os.path.join(OUTPUT_DIR, "vax_susceptibility_score.json")
BAR_FILE = os.path.join(OUTPUT_DIR, "vax_barrier_score.json")

# ============================================================
# INIT
# ============================================================
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    print("ERROR: Set OPENAI_API_KEY in the script or as environment variable")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_response(system, question):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":system},{"role":"user","content":question}],
        temperature=0.7,
    )
    return response.choices[0].message.content

def save_json(path, data):
    existing = []
    if os.path.exists(path):
        with open(path,'r') as f:
            existing = json.load(f)
    if not isinstance(existing, list): existing = [existing]
    existing.append(data)
    with open(path,'w') as f:
        json.dump(existing, f, indent=2)

# ============================================================
# LOAD DATA
# ============================================================
print("="*60)
print("FLARE-VAX: Vaccination Decision Prediction")
print("="*60)

df = pd.read_csv(DATA_PATH)
for col in df.columns:
    if df[col].dtype in ['int64','float64']:
        df[col] = df[col].replace([7,8,9,97,98,99], np.nan)

df_valid = df.dropna(subset=['vaccinated','age','sex','health_status']).copy()
df_s = df_valid.groupby('vaccinated', group_keys=False).apply(
    lambda x: x.sample(n=min(SAMPLE_SIZE//2, len(x)), random_state=42)
).reset_index(drop=True)

data_len = len(df_s)
train_size = round(data_len * TRAIN_RATIO)
labels = df_s['vaccinated'].values

print(f"Samples: {data_len} | Train: {train_size} | Test: {data_len-train_size}")
print(f"Vax rate: {df_s['vaccinated'].mean():.1%}")

# ============================================================
# BUILD PERSON PROFILES
# ============================================================
def build_profile(row):
    sex = {1:"male",2:"female"}.get(row.get('sex'),"unknown")
    race = {1:"White",2:"Black",3:"AIAN",4:"Asian",5:"NHPI",6:"Multiple"}.get(row.get('race'),"unknown")
    hisp = "Hispanic" if row.get('hispanic')==1 else "non-Hispanic"
    edu = {0:"no formal ed",1:"<high school",2:"HS diploma",3:"some college",4:"Associate",5:"Bachelor's",6:"Master's+"}.get(row.get('education'),"unknown")
    health = {1:"excellent",2:"very good",3:"good",4:"fair",5:"poor"}.get(row.get('health_status'),"unknown")
    
    ins = "uninsured" if row.get('uninsured')==1 else "has Medicare" if row.get('medicare')==1 else "has Medicaid" if row.get('medicaid')==1 else "has insurance"
    
    conds = []
    if row.get('diabetes')==1: conds.append("diabetes")
    if row.get('copd')==1: conds.append("COPD")
    if row.get('heart_disease')==1: conds.append("heart disease")
    if row.get('hypertension')==1: conds.append("hypertension")
    if row.get('cancer_ever')==1: conds.append("cancer history")
    cond_str = ", ".join(conds) if conds else "no chronic conditions"
    
    cost = "has cost barriers" if row.get('any_cost_barrier')==1 else "no cost barriers"
    care = "has usual care place" if row.get('usual_care_place')==1 else "no usual care place"
    region = {1:"Northeast",2:"Midwest",3:"South",4:"West"}.get(row.get('region'),"unknown")
    age = int(row.get('age',0))
    
    return (f"{age}yo {hisp} {race} {sex}, {edu}, {region}. "
            f"Health: {health}. Conditions: {cond_str}. "
            f"Insurance: {ins}. {cost}. {care}.")

def build_sus_info(row):
    health = {1:"excellent",2:"very good",3:"good",4:"fair",5:"poor"}.get(row.get('health_status'),"unknown")
    age = int(row.get('age',0))
    risk = "HIGH-RISK" if age>=65 else "moderate-risk" if age>=50 else "lower-risk"
    conds = []
    if row.get('diabetes')==1: conds.append("diabetes")
    if row.get('copd')==1: conds.append("COPD")  
    if row.get('heart_disease')==1: conds.append("heart disease")
    if row.get('hypertension')==1: conds.append("hypertension")
    cond_str = ", ".join(conds) if conds else "none"
    chronic = "YES" if row.get('high_risk_chronic')==1 else "NO"
    return f"Health: {health}. Age {age} ({risk}). Chronic conditions: {cond_str}. CDC high-risk: {chronic}"

def build_bar_info(row):
    ins = "UNINSURED" if row.get('uninsured')==1 else "Insured"
    cost = "YES" if row.get('any_cost_barrier')==1 else "No"
    stop = "YES" if row.get('stopped_care_cost')==1 else "No"
    care = "Yes" if row.get('usual_care_place')==1 else "NO"
    return f"Insurance: {ins}. Cost barriers: {cost}. Stopped care due to cost: {stop}. Usual care place: {care}"

print("Building profiles...")
profiles, sus_infos, bar_infos = [], [], []
for _, row in df_s.iterrows():
    profiles.append(build_profile(row))
    sus_infos.append(build_sus_info(row))
    bar_infos.append(build_bar_info(row))

# ============================================================
# HBM PROMPT TEMPLATES (replacing FLARE's PADM prompts)
# ============================================================
SYS = "You are an expert at behavioral reasoning using the Health Belief Model."

SUS_PROMPT = """Analyze: An individual is deciding whether to get a flu vaccine. Based on their health profile, summarize their perceived susceptibility to influenza and perceived severity of complications.
Health profile: {info}"""

BAR_PROMPT = """Based on their susceptibility assessment and healthcare access, summarize perceived barriers to vaccination and self-efficacy.
Susceptibility: {sus}
Access info: {info}"""

DECISION_PROMPT = """Determine if this individual will get vaccinated. Use Health Belief Model reasoning (susceptibility, severity, barriers, benefits, self-efficacy). End with: Answer: YES or Answer: NO

Susceptibility Assessment: {sus}
Barrier Assessment: {bar}
Profile: {profile}"""

DECISION_WITH_MEMORY = """Predict vaccination decision using HBM reasoning. Learn from previous examples.

Previous Examples:
{history}

Current individual:
Susceptibility: {sus}
Barriers: {bar}  
Profile: {profile}

End with: Answer: YES or Answer: NO"""

# ============================================================
# MEMORY (simplified from FLARE's FAISS)
# ============================================================
class Memory:
    def __init__(self, k=3):
        self.items = []
        self.k = k
    def save(self, text):
        self.items.append(text)
    def get(self):
        recent = self.items[-self.k:] if len(self.items)>=self.k else self.items
        return "\n".join([f"Example {i+1}: {t}" for i,t in enumerate(recent)]) if recent else ""

memory = Memory(k=3)

# ============================================================
# MAIN LOOP
# ============================================================
print(f"\nRunning pipeline with {MODEL_NAME}...")
print(f"Phase 1: Training (building memory) on {train_size} samples")
print(f"Phase 2: Testing on {data_len-train_size} samples\n")

all_preds, all_outputs = [], []
test_preds, test_actuals = [], []

for idx in tqdm(range(data_len), desc="Processing"):
    is_train = idx < train_size
    try:
        # Step 1: Susceptibility assessment
        sus = get_response(SYS, SUS_PROMPT.format(info=sus_infos[idx]))
        
        # Step 2: Barrier assessment  
        bar = get_response(SYS, BAR_PROMPT.format(sus=sus, info=bar_infos[idx]))
        
        # Step 3: Decision
        hist = memory.get()
        if hist:
            dec = get_response(SYS, DECISION_WITH_MEMORY.format(
                history=hist, sus=sus, bar=bar, profile=profiles[idx]))
        else:
            dec = get_response(SYS, DECISION_PROMPT.format(
                sus=sus, bar=bar, profile=profiles[idx]))
        
        pred = 1 if 'YES' in dec.upper() else 0
        correct = pred == labels[idx]
        
        # Memory RL: store errors during training
        if not correct and is_train:
            label_str = "got vaccinated" if labels[idx]==1 else "did not get vaccinated"
            memory.save(f"Individual actually {label_str}. Profile: {profiles[idx][:150]}")
            save_json(REFLECT_FILE, {"profile": profiles[idx][:150], "actual": int(labels[idx]), "predicted": pred})
        
        all_preds.append(pred)
        all_outputs.append(f"SUS: {sus[:100]}... | BAR: {bar[:100]}... | DEC: {'YES' if pred==1 else 'NO'}")
        
        if not is_train:
            test_preds.append(pred)
            test_actuals.append(labels[idx])
            
    except Exception as e:
        print(f"Error at {idx}: {e}")
        all_preds.append(0)
        all_outputs.append(f"Error: {e}")
        if not is_train:
            test_preds.append(0)
            test_actuals.append(labels[idx])

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

tp, ta = np.array(test_preds), np.array(test_actuals)
print(f"\nTest accuracy: {accuracy_score(ta, tp):.4f}")
print(f"\nClassification Report:")
print(classification_report(ta, tp, target_names=['Not Vaccinated','Vaccinated']))

# Save
results = pd.DataFrame({
    'prediction': all_preds,
    'actual': list(labels[:len(all_preds)]),
    'output': all_outputs
})
results.to_csv(PRED_FILE, index=False)
print(f"\nSaved: {PRED_FILE}")
print("DONE!")