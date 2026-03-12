"""
FLARE-VAX: Full Pipeline for ASU Sol HPC
==========================================
Uses HuggingFace transformers directly (no vLLM needed)
Supports: Llama 3.1, Qwen 2.5

Usage on Sol:
  source activate flare-vax
  python run_flare_vax_hf.py --backend llama --sample_size 1000
  python run_flare_vax_hf.py --backend qwen --sample_size 1000
  python run_flare_vax_hf.py --backend llama --sample_size 0   # all data
"""

import pandas as pd
import numpy as np
import json, os, argparse, time, sys, gc
from tqdm import tqdm
from datetime import datetime
import torch

# ============================================================
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='FLARE-VAX Full Pipeline')
parser.add_argument('--backend', type=str, default='llama', choices=['llama', 'qwen'])
parser.add_argument('--sample_size', type=int, default=1000,
                    help='0 = use all data, otherwise balanced sample')
parser.add_argument('--data_path', type=str, default='adult24csv/nhis2024_with_patterns.csv')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--memory_k', type=int, default=3)
parser.add_argument('--checkpoint_every', type=int, default=100)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================
# MODEL CONFIG
# ============================================================
MODEL_MAP = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}
MODEL_NAME = MODEL_MAP[args.backend]

print(f"\n{'='*60}")
print(f"FLARE-VAX Pipeline")
print(f"{'='*60}")
print(f"Backend: {args.backend} | Model: {MODEL_NAME}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# LOAD MODEL
# ============================================================
print(f"\nLoading {MODEL_NAME}... (may take 2-5 min first time)")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded! GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ============================================================
# LLM CALL
# ============================================================
def get_response(system, question, max_new_tokens=400):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ============================================================
# LOAD DATA
# ============================================================
print(f"\nLoading: {args.data_path}")
df = pd.read_csv(args.data_path)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].replace([7, 8, 9, 97, 98, 99], np.nan)

df_valid = df.dropna(subset=['vaccinated', 'age', 'sex', 'health_status']).copy()

if args.sample_size > 0:
    n = min(args.sample_size // 2, df_valid['vaccinated'].value_counts().min())
    df_s = df_valid.groupby('vaccinated', group_keys=False).apply(
        lambda x: x.sample(n=n, random_state=42)).reset_index(drop=True)
else:
    df_s = df_valid.sample(frac=1, random_state=42).reset_index(drop=True)

data_len = len(df_s)
train_size = round(data_len * args.train_ratio)
labels = df_s['vaccinated'].values
print(f"Samples: {data_len} | Train: {train_size} | Test: {data_len-train_size} | Vax: {labels.mean():.1%}")

# ============================================================
# BASELINES
# ============================================================
print(f"\n{'='*60}")
print("BASELINE MODELS (5-fold CV)")
print(f"{'='*60}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

feat = [c for c in ['age','sex','race','hispanic','education','marital_status','region',
    'income_poverty_ratio','health_status','diabetes','copd','cancer_ever','heart_disease',
    'hypertension','uninsured','medicare','medicaid','cost_barrier_1','cost_barrier_2',
    'usual_care_place','smoking_status','bmi_category','high_risk_chronic','any_cost_barrier',
    'access_barrier','num_children','disability'] if c in df_s.columns]

X = df_s[feat].fillna(0).values
Xs = StandardScaler().fit_transform(X)

bl = {
    'Random Forest': (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X),
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), Xs),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42), X),
    'MLP Neural Net': (MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42, early_stopping=True), Xs),
    'KNN (k=5)': (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), Xs),
    'SVM (RBF)': (SVC(kernel='rbf', random_state=42), Xs),
}
try:
    from xgboost import XGBClassifier
    bl['XGBoost'] = (XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1), X)
except: pass

br = {}
print(f"  {'Method':<25s} {'Acc':>8s} {'Std':>7s}")
print(f"  {'-'*42}")
for name, (m, Xu) in bl.items():
    try:
        s = cross_val_score(m, Xu, labels, cv=5, scoring='accuracy', n_jobs=-1)
        br[name] = {'accuracy': round(float(s.mean()),4), 'std': round(float(s.std()),4)}
        print(f"  {name:<25s} {s.mean():>7.4f} {s.std():>6.4f}")
    except Exception as e:
        print(f"  {name:<25s} FAIL: {e}")

with open(os.path.join(args.output_dir, f'baselines_{args.backend}.json'), 'w') as f:
    json.dump(br, f, indent=2)

# ============================================================
# PROFILES
# ============================================================
def build_profile(r):
    sex={1:"male",2:"female"}.get(r.get('sex'),"unk")
    race={1:"White",2:"Black",3:"AIAN",4:"Asian",5:"NHPI",6:"Multi"}.get(r.get('race'),"unk")
    hisp="Hispanic" if r.get('hispanic')==1 else "non-Hispanic"
    edu={0:"no ed",1:"<HS",2:"HS",3:"some college",4:"Associate",5:"Bachelor's",6:"Master's+"}.get(r.get('education'),"unk")
    hl={1:"excellent",2:"very good",3:"good",4:"fair",5:"poor"}.get(r.get('health_status'),"unk")
    reg={1:"NE",2:"MW",3:"South",4:"West"}.get(r.get('region'),"unk")
    ins="uninsured" if r.get('uninsured')==1 else "Medicare" if r.get('medicare')==1 else "Medicaid" if r.get('medicaid')==1 else "insured"
    cd=[]
    if r.get('diabetes')==1: cd.append("diabetes")
    if r.get('copd')==1: cd.append("COPD")
    if r.get('heart_disease')==1: cd.append("heart disease")
    if r.get('hypertension')==1: cd.append("hypertension")
    if r.get('cancer_ever')==1: cd.append("cancer")
    cs=", ".join(cd) if cd else "no chronic conditions"
    ct="cost barriers" if r.get('any_cost_barrier')==1 else "no cost barriers"
    cr="usual care" if r.get('usual_care_place')==1 else "no usual care"
    return f"{int(r.get('age',0))}yo {hisp} {race} {sex}, {edu}, {reg}. Health:{hl}. {cs}. {ins}. {ct}. {cr}."

def build_sus(r):
    hl={1:"excellent",2:"very good",3:"good",4:"fair",5:"poor"}.get(r.get('health_status'),"unk")
    age=int(r.get('age',0))
    rk="HIGH-RISK" if age>=65 else "moderate" if age>=50 else "lower"
    cd=[]
    if r.get('diabetes')==1: cd.append("diabetes")
    if r.get('copd')==1: cd.append("COPD")
    if r.get('heart_disease')==1: cd.append("heart disease")
    if r.get('hypertension')==1: cd.append("hypertension")
    return f"Health:{hl}. Age {age}({rk}). Conditions:{', '.join(cd) if cd else 'none'}. High-risk:{('Y' if r.get('high_risk_chronic')==1 else 'N')}"

def build_bar(r):
    ins="UNINSURED" if r.get('uninsured')==1 else "Insured"
    cost="Y" if r.get('any_cost_barrier')==1 else "N"
    care="Y" if r.get('usual_care_place')==1 else "N"
    return f"Insurance:{ins}. Cost barriers:{cost}. Usual care:{care}"

print("\nBuilding profiles...")
profiles, sus_infos, bar_infos = [], [], []
for _, r in df_s.iterrows():
    profiles.append(build_profile(r))
    sus_infos.append(build_sus(r))
    bar_infos.append(build_bar(r))

# ============================================================
# PROMPTS
# ============================================================
SYS = "You are an expert at behavioral reasoning using the Health Belief Model."
SP = "Analyze: Person deciding on flu vaccine. Summarize perceived susceptibility and severity. Rate 1-5.\nProfile: {info}"
BP = "Based on susceptibility and access info, summarize barriers to vaccination. Rate 1-5.\nSusceptibility: {sus}\nAccess: {info}"
DP = "Will this person get vaccinated? Use HBM reasoning. End with: Answer: YES or Answer: NO\nSusceptibility: {sus}\nBarriers: {bar}\nProfile: {profile}"
DMP = "Predict vaccination. Learn from errors:\n{history}\n\nCurrent:\nSusceptibility: {sus}\nBarriers: {bar}\nProfile: {profile}\nEnd with: Answer: YES or Answer: NO"

# ============================================================
# MEMORY
# ============================================================
class Memory:
    def __init__(self, k=3):
        self.items = []
        self.k = k
    def store(self, prof, actual, pred, sus, bar):
        a="DID vaccinate" if actual==1 else "DID NOT vaccinate"
        p="would vaccinate" if pred==1 else "would not vaccinate"
        self.items.append(f"ERROR: Predicted {p}, actually {a}. {prof[:120]}")
    def get(self):
        if not self.items: return ""
        return "\n".join([f"Ex{i+1}: {t}" for i,t in enumerate(self.items[-self.k:])])

memory = Memory(k=args.memory_k)

def parse_dec(t):
    u=t.upper()
    if "ANSWER: YES" in u or "ANSWER:YES" in u: return 1
    if "ANSWER: NO" in u or "ANSWER:NO" in u: return 0
    ls=u.strip().split('\n')
    l=ls[-1] if ls else ""
    if "YES" in l: return 1
    if "NO" in l: return 0
    return 1 if u.count("YES")>u.count("NO") else 0

# ============================================================
# MAIN LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"RUNNING FLARE-VAX ({args.backend.upper()}) on {data_len} samples")
print(f"{'='*60}\n")

PRED_FILE = os.path.join(args.output_dir, f'predictions_{args.backend}.csv')
REFLECT_FILE = os.path.join(args.output_dir, f'reflections_{args.backend}.json')

all_preds, all_sus, all_bar, all_dec = [], [], [], []
test_preds, test_actuals = [], []
errors_count = 0
start_time = time.time()

for idx in tqdm(range(data_len), desc=f"FLARE-VAX"):
    is_train = idx < train_size
    try:
        sus = get_response(SYS, SP.format(info=sus_infos[idx]))
        bar = get_response(SYS, BP.format(sus=sus[:250], info=bar_infos[idx]))
        hist = memory.get()
        if hist:
            dec = get_response(SYS, DMP.format(history=hist, sus=sus[:250], bar=bar[:250], profile=profiles[idx]))
        else:
            dec = get_response(SYS, DP.format(sus=sus[:250], bar=bar[:250], profile=profiles[idx]))

        pred = parse_dec(dec)
        if pred != labels[idx] and is_train:
            errors_count += 1
            memory.store(profiles[idx], int(labels[idx]), pred, sus[:150], bar[:150])
            try:
                ex = json.load(open(REFLECT_FILE)) if os.path.exists(REFLECT_FILE) else []
                ex.append({"idx":idx,"actual":int(labels[idx]),"predicted":pred,"profile":profiles[idx][:150]})
                json.dump(ex, open(REFLECT_FILE,'w'))
            except: pass

        all_preds.append(pred)
        all_sus.append(sus[:150])
        all_bar.append(bar[:150])
        all_dec.append(dec[:150])
        if not is_train:
            test_preds.append(pred)
            test_actuals.append(labels[idx])
        if idx % 50 == 0: torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n  Err@{idx}: {e}")
        all_preds.append(0); all_sus.append(""); all_bar.append(""); all_dec.append(str(e))
        if not is_train: test_preds.append(0); test_actuals.append(labels[idx])

    if (idx+1) % args.checkpoint_every == 0:
        el = time.time()-start_time
        rate = (idx+1)/el
        eta = (data_len-idx-1)/rate/3600
        acc = (np.array(all_preds)==labels[:len(all_preds)]).mean()
        print(f"\n  [{idx+1}/{data_len}] {el/60:.0f}min | {rate:.2f}/s | ETA:{eta:.1f}h | Acc:{acc:.3f} | Err:{errors_count}")
        pd.DataFrame({'prediction':all_preds,'actual':list(labels[:len(all_preds)])}).to_csv(PRED_FILE.replace('.csv','_ckpt.csv'),index=False)

# ============================================================
# RESULTS
# ============================================================
el = time.time()-start_time
print(f"\n{'='*60}")
print(f"RESULTS — {args.backend.upper()} ({MODEL_NAME})")
print(f"{'='*60}")
print(f"Time: {el/3600:.2f}h | Samples: {len(all_preds)}")

tp_arr = np.array(all_preds[:train_size])
ta_arr = labels[:train_size]
tr_acc = accuracy_score(ta_arr, tp_arr)
print(f"Training: {tr_acc:.4f} ({sum(tp_arr==ta_arr)}/{train_size}) | Errors: {errors_count}")

te_acc, te_f1 = None, None
if test_preds:
    tp, ta = np.array(test_preds), np.array(test_actuals)
    te_acc = accuracy_score(ta, tp)
    te_f1 = f1_score(ta, tp, average='weighted')
    print(f"Test: {te_acc:.4f} | F1: {te_f1:.4f}")
    print(classification_report(ta, tp, target_names=['Not Vax','Vax'], labels=[0,1]))

pd.DataFrame({'prediction':all_preds,'actual':list(labels[:len(all_preds)]),
    'susceptibility':all_sus,'barriers':all_bar,'decision':all_dec,
    'phase':['train' if i<train_size else 'test' for i in range(len(all_preds))]
}).to_csv(PRED_FILE, index=False)

json.dump({
    'backend':args.backend,'model':MODEL_NAME,'samples':data_len,
    'train_acc':float(tr_acc),'test_acc':float(te_acc) if te_acc else None,
    'test_f1':float(te_f1) if te_f1 else None,'errors':errors_count,
    'time_hours':el/3600,'baselines':br
}, open(os.path.join(args.output_dir,f'summary_{args.backend}.json'),'w'), indent=2)

print(f"\nSaved: {PRED_FILE}")
print("DONE!")
