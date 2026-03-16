"""
FLARE Model Comparison: Test Qwen/Llama on Original Wildfire Task
==================================================================
Uses the existing FLARE GPT-4o outputs to extract the reasoning prompts,
then re-runs those same prompts through Qwen and Llama to compare performance.

This validates whether FLARE's success is due to the large commercial model (GPT-4o)
or the architecture itself.

Usage on Sol:
  python test_flare_original.py --backend qwen
  python test_flare_original.py --backend llama --hf_token YOUR_TOKEN

Prerequisite: Clone FLARE repo first:
  git clone https://github.com/SusuXu-s-Lab/FLARE.git
"""

import pandas as pd
import numpy as np
import json, os, argparse, time
from datetime import datetime
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default='qwen', choices=['llama', 'qwen'])
parser.add_argument('--flare_dir', type=str, default='FLARE')
parser.add_argument('--output_dir', type=str, default='results_flare_original')
parser.add_argument('--hf_token', type=str, default='')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

MODEL_MAP = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}
MODEL_NAME = MODEL_MAP[args.backend]

print(f"\n{'='*60}")
print(f"FLARE Original Task: Model Comparison")
print(f"{'='*60}")
print(f"Testing: {args.backend} ({MODEL_NAME})")
print(f"Comparing against: GPT-4o (77.6%) and Claude 3.5 (89.5%)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")

# ============================================================
# LOAD MODEL
# ============================================================
print(f"\nLoading {MODEL_NAME}...")
from transformers import AutoTokenizer, AutoModelForCausalLM

load_kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto', 'trust_remote_code': True}
token = args.hf_token or os.environ.get('HF_TOKEN')
if token:
    load_kwargs['token'] = token

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=token)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Loaded! GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

def get_response(system, question, max_new_tokens=400):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7,
                            top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# ============================================================
# LOAD FLARE DATA
# ============================================================
print(f"\nLoading FLARE original data...")

# Load GPT-4o predictions (contains actual labels + model outputs)
gpt4o_file = os.path.join(args.flare_dir, 'reasoning_process_examples', 'alldata_finaprediction_4o.csv')
claude_file = os.path.join(args.flare_dir, 'reasoning_process_examples', 'alldata_finaprediction_claude.csv')

df_4o = pd.read_csv(gpt4o_file, encoding='latin')
df_claude = pd.read_csv(claude_file, encoding='latin')

actual_labels = df_4o['actual_values'].values
n_samples = len(actual_labels)

print(f"Samples: {n_samples}")
print(f"Actual distribution: evacuated={sum(actual_labels==1)}, stayed={sum(actual_labels==0)}")
print(f"GPT-4o accuracy: {(df_4o['predict_value'].values == actual_labels).mean():.4f}")
print(f"Claude accuracy:  {(df_claude['predict_value'].values == actual_labels).mean():.4f}")

# Load the GPT-4o reasoning to use as reference prompts
# The predict_res column contains the full reasoning chain output
gpt4o_reasoning = df_4o['predict_res'].values
gpt4o_threats = df_4o['threat'].values  # threat scores 1-5
gpt4o_risks = df_4o['risk'].values  # risk scores

# Load reflection base (memory examples from FLARE)
reflection_file = os.path.join(args.flare_dir, 'reasoning_process_examples', 'alldata_reflexion_base.json')
with open(reflection_file) as f:
    reflections = json.load(f)

print(f"Reflection examples available: {len(reflections)}")

# ============================================================
# FLARE PROMPTS (exact same as original paper)
# ============================================================
SYSTEM = 'You are an expert at rationale reasoning.'

THREAT_PROMPT = '''Analyze the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their responses to a wildfire survey, provide a brief summary of the resident's threat perception.

Rate their threat perception on a scale of 1 to 5 (1=very low threat, 5=very high threat).

{context}

End with: Threat Score: [1-5]'''

RISK_PROMPT = '''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Threat Perception and their responses to a wildfire survey, briefly summarize the resident's Risk Perception.

Threat Perception: {threat}

Rate their risk perception on a scale of 1 to 5 (1=very low risk, 5=very high risk).

End with: Risk Score: [1-5]'''

DECISION_PROMPT = '''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Risk Perception Summary, and other information, determine whether the resident will decide to evacuate.

Your response should provide reasoning based on the information provided, and end with the evacuation decision in the following format: Answer: YES or Answer: NO.

Risk Perception Summary:
{risk}

Threat Perception:
{threat}

Threat Score: {threat_score}/5
Risk Score: {risk_score}/5'''

DECISION_WITH_MEMORY = '''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire.

Learn from these previous examples where the model was wrong:
{memory}

Based on the Risk Perception Summary and other information, determine whether the resident will decide to evacuate.

End with: Answer: YES or Answer: NO.

Risk Perception Summary:
{risk}

Threat Perception:
{threat}

Threat Score: {threat_score}/5
Risk Score: {risk_score}/5'''

# ============================================================
# EXTRACT CONTEXT FROM GPT-4O OUTPUTS
# We use the GPT-4o reasoning as the "survey response context"
# since we don't have the raw survey data
# ============================================================
import re

def parse_decision(text):
    u = text.upper()
    if "ANSWER: YES" in u or "ANSWER:YES" in u: return 1
    if "ANSWER: NO" in u or "ANSWER:NO" in u: return 0
    lines = u.strip().split('\n')
    last = lines[-1] if lines else ""
    if "YES" in last: return 1
    if "NO" in last: return 0
    return 1 if u.count("YES") > u.count("NO") else 0

def extract_score(text, keyword="Score"):
    match = re.search(rf'{keyword}[:\s]*(\d)', text)
    if match:
        return int(match.group(1))
    return 3  # default

# ============================================================
# RUN PIPELINE
# ============================================================
print(f"\n{'='*60}")
print(f"RUNNING FLARE with {args.backend.upper()}")
print(f"{'='*60}")

train_size = int(n_samples * 0.7)
memory_items = []
MEMORY_K = 3

all_preds = []
all_threats = []
all_risks = []
all_outputs = []
train_preds, train_actuals = [], []
test_preds, test_actuals = [], []
errors = 0
start = time.time()

for idx in tqdm(range(n_samples), desc=f"FLARE ({args.backend})"):
    is_train = idx < train_size
    
    try:
        # Use GPT-4o's reasoning output as context
        # (since we don't have raw survey data, the GPT-4o reasoning
        # contains the survey Q&A information)
        context = str(gpt4o_reasoning[idx])[:800]
        
        # Step 1: Threat assessment
        threat_resp = get_response(SYSTEM, THREAT_PROMPT.format(context=context))
        threat_score = extract_score(threat_resp, "Threat")
        
        # Step 2: Risk assessment
        risk_resp = get_response(SYSTEM, RISK_PROMPT.format(threat=threat_resp[:400]))
        risk_score = extract_score(risk_resp, "Risk")
        
        # Step 3: Decision
        if memory_items and not is_train:
            memory_text = "\n".join(memory_items[-MEMORY_K:])
            dec = get_response(SYSTEM, DECISION_WITH_MEMORY.format(
                memory=memory_text, risk=risk_resp[:400], threat=threat_resp[:300],
                threat_score=threat_score, risk_score=risk_score))
        else:
            dec = get_response(SYSTEM, DECISION_PROMPT.format(
                risk=risk_resp[:400], threat=threat_resp[:300],
                threat_score=threat_score, risk_score=risk_score))
        
        pred = parse_decision(dec)
        correct = pred == actual_labels[idx]
        
        # Memory RL during training
        if not correct and is_train:
            errors += 1
            action = "evacuated" if actual_labels[idx] == 1 else "did NOT evacuate"
            memory_items.append(f"ERROR: Resident actually {action}. "
                              f"Threat was {threat_score}/5, Risk was {risk_score}/5. "
                              f"Context: {context[:200]}")
        
        all_preds.append(pred)
        all_threats.append(threat_score)
        all_risks.append(risk_score)
        all_outputs.append(dec[:200])
        
        if is_train:
            train_preds.append(pred)
            train_actuals.append(actual_labels[idx])
        else:
            test_preds.append(pred)
            test_actuals.append(actual_labels[idx])
        
        if idx % 10 == 0:
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"\n  Error at {idx}: {e}")
        all_preds.append(0)
        all_threats.append(3)
        all_risks.append(3)
        all_outputs.append(str(e))
        if is_train:
            train_preds.append(0)
            train_actuals.append(actual_labels[idx])
        else:
            test_preds.append(0)
            test_actuals.append(actual_labels[idx])

elapsed = time.time() - start

# ============================================================
# RESULTS
# ============================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report

print(f"\n{'='*60}")
print(f"RESULTS: FLARE Original Wildfire Task")
print(f"{'='*60}")
print(f"Time: {elapsed/60:.1f} minutes")

# Overall
overall_acc = accuracy_score(actual_labels[:len(all_preds)], all_preds)
print(f"\nOverall accuracy: {overall_acc:.4f} ({sum(np.array(all_preds)==actual_labels[:len(all_preds)])}/{len(all_preds)})")

# Train
if train_preds:
    tr_acc = accuracy_score(train_actuals, train_preds)
    print(f"Training accuracy: {tr_acc:.4f} | Errors stored: {errors}")

# Test
if test_preds:
    te_acc = accuracy_score(test_actuals, test_preds)
    te_f1 = f1_score(test_actuals, test_preds, average='weighted')
    print(f"Test accuracy: {te_acc:.4f} | F1: {te_f1:.4f}")
    print(classification_report(test_actuals, test_preds, 
                                target_names=['Stayed', 'Evacuated'], labels=[0, 1]))

# Comparison
print(f"\n{'='*60}")
print("MODEL COMPARISON ON FLARE WILDFIRE TASK")
print(f"{'='*60}")
print(f"  {'Model':<25s} {'Accuracy':>10s} {'Notes':>20s}")
print(f"  {'-'*57}")

gpt4o_acc = (df_4o['predict_value'].values == actual_labels).mean()
claude_acc = (df_claude['predict_value'].values == actual_labels).mean()

print(f"  {'Claude 3.5':<25s} {claude_acc:>9.4f} {'(from FLARE paper)':>20s}")
print(f"  {'GPT-4o':<25s} {gpt4o_acc:>9.4f} {'(from FLARE paper)':>20s}")
print(f"  {args.backend + ' (ours)':<25s} {overall_acc:>9.4f} {'(this experiment)':>20s}")

if test_preds:
    print(f"\n  Test-only comparison:")
    # GPT-4o test accuracy (last 30%)
    gpt4o_test = df_4o['predict_value'].values[train_size:]
    actual_test = actual_labels[train_size:]
    gpt4o_test_acc = accuracy_score(actual_test, gpt4o_test)
    claude_test = df_claude['predict_value'].values[train_size:]
    claude_test_acc = accuracy_score(actual_test, claude_test)
    
    print(f"  {'Claude 3.5 (test)':<25s} {claude_test_acc:>9.4f}")
    print(f"  {'GPT-4o (test)':<25s} {gpt4o_test_acc:>9.4f}")
    print(f"  {args.backend + ' (test)':<25s} {te_acc:>9.4f}")

# Performance drop
drop_from_gpt4o = gpt4o_acc - overall_acc
drop_from_claude = claude_acc - overall_acc
print(f"\n  Performance drop from GPT-4o: {drop_from_gpt4o:+.1%}")
print(f"  Performance drop from Claude: {drop_from_claude:+.1%}")

# ============================================================
# SAVE
# ============================================================
results = {
    'backend': args.backend,
    'model': MODEL_NAME,
    'task': 'FLARE_wildfire_original',
    'n_samples': n_samples,
    'train_size': train_size,
    'test_size': n_samples - train_size,
    'overall_accuracy': float(overall_acc),
    'train_accuracy': float(tr_acc) if train_preds else None,
    'test_accuracy': float(te_acc) if test_preds else None,
    'test_f1': float(te_f1) if test_preds else None,
    'errors_stored': errors,
    'time_minutes': elapsed / 60,
    'comparison': {
        'gpt4o_overall': float(gpt4o_acc),
        'claude_overall': float(claude_acc),
        'drop_from_gpt4o': float(drop_from_gpt4o),
        'drop_from_claude': float(drop_from_claude),
    }
}

with open(os.path.join(args.output_dir, f'flare_original_{args.backend}.json'), 'w') as f:
    json.dump(results, f, indent=2)

pd.DataFrame({
    'prediction': all_preds,
    'actual': list(actual_labels[:len(all_preds)]),
    'threat_score': all_threats,
    'risk_score': all_risks,
    'output': all_outputs,
}).to_csv(os.path.join(args.output_dir, f'flare_original_predictions_{args.backend}.csv'), index=False)

print(f"\nSaved: results_flare_original/flare_original_{args.backend}.json")
print("DONE!")
