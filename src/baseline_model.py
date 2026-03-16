"""
FLARE-VAX: Baseline Models on Full Dataset
============================================
Runs 7 ML baselines on the complete NHIS 2024 dataset (32,000+ samples)
No GPU needed — runs on CPU only.

Usage on Sol:
  python run_baselines_full.py --data_path nhis2024_with_patterns.csv

Or on your laptop:
  python run_baselines_full.py --data_path C:\Users\harul\Downloads\adult24csv\nhis2024_with_patterns.csv
"""

import pandas as pd
import numpy as np
import json, os, argparse, time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='nhis2024_with_patterns.csv')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--test_ratio', type=float, default=0.3)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"\n{'='*60}")
print("FLARE-VAX: Baseline Models on Full Dataset")
print(f"{'='*60}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# LOAD DATA
# ============================================================
print(f"\nLoading: {args.data_path}")
df = pd.read_csv(args.data_path)
print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# Replace NHIS missing codes
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].replace([7, 8, 9, 97, 98, 99], np.nan)

df_valid = df.dropna(subset=['vaccinated', 'age', 'sex', 'health_status']).copy()
print(f"Valid rows: {len(df_valid)}")
print(f"Vaccination rate: {df_valid['vaccinated'].mean():.1%}")

# ============================================================
# FEATURES
# ============================================================
feat = [c for c in [
    'age', 'sex', 'race', 'hispanic', 'education', 'marital_status', 'region',
    'income_poverty_ratio', 'health_status', 'diabetes', 'copd', 'cancer_ever',
    'heart_disease', 'hypertension', 'uninsured', 'medicare', 'medicaid',
    'cost_barrier_1', 'cost_barrier_2', 'usual_care_place', 'smoking_status',
    'bmi_category', 'high_risk_chronic', 'any_cost_barrier', 'access_barrier',
    'num_children', 'disability', 'angina', 'stroke', 'has_insurance'
] if c in df_valid.columns]

print(f"Features used: {len(feat)}")

X = df_valid[feat].fillna(0).values
y = df_valid['vaccinated'].values

# ============================================================
# STRATIFIED TRAIN-TEST SPLIT
# ============================================================
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_ratio, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} (vax: {y_train.mean():.1%})")
print(f"Test:  {len(X_test)} (vax: {y_test.mean():.1%})")

# Scale for models that need it
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# ============================================================
# BASELINE MODELS
# ============================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

baselines = {}

# 1. Random Forest
baselines['Random Forest'] = {
    'model': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    'scale': False
}

# 2. XGBoost
try:
    from xgboost import XGBClassifier
    baselines['XGBoost'] = {
        'model': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, eval_metric='logloss', n_jobs=-1),
        'scale': False
    }
except ImportError:
    print("XGBoost not installed, skipping")

# 3. Logistic Regression
baselines['Logistic Regression'] = {
    'model': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'scale': True
}

# 4. SVM (RBF)
baselines['SVM (RBF)'] = {
    'model': SVC(kernel='rbf', random_state=42, C=1.0, probability=True),
    'scale': True
}

# 5. Gradient Boosting
baselines['Gradient Boosting'] = {
    'model': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                        learning_rate=0.1, random_state=42),
    'scale': False
}

# 6. MLP Neural Network
baselines['MLP Neural Net'] = {
    'model': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          random_state=42, early_stopping=True),
    'scale': True
}

# 7. KNN
baselines['KNN (k=5)'] = {
    'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'scale': True
}

# ============================================================
# RUN ALL BASELINES
# ============================================================
print(f"\n{'='*60}")
print("RUNNING BASELINES")
print(f"{'='*60}")
print(f"\n  {'Method':<25s} {'Train Acc':>10s} {'Test Acc':>10s} {'F1':>8s} {'AUC':>8s} {'Time':>8s}")
print(f"  {'-'*73}")

all_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, cfg in baselines.items():
    start = time.time()
    try:
        m = cfg['model']
        use_scaled = cfg['scale']

        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc if use_scaled else X_test

        # Train
        m.fit(Xtr, y_train)

        # Predict
        train_pred = m.predict(Xtr)
        test_pred = m.predict(Xte)

        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        test_prec = precision_score(y_test, test_pred, average='weighted')
        test_rec = recall_score(y_test, test_pred, average='weighted')

        # AUC
        try:
            if hasattr(m, 'predict_proba'):
                test_proba = m.predict_proba(Xte)[:, 1]
            else:
                test_proba = m.decision_function(Xte)
            test_auc = roc_auc_score(y_test, test_proba)
        except:
            test_auc = 0.0

        # Cross-validation on full data
        Xcv = scaler.transform(X) if use_scaled else X
        cv_scores = cross_val_score(clone(cfg['model']), Xcv, y, cv=cv, scoring='accuracy', n_jobs=-1)

        elapsed = time.time() - start

        all_results[name] = {
            'train_accuracy': round(float(train_acc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'test_f1': round(float(test_f1), 4),
            'test_precision': round(float(test_prec), 4),
            'test_recall': round(float(test_rec), 4),
            'test_auc': round(float(test_auc), 4),
            'cv_accuracy_mean': round(float(cv_scores.mean()), 4),
            'cv_accuracy_std': round(float(cv_scores.std()), 4),
            'time_seconds': round(elapsed, 1),
        }

        print(f"  {name:<25s} {train_acc:>9.4f} {test_acc:>9.4f} {test_f1:>7.4f} {test_auc:>7.4f} {elapsed:>6.1f}s")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  {name:<25s} FAILED: {e} ({elapsed:.1f}s)")

# ============================================================
# DETAILED CLASSIFICATION REPORTS
# ============================================================
print(f"\n{'='*60}")
print("DETAILED CLASSIFICATION REPORTS")
print(f"{'='*60}")

for name, cfg in baselines.items():
    if name not in all_results:
        continue
    m = cfg['model']
    Xte = X_test_sc if cfg['scale'] else X_test
    test_pred = m.predict(Xte)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, test_pred, target_names=['Not Vaccinated', 'Vaccinated'], labels=[0, 1]))

# ============================================================
# FEATURE IMPORTANCE (from Random Forest)
# ============================================================
print(f"\n{'='*60}")
print("TOP 15 FEATURES (Random Forest Importance)")
print(f"{'='*60}")

rf = baselines['Random Forest']['model']
imp = pd.DataFrame({'feature': feat, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for i, row in imp.head(15).iterrows():
    print(f"  {row['feature']:<30s} {row['importance']:.4f}")

imp.to_csv(os.path.join(args.output_dir, 'feature_importance_full.csv'), index=False)

# ============================================================
# SUMMARY TABLE
# ============================================================
print(f"\n{'='*60}")
print("FINAL SUMMARY TABLE")
print(f"{'='*60}")
print(f"\n  {'Method':<25s} {'CV Acc':>8s} {'Test Acc':>10s} {'F1':>8s} {'AUC':>8s}")
print(f"  {'-'*61}")

for name in sorted(all_results.keys(), key=lambda x: all_results[x]['test_accuracy'], reverse=True):
    r = all_results[name]
    print(f"  {name:<25s} {r['cv_accuracy_mean']:>7.4f} {r['test_accuracy']:>9.4f} {r['test_f1']:>7.4f} {r['test_auc']:>7.4f}")

print(f"\n  Dataset size: {len(df_valid)}")
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Features: {len(feat)}")

# ============================================================
# SAVE
# ============================================================
output_file = os.path.join(args.output_dir, 'baselines_full_dataset.json')
save_data = {
    'dataset': {
        'total_samples': len(df_valid),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': len(feat),
        'feature_names': feat,
        'vaccination_rate': round(float(df_valid['vaccinated'].mean()), 4),
    },
    'results': all_results,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

with open(output_file, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved: {output_file}")
print(f"Saved: {os.path.join(args.output_dir, 'feature_importance_full.csv')}")
print("\nDONE!")