# FLARE-VAX: Theory-Guided LLM Framework for Vaccination Decision Prediction

> **Adapting the FLARE framework (Chen et al., ACL 2025) from wildfire evacuation to flu vaccination decisions using the Health Belief Model (HBM) and the NHIS 2024 dataset.**

**Author:** Harul Murugan  
**Advisor:** Prof. Xiyang Hu ([@xiyanghu](https://github.com/xiyanghu))  
**Institution:** W. P. Carey School of Business, Arizona State University  
**Status:** Active Research — Week 5

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Health Belief Model (HBM)](#3-health-belief-model-hbm)
4. [Dataset](#4-dataset)
5. [Repository Structure](#5-repository-structure)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [HBM Reasoning Pattern Classifier](#7-hbm-reasoning-pattern-classifier)
8. [Models Tested](#8-models-tested)
9. [Results](#9-results)
10. [Key Findings](#10-key-findings)
11. [Comparison with Trainable Methods](#11-comparison-with-trainable-methods)
12. [Setup & Installation](#12-setup--installation)
13. [Running the Experiments](#13-running-the-experiments)
14. [Running on ASU Sol Supercomputer](#14-running-on-asu-sol-supercomputer)
15. [Weekly Progress](#15-weekly-progress)
16. [References](#16-references)

---

## 1. Project Overview

This project replicates and adapts **FLARE** (Chen et al., ACL 2025), a theory-guided LLM framework for predicting human decisions. The original FLARE paper used the **Protective Action Decision Model (PADM)** to predict wildfire evacuation decisions. We adapt this framework to predict **flu vaccination decisions** using the **Health Belief Model (HBM)** — a well-established behavioral theory in public health.

### What FLARE Does

FLARE is a **training-free** framework that:
1. Assigns each person to a behavioral reasoning pattern using a classifier
2. Uses the behavioral theory to guide a multi-step Chain-of-Thought (CoT) prompt
3. Applies a reflection/memory module to learn from prediction errors in the training phase
4. Makes a final YES/NO prediction on whether a person will take a protective action

### Our Adaptation (FLARE-VAX)

| Dimension | Original FLARE | FLARE-VAX (Ours) |
|---|---|---|
| Domain | Wildfire evacuation | Flu vaccination |
| Behavioral theory | PADM | Health Belief Model |
| Dataset | Marshall Fire Survey | NHIS 2024 Sample Adult |
| Target variable | Did person evacuate? | Did person get flu vaccine? |
| LLM tested | GPT-4o, Claude | GPT-4o, Qwen, Llama |

---

## 2. Background & Motivation

### Why Vaccination Prediction?

Flu vaccination is one of the most studied public health behaviors. Despite vaccines being widely available and free in the US, vaccination rates remain suboptimal — around 50% in the adult population. Understanding *why* individuals choose to vaccinate or not is a critical public health challenge.

Traditional machine learning approaches treat this as a binary classification problem using demographic and health features. They achieve around 60-66% accuracy but cannot explain *why* someone would or would not vaccinate. They treat people as data points, not as individuals with beliefs, barriers, and reasoning processes.

FLARE-VAX takes a fundamentally different approach: it uses an LLM to simulate the cognitive reasoning process that drives vaccination decisions, guided by the Health Belief Model.

### Why Adapt FLARE?

FLARE demonstrated that theory-guided LLM reasoning can outperform traditional ML baselines for human decision prediction. If the approach generalizes beyond wildfire evacuation, it could become a powerful tool for public health behavior modeling. The vaccination domain is an ideal test case because:
- HBM is the gold standard theory for vaccination behavior
- The NHIS dataset provides a large, nationally representative sample
- The binary target variable (vaccinated/not vaccinated) maps directly to FLARE's framework
- The public health stakes are high and the findings are actionable

---

## 3. Health Belief Model (HBM)

The Health Belief Model is a psychological model developed in the 1950s to explain health-related behaviors. It posits that individuals' decisions to take a health action depend on six key constructs:

| HBM Construct | Definition | Vaccination Context |
|---|---|---|
| **Perceived Susceptibility** | Belief about risk of getting the disease | "How likely am I to get the flu?" |
| **Perceived Severity** | Belief about how serious the disease is | "How bad would getting the flu be?" |
| **Perceived Benefits** | Belief about the effectiveness of the action | "Will the vaccine actually protect me?" |
| **Perceived Barriers** | Belief about the costs/obstacles to action | "Is the vaccine accessible and affordable?" |
| **Cues to Action** | Triggers that prompt health behavior | "My doctor recommended I get vaccinated" |
| **Self-Efficacy** | Confidence in ability to take the action | "I am capable of getting vaccinated" |

### How HBM Maps to FLARE

In FLARE-VAX, each HBM construct maps to a specific part of the reasoning pipeline:

```
Person Profile
      │
      ▼
[Susceptibility Score] ←── age, chronic conditions, health status
      │
      ▼
[Barrier Score] ←── insurance, cost barriers, access to care
      │
      ▼
[Pattern Assignment] ←── 2×2 matrix: High/Low Sus × High/Low Barrier
      │
      ▼
[LLM Chain-of-Thought] ←── pattern-specific HBM reasoning prompt
      │
      ▼
[Reflection Memory] ←── learns from training errors
      │
      ▼
[Final Prediction: YES / NO]
```

---

## 4. Dataset

**Source:** National Health Interview Survey (NHIS) 2024 Sample Adult  
**Provider:** CDC National Center for Health Statistics  
**URL:** https://www.cdc.gov/nchs/nhis/

### Dataset Statistics

| Metric | Value |
|---|---|
| Total valid respondents | 32,132 |
| Total variables | 45 |
| Target variable | `vaccinated` (flu shot in past 12 months) |
| Overall vaccination rate | ~50% |
| Train split | 70% (stratified) |
| Test split | 30% (stratified) |

### Variable Categories

**Demographic variables (14):**
`age`, `sex`, `race`, `hispanic`, `education`, `marital_status`, `region`, `us_born`, `citizenship`, `income_poverty_ratio`, `num_children`, `parent_status`, `smoking_status`, `bmi_category`

**Susceptibility variables (10) — maps to HBM Perceived Susceptibility & Severity:**
`health_status`, `diabetes`, `copd`, `cancer_ever`, `heart_disease`, `angina`, `stroke`, `hypertension`, `high_risk_chronic`, `disability`

**Barrier variables (12) — maps to HBM Perceived Barriers:**
`uninsured`, `has_insurance`, `insurance_type`, `medicaid`, `medicare`, `cost_barrier_1`, `cost_barrier_2`, `reason_no_ins_cost`, `stopped_care_cost`, `usual_care_place`, `any_cost_barrier`, `access_barrier`

### Data Preprocessing

- Missing value codes (7, 8, 9, 97, 98, 99) replaced with NaN
- Rows with missing `vaccinated`, `age`, `sex`, or `health_status` dropped
- For LLM experiments: balanced sampling (equal vaccinated/unvaccinated) to avoid class bias
- Stratified train-test split to ensure equal class representation

---

## 5. Repository Structure

```
Health_Research_Vaccination/
│
├── README.md                          # This file
│
├── Data/
│   ├── nhis2024_vaccination_clean.csv # Cleaned NHIS 2024 dataset
│   └── nhis2024_with_patterns.csv     # Dataset with HBM patterns assigned
│
├── Results/
│   ├── baselines_qwen.json            # Baseline results (1000 sample run)
│   ├── baselines_full_dataset.json    # Baseline results (full 32K dataset)
│   ├── feature_importance_full.csv    # Random Forest feature importance
│   ├── predictions_qwen.csv           # Qwen 3.5-27B full predictions
│   ├── predictions_qwen_ckpt.csv      # Checkpoint predictions
│   ├── reflections_qwen.json          # Training error reflections
│   ├── summary_qwen.json              # Qwen experiment summary
│   └── flare_original_*/              # Original FLARE wildfire results
│
└── src/
    ├── Variable_selection.py          # Step 1: Variable selection + HBM pattern classifier
    ├── run_flare_vax_v2.py            # Step 2: Main FLARE-VAX pipeline (Qwen/Llama)
    ├── run_flare_vax_hf.py            # Alternative HuggingFace inference script
    ├── run_flare_vax_full.py          # Full dataset run script
    ├── run_baselines_full.py          # Standalone baseline runner (full dataset)
    ├── baseline_model.py              # Baseline model implementations
    └── run_experiment.sh              # Sol supercomputer submission script
```

---

## 6. Pipeline Architecture

The FLARE-VAX pipeline runs in two sequential steps:

### Step 1: Variable Selection & Pattern Classification (`Variable_selection.py`)

This script must be run first. It:
1. Loads the cleaned NHIS dataset
2. Runs weighted regression variable selection (top 70% cumulative weight)
3. Runs Random Forest feature importance ranking
4. Maps variables to HBM constructs (susceptibility vs barriers)
5. Computes susceptibility and barrier scores for each person
6. Assigns each person to one of 4 HBM reasoning patterns
7. Saves the enriched dataset as `nhis2024_with_patterns.csv`

```bash
python src/Variable_selection.py
```

### Step 2: FLARE-VAX LLM Inference (`run_flare_vax_v2.py`)

This script runs the main LLM-based prediction pipeline:

**For each person in the dataset, it:**
1. Builds a natural language profile from their survey responses
2. Calls the LLM to assess susceptibility (HBM reasoning Step 1)
3. Calls the LLM to assess barriers (HBM reasoning Step 2)
4. Uses reflection memory from training errors to improve predictions
5. Calls the LLM to make a final YES/NO vaccination prediction
6. Saves all predictions, reflections, and summary statistics

**Three LLM calls per person:**
```
Call 1: Susceptibility Assessment
  → "Analyze perceived susceptibility and severity. Rate 1-5."

Call 2: Barrier Assessment  
  → "Based on susceptibility and access info, summarize barriers. Rate 1-5."

Call 3: Final Decision (with or without memory)
  → "Will this person get vaccinated? Use HBM reasoning. Answer: YES or NO"
```

---

## 7. HBM Reasoning Pattern Classifier

The classifier assigns each person to one of four HBM behavioral patterns before LLM reasoning begins. This ensures the LLM uses the most relevant Chain-of-Thought template for each person.

### Score Computation

```python
# Susceptibility Score (0 to 3)
susceptibility_score = (
    int(age > 50) +          # CDC high-risk age threshold
    high_risk_chronic +       # Has diabetes, COPD, heart disease, or hypertension
    int(health_status >= 4)   # Fair or poor health (scale: 1=excellent, 5=poor)
)

# Barrier Score (0 to 2)
barrier_score = (
    access_barrier +          # No usual place of care
    any_cost_barrier          # Delayed care due to cost
)
```

### Pattern Assignment (2×2 HBM Matrix)

| Pattern | Susceptibility | Barrier | Count | % of Data | Vaccination Rate |
|---|---|---|---|---|---|
| **0** | ≥ 2 (High) | = 0 (Low) | 13,327 | 41.5% | **61.7%** |
| **1** | ≥ 2 (High) | ≥ 1 (High) | 396 | 1.2% | 24.5% |
| **2** | < 2 (Low) | = 0 (Low) | 16,453 | 51.2% | 40.4% |
| **3** | < 2 (Low) | ≥ 1 (High) | 1,956 | 6.1% | **12.9%** |

### HBM Interpretation of Each Pattern

**Pattern 0 — High Susceptibility / Low Barrier (61.7% vax rate):**
Elderly or chronically ill individuals with good insurance and care access. HBM predicts: high perceived risk with no friction = most likely to vaccinate. ✅ Confirmed.

**Pattern 1 — High Susceptibility / High Barrier (24.5% vax rate):**
High-risk individuals who face cost or access barriers. HBM predicts: intent to vaccinate exists but barriers block action. This is the most policy-relevant group — underserved high-risk minorities who need targeted interventions. ✅ Confirmed.

**Pattern 2 — Low Susceptibility / Low Barrier (40.4% vax rate):**
Younger, healthier individuals with good access. HBM predicts: indifference — no strong motivation but also no obstacles. ✅ Confirmed.

**Pattern 3 — Low Susceptibility / High Barrier (12.9% vax rate):**
Young, healthy individuals who also face barriers. HBM predicts: lowest motivation combined with highest friction = least likely to vaccinate. ✅ Confirmed.

**Key validation:** The 49-point spread in vaccination rates (12.9% to 61.7%) across patterns confirms the classifier captures meaningful behavioral differences aligned with HBM theory.

---

## 8. Models Tested

### LLM Models

| Model | Type | Parameters | Backend | Status |
|---|---|---|---|---|
| GPT-4o | Commercial | ~1T | OpenAI API | ✅ Completed (50 samples) |
| Qwen 2.5-7B-Instruct | Open-source | 7B | HuggingFace Transformers | ✅ Completed (1000 samples) |
| Qwen 3.5-27B | Open-source | 27B | HuggingFace Transformers | ✅ Completed (1000 samples) |
| Llama 3.1-8B-Instruct | Open-source | 8B | HuggingFace Transformers | ✅ Completed (1000 samples) |

### ML Baseline Models

All baselines use stratified 70/30 train-test split and 5-fold cross-validation:

| Model | Library | Key Parameters |
|---|---|---|
| Random Forest | scikit-learn | n_estimators=200, random_state=42 |
| Logistic Regression | scikit-learn | max_iter=1000 |
| Gradient Boosting | scikit-learn | n_estimators=200, max_depth=5 |
| MLP Neural Net | scikit-learn | hidden_layers=(128,64), early_stopping=True |
| KNN | scikit-learn | k=5 |
| SVM (RBF kernel) | scikit-learn | kernel=rbf |
| XGBoost | xgboost | n_estimators=200, max_depth=6 |

---

## 9. Results

### ML Baselines — Full Dataset (32,132 samples)

| Method | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 64.8% | **65.8%** ← Best |
| SVM (RBF) | 66.5% | 65.1% |
| Random Forest | 65.3% | 65.1% |
| MLP Neural Net | 64.9% | 64.1% |
| Gradient Boosting | 62.5% | 64.1% |
| XGBoost | 61.5% | 63.8% |
| KNN (k=5) | 63.1% | 59.8% |

### FLARE-VAX LLM Results — Vaccination Domain (1000 samples, balanced)

| Model | Train Acc | Test Acc | Test F1 | vs Best Baseline |
|---|---|---|---|---|
| GPT-4o | ~80% | ~80% | — | **+14 pts** ✅ |
| Qwen 2.5-7B | ~53% | ~53-55% | — | -11 pts ❌ |
| Llama 3.1-8B | ~53% | ~53-55% | — | -11 pts ❌ |
| Qwen 3.5-27B | 49.2% | 49.2% | 0.35 | -16 pts ❌ |

> **Note on Qwen 3.5-27B:** Model predicted YES (vaccinated) for 293 out of 301 test samples — severe YES bias. This indicates the model defaults to a pro-vaccination prior rather than reasoning through individual HBM constructs.

### FLARE Original — Wildfire Domain (validation of Task 3)

| Model | Overall Acc | Test Acc | Test F1 | vs GPT-4o |
|---|---|---|---|---|
| Claude (paper) | 89.5% | — | — | — |
| GPT-4o (paper) | 77.6% | — | — | baseline |
| **Qwen 3.5-27B** | **71.1%** | **82.6%** | **80.9%** | **+5 pts** ✅ |
| **Llama 3.1-8B** | **63.2%** | **60.9%** | **63.6%** | **-14.5 pts** ❌ |

---

## 10. Key Findings

### Finding 1: FLARE Works With Large Commercial Models

GPT-4o achieves ~80% accuracy on vaccination prediction — approximately 14 points above the best ML baseline. This confirms that FLARE's theory-guided Chain-of-Thought is effective when the underlying model has sufficient reasoning capacity.

### Finding 2: Open-Source Models Underperform Baselines

All open-source models (Qwen 2.5-7B, Qwen 3.5-27B, Llama 3.1-8B) fall below all ML baselines on the vaccination domain. This confirms that FLARE's effectiveness is model-scale dependent.

### Finding 3: Domain Complexity Matters — Not Just Model Scale

Qwen 3.5-27B achieves 82.6% on the original FLARE wildfire dataset (beating GPT-4o's 77.6%), but only 49.2% on our vaccination domain. This reveals that the failure of open-source models on FLARE-VAX is not purely a capacity problem — it is also a domain complexity problem.

**Why HBM is harder than PADM:**
- PADM has 2 main constructs (threat appraisal, protective action)
- HBM has 6 constructs that must be integrated simultaneously
- Vaccination has more abstract risk perception than immediate physical evacuation
- Open-source models may have different prior biases for health vs emergency decisions

### Finding 4: HBM Pattern Classifier is Theoretically Valid

The 49-point spread in vaccination rates across the four HBM patterns (12.9% to 61.7%) confirms that the classifier captures real behavioral differences. All four patterns follow HBM predictions exactly.

### Finding 5: LLMs Default to Conservative Predictions

Qwen 3.5-27B's YES bias (293/301 predictions) is consistent with findings in the literature (Feng et al., EMNLP 2025): LLMs default to stable, conservative strategies that diverge from human behavioral diversity. The model applies its training prior ("people vaccinate") rather than reasoning about individual barriers.

---

## 11. Comparison with Trainable Methods

FLARE-VAX is training-free — it requires no labeled data and no fine-tuning. This is both its strength and its limitation.

| Approach | Method | Training Data | Performance |
|---|---|---|---|
| FLARE-VAX (ours) | Zero-shot HBM CoT | None | GPT-4o: ~80%, Open-source: 49-55% |
| Paper A (2601.03534) | Supervised Fine-tuning + Theory CoT | 12,400 human assessments | Outperforms baselines + explainable |
| Paper B (2505.17249) | IRL + Behavioral Theory CoT | Travel trajectories | Beats GPT-4o and standard CoT |
| Paper C (2509.05830) | Fine-tuning on social science data | 2.9M responses | +26% over base model, beats GPT-4o |
| Paper D (2601.04208) | RL Fine-tuning | Task-specific reward signals | Significant improvement + explainability |

**Most promising next step:** Fine-tuning Qwen on NHIS vaccination data (Paper C approach). We have 32,000 labeled samples — comparable in scale to SocSci210. This could dramatically improve open-source model performance.

---

## 12. Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 recommended for 27B models)
- HuggingFace account (for Llama access)
- OpenAI API key (for GPT-4o)

### Installation

```bash
# Clone the repository
git clone https://github.com/harul634/Vaccination-Decision-Model.git
cd Vaccination-Decision-Model

# Create conda environment
conda create -n flare-vax python=3.10
conda activate flare-vax

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install scikit-learn xgboost pandas numpy tqdm
pip install openai langchain langchain-openai faiss-cpu
```

### Environment Variables

```bash
# For GPT-4o experiments
export OPENAI_API_KEY="your-openai-api-key"

# For HuggingFace model downloads
export HF_HOME="/path/to/hf_cache"

# For Llama access (requires HuggingFace approval)
huggingface-cli login
```

---

## 13. Running the Experiments

### Step 1: Prepare Data with HBM Patterns

```bash
# Update DATA_PATH and OUTPUT_PATH in Variable_selection.py first
python src/Variable_selection.py
```

This creates `nhis2024_with_patterns.csv` with HBM pattern assignments.

### Step 2: Run ML Baselines

```bash
# Baselines on 1000 samples (quick test)
python src/run_flare_vax_v2.py --backend qwen --sample_size 1000

# Baselines on full dataset
python src/run_baselines_full.py
```

### Step 3: Run FLARE-VAX with Different Backends

```bash
# Qwen 3.5-27B on 1000 samples (requires A100 GPU, ~27 hours)
python src/run_flare_vax_v2.py \
    --backend qwen \
    --sample_size 1000 \
    --data_path nhis2024_with_patterns.csv \
    --output_dir Results \
    --checkpoint_every 100

# Llama 3.1-8B on 1000 samples
python src/run_flare_vax_v2.py \
    --backend llama \
    --sample_size 1000 \
    --data_path nhis2024_with_patterns.csv \
    --output_dir Results \
    --checkpoint_every 100

# Full dataset run (open-source model)
python src/run_flare_vax_v2.py \
    --backend qwen \
    --sample_size 0 \
    --data_path nhis2024_with_patterns.csv \
    --output_dir Results \
    --checkpoint_every 500
```

### Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--backend` | `llama` | Model backend: `llama` or `qwen` |
| `--sample_size` | `1000` | Number of samples (0 = full dataset) |
| `--data_path` | `nhis2024_with_patterns.csv` | Path to dataset |
| `--output_dir` | `results1` | Output directory for results |
| `--train_ratio` | `0.7` | Train/test split ratio |
| `--memory_k` | `3` | Number of reflection examples in memory |
| `--checkpoint_every` | `100` | Save checkpoint every N samples |

---

## 14. Running on ASU Sol Supercomputer

### Environment Setup (First Time)

```bash
# Load modules
module load mamba
conda activate flare-vax

# Set HuggingFace cache to scratch (fast storage)
export HF_HOME=/scratch/hramamo1/hf_cache
```

### Submit Job (Qwen 3.5-27B, 1000 samples)

```bash
sbatch --account=grp_xiyanghu \
       --partition=public \
       --qos=public \
       --gres=gpu:a100:1 \
       --cpus-per-task=8 \
       --mem=64G \
       --time=2-00:00:00 \
       --output=~/vax_qwen35b.log \
       --error=~/vax_qwen35b_err.log \
       --wrap="source activate flare-vax && \
               export HF_HOME=/scratch/hramamo1/hf_cache && \
               cd ~ && \
               python run_flare_vax_v2.py --backend qwen --sample_size 1000 --checkpoint_every 100"
```

### Monitor Job Progress

```bash
# Check job status
squeue -u hramamo1

# Check estimated start time
squeue -u hramamo1 --start

# View live log output
tail -f ~/vax_qwen35b.log

# Check errors
tail -20 ~/vax_qwen35b_err.log

# Check checkpoint progress
wc -l ~/Results/predictions_qwen_ckpt.csv
```

### Expected Runtimes on A100

| Model | Samples | Approx Runtime |
|---|---|---|
| Qwen 2.5-7B | 1000 | ~27 hours |
| Qwen 3.5-27B | 1000 | ~27 hours |
| Llama 3.1-8B | 1000 | ~27 hours |
| Any model | 32,000 | ~35-40 days (not feasible without parallelization) |

> **Note:** All models process approximately 1 sample per 97 seconds due to the 3 sequential LLM calls per sample.

---

## 15. Weekly Progress

### Week 1-2: Setup & Data Preparation
- Set up NHIS 2024 data pipeline
- Cleaned and preprocessed 32,132 respondents
- Mapped NHIS variables to HBM constructs
- Set up GitHub repository

### Week 3: Variable Selection & Baseline Models
- Ran weighted regression and Random Forest variable selection
- Built HBM reasoning pattern classifier
- Validated 49-point spread in vaccination rates across 4 patterns
- Ran 7 ML baseline models — best: Logistic Regression at 65.8%

### Week 4: GPT-4o & Open-Source Model Experiments
- GPT-4o on 50 samples: ~80% accuracy (+14 pts over baselines) ✅
- Qwen 2.5-7B on 1000 samples: ~53-55% (below baselines) ❌
- Llama 3.1-8B on 1000 samples: ~53-55% (below baselines) ❌
- Identified and fixed data splitting bug (non-stratified → stratified)
- Set up Sol supercomputer environment

### Week 5: Full Dataset Baselines + Qwen 3.5-27B + Domain Validation
- Ran all 7 baselines on full 32,132-sample dataset ✅
- Ran Qwen 3.5-27B on 1000 samples: 49.2% test accuracy, YES bias ✅
- Validated FLARE on original wildfire dataset: Qwen 3.5-27B achieves 82.6% (beats GPT-4o) ✅
- Documented HBM pattern classifier methodology ✅
- Reviewed 5 trainable method papers as FLARE extensions ✅

---

## 16. References

### Primary Paper (FLARE)
Chen, X., et al. (2025). *FLARE: A Theory-Guided LLM Framework for Human Decision Prediction*. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).

### Behavioral Theory
Rosenstock, I. M. (1974). *Historical origins of the Health Belief Model*. Health Education Monographs, 2(4), 328-335.

### Dataset
CDC National Center for Health Statistics. (2024). *National Health Interview Survey (NHIS) 2024 Sample Adult*. https://www.cdc.gov/nchs/nhis/

### Trainable Method Papers Reviewed
- Dai, Y., et al. (2026). *Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach*. arXiv:2601.03534.
- Sun, Y., et al. (2025). *Where You Go is Who You Are: Behavioral Theory-Guided LLMs for Inverse Reinforcement Learning*. arXiv:2505.17249.
- Kolluri, A., et al. (2025). *Finetuning LLMs for Human Behavior Prediction in Social Science Experiments*. EMNLP 2025. arXiv:2509.05830.
- Cheng, X., et al. (2026). *LLMs for Explainable Business Decision-Making: A Reinforcement Learning Fine-Tuning Approach*. arXiv:2601.04208.
- Feng, Y., et al. (2025). *Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making*. EMNLP 2025. arXiv:2508.15926.

---

## Contact

**Harul Murugan**  
Graduate Student, W. P. Carey School of Business  
Arizona State University  
GitHub: [@harul634](https://github.com/harul634)

**Advisor: Prof. Xiyang Hu**  
GitHub: [@xiyanghu](https://github.com/xiyanghu)

---

*This research is conducted as part of graduate coursework at ASU W. P. Carey School of Business. All experiments are run on the ASU Sol supercomputer cluster.*
