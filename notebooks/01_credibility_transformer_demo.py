# Databricks notebook source
# MAGIC %md
# MAGIC # Credibility Transformer: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the `insurance-credibility-transformer` library on synthetic
# MAGIC French MTPL-like data. It covers:
# MAGIC 1. Synthetic data generation mimicking the paper's setup
# MAGIC 2. Base Credibility Transformer (1,746 parameters, CPU-trainable)
# MAGIC 3. Deep CT with PLE and SwiGLU (GPU-recommended)
# MAGIC 4. ICL-CT (In-Context Learning extension)
# MAGIC 5. Explainability: attention-as-credibility
# MAGIC 6. Benchmarking against null model and GLM
# MAGIC
# MAGIC **Paper**: Richman, Scognamiglio & Wüthrich (2024). arXiv:2409.16653
# MAGIC **ICL extension**: Padayachy et al. (2026). arXiv:2509.08122

# COMMAND ----------

# MAGIC %pip install insurance-credibility-transformer polars

# COMMAND ----------

import numpy as np
import polars as pl
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# COMMAND ----------

# MAGIC %md ## 1. Synthetic Data Generation
# MAGIC
# MAGIC We generate data that mimics the French MTPL dataset structure:
# MAGIC - 4 categorical features (vehicle class, fuel type, region, driver age band)
# MAGIC - 5 continuous features (bonus-malus, vehicle age, density, etc.)
# MAGIC - Poisson claim counts with exposure

# COMMAND ----------

from insurance_credibility_transformer import InsuranceDataset

np.random.seed(42)
N = 10_000

# Categorical features (cardinalities matching French MTPL)
veh_class = np.random.randint(0, 6, N)    # 6 vehicle classes
fuel_type = np.random.randint(0, 2, N)    # diesel/petrol
region = np.random.randint(0, 11, N)      # 11 regions
age_band = np.random.randint(0, 10, N)    # 10 age bands

x_cat = np.column_stack([veh_class, fuel_type, region, age_band]).astype(np.int64)

# Continuous features
bonus_malus = np.random.uniform(50, 350, N).astype(np.float32)
veh_age = np.random.uniform(0, 20, N).astype(np.float32)
density = np.exp(np.random.uniform(0, 10, N)).astype(np.float32)
driver_age = np.random.uniform(18, 85, N).astype(np.float32)
power = np.random.uniform(30, 200, N).astype(np.float32)

x_num = np.column_stack([
    bonus_malus / 100,
    np.log1p(veh_age),
    np.log1p(density / 1000),
    driver_age / 50,
    power / 100,
]).astype(np.float32)

# Exposure (policy years)
exposure = np.random.uniform(0.1, 1.0, N).astype(np.float32)

# True claim rate (Poisson DGP)
log_rate = (
    -2.5
    + 0.1 * (veh_class - 2)
    - 0.2 * fuel_type
    + 0.05 * region
    - 0.3 * np.log(bonus_malus / 100 + 1)
    + 0.1 * np.sin(driver_age / 20)
)
rate = np.exp(log_rate)
y = np.random.poisson(rate * exposure).astype(np.float32)

# Train/test split (80/20)
split = int(0.8 * N)
train_slice = slice(None, split)
test_slice = slice(split, None)

x_cat_train, x_cat_test = x_cat[train_slice], x_cat[test_slice]
x_num_train, x_num_test = x_num[train_slice], x_num[test_slice]
y_train, y_test = y[train_slice], y[test_slice]
exp_train, exp_test = exposure[train_slice], exposure[test_slice]

print(f"Training: {len(y_train):,} policies, {y_train.sum():.0f} claims")
print(f"Test:     {len(y_test):,} policies, {y_test.sum():.0f} claims")
print(f"Average claim frequency: {(y_train / exp_train).mean():.4f}")

# COMMAND ----------

# MAGIC %md ## 2. Baseline: Null Model and GLM
# MAGIC
# MAGIC The Poisson deviance of the null model is our benchmark. It equals the deviance of
# MAGIC predicting the portfolio mean frequency for every policy.

# COMMAND ----------

from insurance_credibility_transformer import PoissonDevianceLoss

def poisson_deviance(y_true, y_pred, exposure):
    """Compute out-of-sample Poisson deviance × 10^-2 (paper metric)."""
    loss_fn = PoissonDevianceLoss(use_exposure=True)
    y_t = torch.tensor(y_true, dtype=torch.float32)
    exp_t = torch.tensor(exposure, dtype=torch.float32)
    pred_t = torch.tensor(y_pred, dtype=torch.float32)
    with torch.no_grad():
        d = loss_fn(pred_t, y_t, exp_t)
    return float(d) * 100  # × 10^-2

# Null model: portfolio mean rate × exposure
mean_rate_train = y_train.sum() / exp_train.sum()
null_preds = mean_rate_train * exp_test
null_deviance = poisson_deviance(y_test, null_preds, exp_test)
print(f"Null model deviance × 10^-2: {null_deviance:.4f}")

# COMMAND ----------

# MAGIC %md ## 3. Base Credibility Transformer
# MAGIC
# MAGIC The base CT has ~1,746 parameters for the paper's setup (b=5, 1 head, 1 layer).
# MAGIC It trains on CPU in a few minutes. For this demo we use a slightly larger model.

# COMMAND ----------

from insurance_credibility_transformer import (
    CredibilityTransformer,
    CredibilityTransformerTrainer,
)

# Base CT: matches the paper's architecture
ct = CredibilityTransformer(
    cat_cardinalities=[6, 2, 11, 10],   # vehicle class, fuel, region, age band
    n_num_features=5,
    embed_dim=5,           # b = 5 (paper default)
    n_heads=1,             # M = 1 (base CT)
    n_layers=1,            # L = 1 (base CT)
    alpha=0.90,            # credibility parameter
    dropout=0.01,
    link="log",
)

print(f"Base CT parameters: {ct.count_parameters():,}")

trainer = CredibilityTransformerTrainer(
    model=ct,
    loss="poisson",
    lr=1e-3,
    weight_decay=1e-2,
    beta2=0.95,
    batch_size=1024,
    val_split=0.15,
    early_stopping_patience=20,
    max_epochs=100,
    n_ensemble=3,           # Paper uses 20; use 3 for demo speed
    random_seed=42,
    verbose=10,
    device=device,
)

print("Training base CT...")
trainer.fit(x_cat_train, x_num_train, y_train, exp_train)

# COMMAND ----------

# Evaluate
ct_preds = trainer.predict(x_cat_test, x_num_test, exp_test)
ct_deviance = poisson_deviance(y_test, ct_preds, exp_test)
print(f"\nBase CT deviance × 10^-2: {ct_deviance:.4f}")
print(f"Improvement over null: {null_deviance - ct_deviance:.4f}")

# COMMAND ----------

# MAGIC %md ## 4. Deep CT: Multi-Head, PLE, SwiGLU
# MAGIC
# MAGIC The deep CT uses b=40, M=2 heads, L=3 layers, PLE for continuous features,
# MAGIC and SwiGLU gating. ~320K parameters, GPU recommended (7 min/run on L4).

# COMMAND ----------

deep_ct = CredibilityTransformer(
    cat_cardinalities=[6, 2, 11, 10],
    n_num_features=5,
    embed_dim=20,          # smaller than paper's b=40 for demo
    n_heads=2,             # M = 2
    n_layers=3,            # L = 3
    alpha=0.98,            # paper uses α=98% for deep CT
    use_ple=True,          # differentiable PLE for continuous features
    n_ple_bins=16,
    use_swiglu=True,       # SwiGLU gating
    dropout=0.01,
)

print(f"Deep CT parameters: {deep_ct.count_parameters():,}")

deep_trainer = CredibilityTransformerTrainer(
    model=deep_ct,
    lr=1e-3,
    weight_decay=1e-2,
    beta2=0.95,
    batch_size=1024,
    early_stopping_patience=20,
    max_epochs=50,         # Paper uses 100
    n_ensemble=2,
    device=device,
    verbose=10,
)

print("Training deep CT...")
deep_trainer.fit(x_cat_train, x_num_train, y_train, exp_train)
deep_preds = deep_trainer.predict(x_cat_test, x_num_test, exp_test)
deep_deviance = poisson_deviance(y_test, deep_preds, exp_test)
print(f"\nDeep CT deviance × 10^-2: {deep_deviance:.4f}")

# COMMAND ----------

# MAGIC %md ## 5. Explainability: Attention as Credibility
# MAGIC
# MAGIC The CLS self-attention weight P = a_{T+1,T+1} is the Bühlmann-Straub prior weight.
# MAGIC (1-P) is the individual credibility weight. These are policy-specific and learned.

# COMMAND ----------

from insurance_credibility_transformer import AttentionExplainer

explainer = AttentionExplainer(ct)

# Get credibility weights for a sample of test policies
sample_size = 500
x_cat_sample = torch.tensor(x_cat_test[:sample_size], dtype=torch.long)
x_num_sample = torch.tensor(x_num_test[:sample_size], dtype=torch.float32)

P = explainer.cls_attention(x_cat_sample, x_num_sample)
individual_cred = explainer.individual_credibility(x_cat_sample, x_num_sample)

print("Credibility weight distribution (1-P = individual weight):")
print(f"  Mean:   {individual_cred.mean():.4f}")
print(f"  Std:    {individual_cred.std():.4f}")
print(f"  Min:    {individual_cred.min():.4f}")
print(f"  Max:    {individual_cred.max():.4f}")

# COMMAND ----------

# Feature attention: which features drive individual credibility?
feat_names = ["veh_class", "fuel_type", "region", "age_band",
              "bonus_malus", "veh_age", "density", "driver_age", "power"]
feat_attn = explainer.feature_attention(x_cat_sample, x_num_sample, feat_names)

print("\nMean CLS→feature attention weights (higher = more influential):")
for name, weights in sorted(feat_attn.items(), key=lambda kv: -kv[1].mean()):
    print(f"  {name:<15}: {weights.mean():.4f} ± {weights.std():.4f}")

# COMMAND ----------

# MAGIC %md ## 6. ICL-Credibility Transformer
# MAGIC
# MAGIC The ICL-CT augments the base CT with a context batch of similar policies
# MAGIC whose outcomes are known. This implements the 3-phase training protocol.

# COMMAND ----------

from insurance_credibility_transformer import ICLCredibilityTransformer, ICLTrainer

# Use the trained base CT from step 3 (phase 1 already complete)
icl_ct = ICLCredibilityTransformer(
    base_ct=ct,
    icl_layers=1,          # 1 = linearized (nearly as good as 2-layer)
    kappa_init=1.0,        # initial Bühlmann credibility coefficient
    linearized=False,
)

icl_trainer = ICLTrainer(
    model=icl_ct,
    lr_phase2=3e-4,
    lr_phase3=3e-5,
    batch_size=32,
    context_size=64,
    epochs_phase2=20,      # Paper uses 50
    epochs_phase3=10,      # Paper uses 20
    patience=10,
    device=device,
    verbose=5,
)

print("Phase 2: training ICL layer (decoder frozen)...")
icl_trainer.fit(
    x_cat_train, x_num_train, y_train, exp_train,
    run_phase3=True,
)

# COMMAND ----------

# Predict with ICL: provide context from training set
n_context = 200
context_idx = np.random.choice(len(y_train), n_context, replace=False)

icl_preds = icl_trainer.predict(
    x_cat_target=x_cat_test[:500],
    x_num_target=x_num_test[:500],
    exposure_target=exp_test[:500],
    x_cat_context=x_cat_train[context_idx],
    x_num_context=x_num_train[context_idx],
    y_context=y_train[context_idx],
    exposure_context=exp_train[context_idx],
)
icl_deviance = poisson_deviance(y_test[:500], icl_preds, exp_test[:500])
print(f"ICL-CT deviance × 10^-2 (500 test, 200 context): {icl_deviance:.4f}")

# COMMAND ----------

# MAGIC %md ## 7. Summary of Results

# COMMAND ----------

results = pl.DataFrame({
    "Model": ["Null model", "Base CT (3 runs)", "Deep CT (2 runs)", "ICL-CT"],
    "Deviance × 10^-2": [null_deviance, ct_deviance, deep_deviance, icl_deviance],
    "Parameters": [0, ct.count_parameters(), deep_ct.count_parameters(), icl_ct.count_parameters()],
})
print(results)

# COMMAND ----------

# MAGIC %md ## Appendix: Loading French MTPL Data
# MAGIC
# MAGIC To reproduce the paper's results, download the actual French MTPL data:
# MAGIC ```python
# MAGIC # freMTPL2freq is available from:
# MAGIC # https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda
# MAGIC # Or via the CASdatasets R package
# MAGIC #
# MAGIC # After loading, the schema is:
# MAGIC # PolicyID, ClaimNb, Exposure, Area, VehPower, VehAge, DrivAge,
# MAGIC # BonusMalus, VehBrand, VehGas, Density, Region
# MAGIC #
# MAGIC # Preprocessing (as in the paper):
# MAGIC # - ClaimNb capped at 4
# MAGIC # - Exposure clipped to [0, 1]
# MAGIC # - log(Density) normalised
# MAGIC # - BonusMalus clipped at 150 and log-transformed
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Notes
# MAGIC
# MAGIC **Why this library exists**: No public PyTorch implementation of the CT existed before.
# MAGIC The paper uses R/Keras. The ICL extension (2026) likely uses Python internally but
# MAGIC hasn't been released. This fills the gap.
# MAGIC
# MAGIC **Key design decisions**:
# MAGIC - c_prior uses a *separate* FNN (not the Transformer layer's FNN) to avoid
# MAGIC   attention contaminating the prior. This matches the paper's description.
# MAGIC - Bernoulli(alpha) sampling is the right approach — no straight-through estimator
# MAGIC   needed because the two branches create separate computational graphs.
# MAGIC - FAISS is optional. Brute-force cosine (O(n²)) is fine for n<50K at inference time.
# MAGIC - ICL decoder is shared with base_ct.decoder (same Python object), so freezing
# MAGIC   the ICL decoder freezes the base CT decoder correctly.
