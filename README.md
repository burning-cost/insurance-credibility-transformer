# insurance-credibility-transformer

PyTorch implementation of the **Credibility Transformer** from Richman, Scognamiglio & Wüthrich (2024) with the **ICL extension** from Padayachy et al. (2026).

## The problem

Classical Bühlmann-Straub credibility assigns a fixed weight to individual experience versus the portfolio mean. That weight is a function of claim volume only — it doesn't care what the covariates look like. A policy with 5 years of claim-free history in a low-risk segment gets the same credibility as 5 years in a high-volatility segment.

GLMs ignore this problem entirely. Every prediction is treated as equally reliable.

Transformers don't have a natural answer either. The FT-Transformer (Gorishniy et al. 2021) is excellent at tabular feature interaction, but it has no mechanism for expressing "I'm uncertain about this risk, default to the portfolio mean".

## The solution

The Credibility Transformer solves this by repurposing the [CLS] token. In standard Transformers, the CLS token is a learned summary of all features. In the CT, it plays a dual role:

- **c_trans**: CLS after full self-attention (sees all covariates). This is the individual risk estimate.
- **c_prior**: CLS through the FNN only, without attention. This is the portfolio mean.

The CLS self-attention weight P = a_{T+1,T+1} is the Bühlmann-Straub prior weight. (1-P) is the individual credibility weight. Both are **learned** and **policy-specific**. A policy with unusual feature combinations gets low P (high individual credibility). A policy that looks like the average gets high P.

During training, a Bernoulli(alpha) switch alternates between the two paths, forcing c_prior to encode the portfolio mean and c_trans to encode covariate signal.

The result: on French MTPL (610K policies), the base CT with 1,746 parameters outperforms a GLM (Poisson deviance 23.711 vs 24.102 × 10^-2), and beats models with 15x more parameters.

## Installation

```bash
pip install insurance-credibility-transformer
```

Optional FAISS for ICL retrieval:

```bash
pip install "insurance-credibility-transformer[faiss]"
```

## Quick start

```python
from insurance_credibility_transformer import (
    CredibilityTransformer,
    CredibilityTransformerTrainer,
    AttentionExplainer,
)

# Base CT (1,746 params, trains on CPU in minutes)
ct = CredibilityTransformer(
    cat_cardinalities=[6, 2, 11, 22],  # levels per categorical feature
    n_num_features=5,
    embed_dim=5,           # b in paper
    n_heads=1,             # M (base CT = 1)
    n_layers=1,            # L (base CT = 1)
    alpha=0.90,            # credibility parameter
    dropout=0.01,
    link="log",            # frequency model
)

# Training
trainer = CredibilityTransformerTrainer(
    model=ct,
    loss="poisson",
    lr=1e-3,
    batch_size=1024,
    early_stopping_patience=20,
    n_ensemble=20,         # paper uses 20 runs averaged
)
trainer.fit(X_cat, X_num, y, exposure)
preds = trainer.predict(X_cat_test, X_num_test, exposure_test)

# Explainability: who gets individual credibility?
explainer = AttentionExplainer(ct)
P = explainer.cls_attention(X_cat, X_num)           # prior weights
z = explainer.individual_credibility(X_cat, X_num)  # individual weights (1-P)
```

## Deep CT

The deep CT uses multi-head attention (M=2), three layers (L=3), SwiGLU gating, and differentiable Piecewise Linear Encoding for continuous features. ~320K parameters. GPU recommended.

```python
deep_ct = CredibilityTransformer(
    cat_cardinalities=[6, 2, 11, 22],
    n_num_features=5,
    embed_dim=40,          # b=40 (paper)
    n_heads=2,
    n_layers=3,
    alpha=0.98,            # paper: alpha=98% for deep CT
    use_ple=True,          # PLE for continuous features
    n_ple_bins=16,
    use_swiglu=True,       # SwiGLU gating
)
```

## ICL extension

The ICL-CT augments inference with a context batch of similar policies whose claim history is known. The ICL layer attends over this context with causal masking (target policies cannot attend to each other).

```python
from insurance_credibility_transformer import ICLCredibilityTransformer, ICLTrainer

# Phase 1: train base CT first
trainer.fit(x_cat_train, x_num_train, y_train, exposure_train)

# Phase 2 + 3: ICL training
icl_ct = ICLCredibilityTransformer(base_ct=ct, icl_layers=2)
icl_trainer = ICLTrainer(icl_ct, lr_phase2=3e-4, lr_phase3=3e-5)
icl_trainer.fit(x_cat_train, x_num_train, y_train, exposure_train, run_phase3=True)

# Predict with context
preds = icl_trainer.predict(
    x_cat_target, x_num_target, exposure_target,
    x_cat_context, x_num_context, y_context, exposure_context,
)
```

## Data format

```
X_cat:    (n, n_cat)   integer-encoded categorical features (0-indexed)
X_num:    (n, n_num)   float32 continuous features
y:        (n,)         claim counts (integer or float)
exposure: (n,)         policy years (v_i in paper)
```

Pass `None` for `X_cat` or `X_num` if there are no features of that type.

## Results (French MTPL)

From arXiv:2409.16653, out-of-sample Poisson deviance × 10^-2:

| Model | Deviance | Parameters |
|-------|----------|------------|
| Null model | 25.445 | — |
| GLM | 24.102 | — |
| FNN ensemble | 23.783 | — |
| CT nadam ensemble | 23.711 | 1,746 |
| Deep CT ensemble | 23.577 | ~320K |
| CAFTT (Brauer 2024) | 23.726 | 27,133 |

The base CT gets within 0.13 units of the best deep model with 0.5% as many parameters.

## Architecture decisions

**Why a separate c_prior FNN, not the Transformer FNN**: The credibility mechanism requires c_prior to be independent of attention. Reusing the Transformer FNN would contaminate c_prior with attention-processed information in multi-layer models. A dedicated FNN keeps the computation graphs clean.

**Why Bernoulli sampling, not mixing**: The paper is explicit — Z is binary, not a soft interpolation. This forces the decoder to work with either the individual or portfolio embedding in each gradient step, preventing the model from learning to average them.

**Why NormFormer by default**: The CT training instability reported in the paper (Section 4) comes from gradient magnitude mismatch between attention heads. Per-head scaling coefficients (NormFormer, Shleifer et al. 2021) are applied inside the attention module by default, even for the base CT. The cost is two extra parameters per head.

**Why no sklearn dependency**: The API is sklearn-compatible (fit/predict) but doesn't depend on sklearn. The library targets actuaries who may not have sklearn installed, and the Transformer training loop doesn't benefit from sklearn's cross-validation infrastructure.

## Performance

Benchmarked on French MTPL (freMTPL2freq, 610k policies). Full results from arXiv:2409.16653 (base CT) and arXiv:2509.08122 (ICL extension). Out-of-sample Poisson deviance x10^-2:

| Model | Deviance | Parameters |
|-------|----------|------------|
| Null model | 25.445 | — |
| GLM | 24.102 | — |
| FNN ensemble | 23.783 | — |
| CT nadam ensemble | 23.711 | 1,746 |
| Deep CT ensemble | 23.577 | ~320K |

The base CT (1,746 params) matches the Credibility TRM (the previous best efficient model) and outperforms the GLM by 0.391 units of deviance, while being interpretable via the individual credibility weights z = 1-P. The ICL extension further improves on the deep CT when a relevant context batch is available (policies from the same scheme or product class whose outcomes are known).

Training time: base CT trains to convergence in under 10 minutes on CPU for 610k policies. Deep CT requires GPU for practical training times (30-60 minutes on a single A100).


## References

- Richman, Scognamiglio & Wüthrich (2024). The Credibility Transformer. arXiv:2409.16653
- Padayachy, Richman, Scognamiglio & Wüthrich (2026). ICL-Enhanced Credibility Transformer. arXiv:2509.08122
- Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. arXiv:2106.11959
- Bühlmann & Straub (1970). Credibility for Loss Ratios.
