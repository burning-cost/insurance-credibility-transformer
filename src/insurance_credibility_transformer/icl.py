"""ICL-Credibility Transformer (arXiv:2509.08122).

Augments the base CT with In-Context Learning: at inference time, the model
is given a context batch of similar policies with known outcomes. The ICL
layer attends over this context to refine predictions.

Four components (Section 2 of arXiv:2509.08122):
1. CredibilityTransformer encoder (base CT, pre-trained)
2. OutcomeTokenDecorator: adds claim info to context instances
3. ICLTransformerLayer: causal attention (target↛target)
4. Frozen Decoder from phase 1

Training protocol:
    Phase 1: Train base CT (CredibilityTransformerTrainer)
    Phase 2: Freeze decoder; train decorator + ICL layer + CT encoder
    Phase 3: Full fine-tune (optional)
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attention import StandardFeedForward
from .datasets import InsuranceDataset, collate_insurance
from .loss import PoissonDevianceLoss
from .retrieval import ContextRetriever
from .transformer import CredibilityTransformer


class OutcomeTokenDecorator(nn.Module):
    """Augment context CLS tokens with claim count information.

    Implements equation (2.4) of arXiv:2509.08122:

        c_decor(x_i) = c_cred(x_i) + (v_i / (v_i + kappa)) * z^FNN1(Y_i)

    For target instances (M_i=0): c_decor = c_cred (no outcome available).
    For context instances (M_i=1): claim count Y_i is embedded and added
    with a Bühlmann credibility weight v_i/(v_i+kappa).

    kappa is a learned non-negative parameter (the credibility coefficient).

    Args:
        token_dim: CLS token dimension (2b).
        kappa_init: Initial value for the credibility coefficient.
    """

    def __init__(self, token_dim: int, kappa_init: float = 1.0) -> None:
        super().__init__()
        # FNN to embed claim count into token space: R → R^{2b}
        self.outcome_ffn = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, token_dim),
        )
        # kappa as log-param for positivity: kappa = softplus(log_kappa)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init)))

    @property
    def kappa(self) -> torch.Tensor:
        """Learned Bühlmann credibility coefficient (positive)."""
        return F.softplus(self.log_kappa)

    def forward(
        self,
        c_cred: torch.Tensor,
        y: torch.Tensor,
        exposure: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decorate CLS tokens with outcome information.

        Args:
            c_cred: (batch, token_dim) credibility CLS tokens for all rows.
            y: (batch,) observed claim counts (context only; target unused).
            exposure: (batch,) policy exposure v_i.
            mask: (batch,) binary, 1 = context instance, 0 = target instance.

        Returns:
            c_decor: (batch, token_dim) decorated tokens.
        """
        kappa = self.kappa

        # Bühlmann credibility weight: v / (v + kappa)
        w = exposure / (exposure + kappa)  # (batch,)

        # Embed claim count: (batch, 1) → (batch, token_dim)
        y_emb = self.outcome_ffn(y.unsqueeze(-1))  # (batch, token_dim)

        # Decoration term (applied only to context instances)
        decoration = w.unsqueeze(-1) * y_emb  # (batch, token_dim)
        decoration = decoration * mask.float().unsqueeze(-1)  # zero out targets

        return c_cred + decoration


class ICLTransformerLayer(nn.Module):
    """ICL Transformer layer with causal masking.

    Implements equations (2.5–2.7) of arXiv:2509.08122.

    Standard self-attention on the stacked decorated tokens, but with
    a mask M^∞ that prevents target↔target interactions:
        M^∞_{i,j} = -∞ if both i and j are target instances AND i ≠ j

    Each target can attend to: all context instances + itself.
    Each context can attend to: all instances (no mask needed).

    Args:
        token_dim: CLS token dimension (2b).
        n_heads: Number of attention heads. Default 1 (linearized ICL).
        dropout: Dropout rate.
        linearized: If True, use only feature information for Q and K
            (ignores outcome decoration in queries/keys but not values).
            This is the linearized variant from Section 4, arXiv:2509.08122.
    """

    def __init__(
        self,
        token_dim: int,
        n_heads: int = 1,
        dropout: float = 0.01,
        linearized: bool = False,
    ) -> None:
        super().__init__()
        assert token_dim % n_heads == 0
        self.token_dim = token_dim
        self.n_heads = n_heads
        self.head_dim = token_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.linearized = linearized

        self.W_Q = nn.Linear(token_dim, token_dim, bias=True)
        self.W_K = nn.Linear(token_dim, token_dim, bias=True)
        self.W_V = nn.Linear(token_dim, token_dim, bias=True)
        self.W_O = nn.Linear(token_dim, token_dim, bias=True)

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = StandardFeedForward(token_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _build_causal_mask(
        self,
        is_target: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the target-target block mask.

        M^∞_{i,j} = -∞ if both i and j are target instances AND i ≠ j
        (prevents target-to-target attention across different policies).

        Args:
            is_target: (batch, n) bool mask, True for target instances.

        Returns:
            mask: (batch, n, n) additive attention mask (0 or -inf).
        """
        batch, n = is_target.shape

        # (batch, n, n): True if i is target AND j is target AND i≠j
        target_i = is_target.unsqueeze(2)  # (batch, n, 1)
        target_j = is_target.unsqueeze(1)  # (batch, 1, n)
        both_target = target_i & target_j  # (batch, n, n)

        # Remove diagonal (i=j): each target can attend to itself
        diag = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
        mask_bool = both_target & ~diag  # (batch, n, n)

        mask = torch.zeros(batch, n, n, device=device)
        mask[mask_bool] = float("-inf")

        return mask

    def forward(
        self,
        c_decor: torch.Tensor,
        is_target: torch.Tensor,
        c_cred_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ICL attention pass.

        Args:
            c_decor: (batch, n, token_dim) decorated CLS tokens.
                batch dimension here is the context+target pool size.
            is_target: (batch, n) bool, True = target instance.
            c_cred_base: (batch, n, token_dim) undecored CLS tokens.
                Only used in linearized mode for Q and K projections.

        Returns:
            c_icl_trans: (batch, n, token_dim) ICL-transformed tokens.
        """
        batch, n, _ = c_decor.shape

        # In linearized mode, Q and K use undecored embeddings
        qk_input = c_cred_base if (self.linearized and c_cred_base is not None) else c_decor

        Q = self.W_Q(qk_input)   # (batch, n, token_dim)
        K = self.W_K(qk_input)
        V = self.W_V(c_decor)    # Values always use decorated tokens

        # Split heads: (batch, n_heads, n, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, n, n)

        # Causal mask: (batch, n, n) → (batch, 1, n, n)
        causal_mask = self._build_causal_mask(is_target, c_decor.device)
        causal_mask = causal_mask.unsqueeze(1)  # (batch, 1, n, n)

        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)

        # H = AV
        H = torch.matmul(attn, V)  # (batch, n_heads, n, head_dim)
        H = H.transpose(1, 2).contiguous().view(batch, n, self.token_dim)
        H = self.W_O(H)

        # Skip + norm
        c_mid = self.norm1(c_decor + self.dropout(H))

        # FNN + skip + norm
        ffn_out = self.ffn(c_mid)
        c_icl_trans = self.norm2(c_mid + self.dropout(ffn_out))

        return c_icl_trans


class ICLCredibilityTransformer(nn.Module):
    """ICL-augmented Credibility Transformer.

    Wraps a pre-trained base CT with:
    - OutcomeTokenDecorator (adds claim outcomes to context)
    - ICLTransformerLayer (causal cross-batch attention)
    - Frozen decoder from base CT phase 1

    For inference, requires a context batch of training instances with
    known outcomes. Use ICLTrainer to manage the 3-phase training.

    Args:
        base_ct: Pre-trained CredibilityTransformer (phase 1 complete).
        icl_layers: Number of ICL Transformer layers. 1 = linearized, 2 = full.
        kappa_init: Initial Bühlmann credibility coefficient.
        linearized: Use linearized (feature-only Q/K) ICL attention.
    """

    def __init__(
        self,
        base_ct: CredibilityTransformer,
        icl_layers: int = 2,
        kappa_init: float = 1.0,
        linearized: bool = False,
    ) -> None:
        super().__init__()
        self.base_ct = base_ct
        token_dim = 2 * base_ct.embed_dim

        self.decorator = OutcomeTokenDecorator(token_dim, kappa_init)

        self.icl_layers = nn.ModuleList([
            ICLTransformerLayer(
                token_dim,
                n_heads=base_ct.n_heads,
                dropout=0.01,
                linearized=linearized,
            )
            for _ in range(icl_layers)
        ])

        # Frozen decoder (from base CT phase 1)
        # We share the decoder weights; freezing is done in the trainer
        self.decoder = base_ct.decoder

    def forward(
        self,
        x_cat_context: Optional[torch.Tensor],
        x_num_context: Optional[torch.Tensor],
        y_context: torch.Tensor,
        exposure_context: torch.Tensor,
        x_cat_target: Optional[torch.Tensor],
        x_num_target: Optional[torch.Tensor],
        exposure_target: torch.Tensor,
    ) -> torch.Tensor:
        """ICL forward pass.

        The context instances have known outcomes (y_context). The target
        instances need predictions.

        Args:
            x_cat_context: (n_ctx, n_cat) context categorical features.
            x_num_context: (n_ctx, n_num) context continuous features.
            y_context: (n_ctx,) context claim counts.
            exposure_context: (n_ctx,) context exposure.
            x_cat_target: (n_tgt, n_cat) target categorical features.
            x_num_target: (n_tgt, n_num) target continuous features.
            exposure_target: (n_tgt,) target exposure.

        Returns:
            mu_target: (n_tgt,) predicted claim frequencies for target rows.
        """
        n_ctx = y_context.shape[0]
        n_tgt = exposure_target.shape[0]

        # Get base CT CLS tokens for all rows
        c_ctx, _ = self.base_ct.encode(x_cat_context, x_num_context)    # (n_ctx, 2b)
        c_tgt, _ = self.base_ct.encode(x_cat_target, x_num_target)      # (n_tgt, 2b)

        # Target claim counts are unknown → use zeros
        y_target_dummy = torch.zeros(n_tgt, device=y_context.device)

        # Stack: (1, n_ctx+n_tgt, 2b) — batch dim = 1 for the ICL layer
        c_all = torch.cat([c_ctx, c_tgt], dim=0).unsqueeze(0)           # (1, n, 2b)
        y_all = torch.cat([y_context, y_target_dummy], dim=0).unsqueeze(0)  # (1, n)
        exp_all = torch.cat([exposure_context, exposure_target], dim=0).unsqueeze(0)  # (1, n)

        # is_target mask: context=False, target=True
        is_target = torch.cat([
            torch.zeros(n_ctx, dtype=torch.bool, device=y_context.device),
            torch.ones(n_tgt, dtype=torch.bool, device=y_context.device),
        ]).unsqueeze(0)  # (1, n)

        # Decorate: context gets outcome info, targets get none
        context_mask = ~is_target  # (1, n): 1 for context, 0 for target
        c_decor = self.decorator(
            c_all.squeeze(0),
            y_all.squeeze(0),
            exp_all.squeeze(0),
            context_mask.squeeze(0),
        ).unsqueeze(0)  # (1, n, 2b)

        # ICL Transformer layers
        c_icl = c_decor
        for layer in self.icl_layers:
            c_icl = layer(c_icl, is_target, c_cred_base=c_all)

        # Extract target tokens and decode
        c_tgt_icl = c_icl[0, n_ctx:, :]  # (n_tgt, 2b)

        log_mu = self.decoder(c_tgt_icl).squeeze(-1)  # (n_tgt,)
        mu = torch.exp(log_mu)  # raw rate

        # Multiply by target exposure
        return mu * exposure_target


class ICLTrainer:
    """Three-phase training for ICLCredibilityTransformer.

    Phase 2: Freeze decoder; train decorator + ICL layers + CT encoder.
    Phase 3: Full fine-tune (optional).

    Args:
        model: ICLCredibilityTransformer with base_ct already trained (phase 1).
        lr_phase2: Learning rate for phase 2.
        lr_phase3: Learning rate for phase 3.
        weight_decay: AdamW weight decay.
        beta2: Adam beta2.
        batch_size: Training batch size (target instances per batch).
        context_size: Number of context instances per training batch.
        epochs_phase2: Max epochs for phase 2.
        epochs_phase3: Max epochs for phase 3.
        patience: Early stopping patience.
        device: Torch device.
        verbose: Print every N epochs.
    """

    def __init__(
        self,
        model: ICLCredibilityTransformer,
        lr_phase2: float = 3e-4,
        lr_phase3: float = 3e-5,
        weight_decay: float = 1e-2,
        beta2: float = 0.95,
        batch_size: int = 64,
        context_size: int = 64,
        epochs_phase2: int = 50,
        epochs_phase3: int = 20,
        patience: int = 10,
        device: Optional[str] = None,
        verbose: int = 10,
    ) -> None:
        self.model = model
        self.lr_phase2 = lr_phase2
        self.lr_phase3 = lr_phase3
        self.weight_decay = weight_decay
        self.beta2 = beta2
        self.batch_size = batch_size
        self.context_size = context_size
        self.epochs_phase2 = epochs_phase2
        self.epochs_phase3 = epochs_phase3
        self.patience = patience
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.loss_fn = PoissonDevianceLoss(use_exposure=True)

    def _freeze_decoder(self) -> None:
        for param in self.model.decoder.parameters():
            param.requires_grad = False

    def _unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def _sample_context(
        self,
        dataset: InsuranceDataset,
        exclude_indices: List[int],
        n_context: int,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample context instances from training data (exclude target batch)."""
        n = len(dataset)
        pool = np.setdiff1d(np.arange(n), exclude_indices)
        if len(pool) < n_context:
            n_context = len(pool)
        ctx_idx = rng.choice(pool, size=n_context, replace=False)

        x_cat_list, x_num_list, y_list, exp_list = [], [], [], []
        for i in ctx_idx:
            xc, xn, y, exp = dataset[i]
            x_cat_list.append(xc)
            x_num_list.append(xn)
            y_list.append(y)
            exp_list.append(exp)

        x_cat = torch.stack(x_cat_list) if x_cat_list[0] is not None else None
        x_num = torch.stack(x_num_list) if x_num_list[0] is not None else None
        y = torch.stack(y_list)
        exposure = torch.stack(exp_list)

        return x_cat, x_num, y, exposure, torch.tensor(ctx_idx, dtype=torch.long)

    def _run_phase(
        self,
        dataset: InsuranceDataset,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        phase_name: str,
    ) -> Dict[str, List[float]]:
        """Generic training loop for one phase."""
        from .trainer import EarlyStopping
        n = len(dataset)
        indices = np.arange(n)
        rng = np.random.default_rng(42)
        stopper = EarlyStopping(patience=self.patience, mode="min")
        history: Dict[str, List[float]] = {"train_loss": []}
        model = self.model.to(self.device)

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_losses = []
            rng.shuffle(indices)

            for start in range(0, n, self.batch_size):
                tgt_idx = indices[start:start + self.batch_size].tolist()
                if len(tgt_idx) == 0:
                    continue

                # Load target batch
                tgt_items = [dataset[i] for i in tgt_idx]
                x_cat_t, x_num_t, y_t_dummy, exp_t = collate_insurance(tgt_items)

                # Sample context (disjoint from target)
                x_cat_c, x_num_c, y_c, exp_c, _ = self._sample_context(
                    dataset, tgt_idx, self.context_size, rng
                )

                # Move to device
                def _to(t):
                    return t.to(self.device) if t is not None else None

                x_cat_c, x_num_c = _to(x_cat_c), _to(x_num_c)
                y_c, exp_c = y_c.to(self.device), exp_c.to(self.device)
                x_cat_t, x_num_t = _to(x_cat_t), _to(x_num_t)
                exp_t = exp_t.to(self.device)

                # We need true y for target too (for loss), but target mask=0
                # in decorator means y_target is not used in decoration
                y_t_true = torch.stack([dataset[i][2] for i in tgt_idx]).to(self.device)

                optimizer.zero_grad()
                pred = model(x_cat_c, x_num_c, y_c, exp_c, x_cat_t, x_num_t, exp_t)
                loss = self.loss_fn(pred, y_t_true, exp_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(epoch_loss)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"  [{phase_name}] Epoch {epoch:4d} | loss={epoch_loss:.6f}")

            if stopper.step(epoch_loss, model):
                if self.verbose > 0:
                    print(f"  [{phase_name}] Early stopping at epoch {epoch}")
                break

        stopper.restore_best(model)
        return history

    def fit(
        self,
        x_cat: Optional[np.ndarray],
        x_num: Optional[np.ndarray],
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        run_phase3: bool = False,
    ) -> "ICLTrainer":
        """Run phase 2 (and optionally phase 3) training.

        Phase 1 must already be complete (base_ct trained).

        Args:
            x_cat: (n, n_cat) categorical features.
            x_num: (n, n_num) continuous features.
            y: (n,) claim counts.
            exposure: (n,) exposure.
            run_phase3: Also run phase 3 fine-tuning after phase 2.

        Returns:
            self
        """
        dataset = InsuranceDataset(x_cat, x_num, y, exposure)

        # Phase 2: freeze decoder
        print("Phase 2: training ICL layer (decoder frozen)")
        self._freeze_decoder()
        phase2_params = [p for p in self.model.parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW(
            phase2_params, lr=self.lr_phase2,
            weight_decay=self.weight_decay, betas=(0.9, self.beta2)
        )
        self._run_phase(dataset, opt2, self.epochs_phase2, "phase2")

        if run_phase3:
            print("Phase 3: full fine-tuning")
            self._unfreeze_all()
            opt3 = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr_phase3,
                weight_decay=self.weight_decay, betas=(0.9, self.beta2)
            )
            self._run_phase(dataset, opt3, self.epochs_phase3, "phase3")

        return self

    def predict(
        self,
        x_cat_target: Optional[np.ndarray],
        x_num_target: Optional[np.ndarray],
        exposure_target: Optional[np.ndarray],
        x_cat_context: Optional[np.ndarray],
        x_num_context: Optional[np.ndarray],
        y_context: np.ndarray,
        exposure_context: Optional[np.ndarray],
    ) -> np.ndarray:
        """Predict for target instances given a context batch.

        Args:
            x_cat_target, x_num_target, exposure_target: Target inputs.
            x_cat_context, x_num_context, y_context, exposure_context:
                Context instances with known outcomes.

        Returns:
            mu: (n_target,) predictions.
        """
        self.model.eval()
        self.model.to(self.device)

        def _arr_to_tensor(arr, dtype=torch.float32):
            if arr is None:
                return None
            return torch.tensor(np.asarray(arr), dtype=dtype).to(self.device)

        def _int_tensor(arr):
            if arr is None:
                return None
            return torch.tensor(np.asarray(arr), dtype=torch.long).to(self.device)

        with torch.no_grad():
            pred = self.model(
                _int_tensor(x_cat_context),
                _arr_to_tensor(x_num_context),
                torch.tensor(np.asarray(y_context), dtype=torch.float32).to(self.device),
                _arr_to_tensor(exposure_context) if exposure_context is not None
                else torch.ones(len(y_context), device=self.device),
                _int_tensor(x_cat_target),
                _arr_to_tensor(x_num_target),
                _arr_to_tensor(exposure_target) if exposure_target is not None
                else torch.ones(len(x_cat_target or x_num_target), device=self.device),
            )

        return pred.cpu().numpy()
