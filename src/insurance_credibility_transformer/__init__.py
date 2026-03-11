"""insurance-credibility-transformer

PyTorch implementation of the Credibility Transformer from:
    Richman, Scognamiglio & Wüthrich (2024). The Credibility Transformer.
    arXiv:2409.16653. European Actuarial Journal.

With the In-Context Learning extension from:
    Padayachy, Richman, Scognamiglio & Wüthrich (2026).
    ICL-Enhanced Credibility Transformer. arXiv:2509.08122.

The Credibility Transformer adapts the FT-Transformer architecture
(Gorishniy et al. 2021) for insurance pricing by giving the CLS token a
Bühlmann-Straub credibility interpretation:
    P = a_{T+1,T+1}  →  portfolio prior weight
    1-P              →  individual credibility weight

Both are learned and policy-specific — unlike classical Bühlmann where the
credibility weight is a function of observed claim volume only.
"""

from .attention import CredibilityMechanism, MultiHeadCredibilityAttention
from .datasets import InsuranceDataset, collate_insurance
from .decoder import FrequencyDecoder, SeverityDecoder
from .explain import AttentionExplainer
from .icl import ICLCredibilityTransformer, ICLTrainer, OutcomeTokenDecorator
from .loss import GammaDevianceLoss, PoissonDevianceLoss
from .retrieval import ContextRetriever
from .tokenizer import EntityEmbedding, FeatureTokenizer, NumericalEmbedding, PiecewiseLinearEncoding
from .trainer import CredibilityTransformerTrainer, EarlyStopping
from .transformer import CredibilityTransformer, CredibilityTransformerLayer

__version__ = "0.1.0"
__all__ = [
    # Main models
    "CredibilityTransformer",
    "ICLCredibilityTransformer",
    # Trainer
    "CredibilityTransformerTrainer",
    "ICLTrainer",
    "EarlyStopping",
    # Explainability
    "AttentionExplainer",
    # Tokenization
    "FeatureTokenizer",
    "EntityEmbedding",
    "NumericalEmbedding",
    "PiecewiseLinearEncoding",
    # Attention
    "MultiHeadCredibilityAttention",
    "CredibilityMechanism",
    "CredibilityTransformerLayer",
    # Decoder
    "FrequencyDecoder",
    "SeverityDecoder",
    # Loss
    "PoissonDevianceLoss",
    "GammaDevianceLoss",
    # Data
    "InsuranceDataset",
    "collate_insurance",
    # ICL components
    "OutcomeTokenDecorator",
    "ContextRetriever",
]
