"""Context retrieval for ICL-Credibility Transformer inference.

Implements the FAISS-based approximate nearest neighbor retrieval from
Section 2.2 of arXiv:2509.08122. At inference:
1. Compute CLS token embeddings for all test rows (c_cred vectors).
2. For each target row, retrieve K=64 nearest training rows by cosine
   similarity in CLS embedding space.
3. Form a context pool of the top c=1000 unique candidates.

Falls back to brute-force cosine similarity if FAISS is not available.
Brute-force is O(n²) but acceptable for training sets <50K at inference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ContextRetriever:
    """Retrieve similar training instances for ICL context.

    Args:
        n_neighbors: K nearest neighbors per target row.
        context_pool_size: Maximum unique context candidates c.
        use_faiss: Whether to attempt FAISS (falls back to cosine if unavailable).
        device: Device for brute-force cosine similarity.
    """

    def __init__(
        self,
        n_neighbors: int = 64,
        context_pool_size: int = 1000,
        use_faiss: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.context_pool_size = context_pool_size
        self.device = torch.device(device or "cpu")
        self._faiss_available = False
        self._index = None  # FAISS index if available

        if use_faiss:
            try:
                import faiss  # noqa: F401
                self._faiss_available = True
            except ImportError:
                pass  # Fall back to brute-force

        # Stored training embeddings
        self._train_embeddings: Optional[np.ndarray] = None

    def fit(self, train_embeddings: np.ndarray) -> "ContextRetriever":
        """Index training embeddings for retrieval.

        Args:
            train_embeddings: (n_train, embed_dim) L2-normalised CLS tokens
                from the training set. Normalisation is applied internally.

        Returns:
            self
        """
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        train_emb_norm = (train_embeddings / norms).astype(np.float32)
        self._train_embeddings = train_emb_norm

        if self._faiss_available:
            import faiss
            d = train_emb_norm.shape[1]
            index = faiss.IndexFlatIP(d)  # Inner product on normalised = cosine
            index.add(train_emb_norm)
            self._index = index

        return self

    def retrieve(
        self,
        query_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find K nearest training instances for each query.

        Args:
            query_embeddings: (n_query, embed_dim) target CLS tokens.

        Returns:
            indices: (n_query, K) training indices per query.
            similarities: (n_query, K) cosine similarities.
        """
        assert self._train_embeddings is not None, "Call fit() before retrieve()"

        # Normalise queries
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        queries_norm = (query_embeddings / norms).astype(np.float32)

        K = min(self.n_neighbors, len(self._train_embeddings))

        if self._faiss_available and self._index is not None:
            sims, indices = self._index.search(queries_norm, K)
        else:
            # Brute-force: (n_query, n_train)
            sims_all = queries_norm @ self._train_embeddings.T
            n_train = sims_all.shape[1]
            if K >= n_train:
                # Return all training points sorted by similarity
                order = np.argsort(-sims_all, axis=1)
                indices = order[:, :K]
                row_idx = np.arange(len(queries_norm))[:, None]
                sims = sims_all[row_idx, indices]
            else:
                # argpartition: kth must be in [0, n_train-1]
                k_idx = np.argpartition(-sims_all, K, axis=1)[:, :K]
                row_idx = np.arange(len(queries_norm))[:, None]
                top_sims = sims_all[row_idx, k_idx]
                order = np.argsort(-top_sims, axis=1)
                indices = k_idx[row_idx, order]
                sims = top_sims[row_idx, order]

        return indices, sims

    def build_context_pool(
        self,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Build context pool: top context_pool_size unique training indices.

        For a chunk of target queries, retrieves all K neighbors, then
        returns the top c unique indices ranked by best similarity across
        the chunk (as per Section 2.2, arXiv:2509.08122).

        Args:
            query_embeddings: (n_chunk, embed_dim) target CLS tokens.

        Returns:
            context_indices: (min(c, n_train),) unique training indices.
        """
        indices, sims = self.retrieve(query_embeddings)

        # For each training index, track best similarity across the chunk
        best_sim: dict[int, float] = {}
        for q_idx in range(len(query_embeddings)):
            for k in range(indices.shape[1]):
                train_idx = int(indices[q_idx, k])
                sim = float(sims[q_idx, k])
                if train_idx not in best_sim or sim > best_sim[train_idx]:
                    best_sim[train_idx] = sim

        # Sort by best similarity, take top c
        sorted_indices = sorted(best_sim.keys(), key=lambda i: -best_sim[i])
        pool_size = min(self.context_pool_size, len(sorted_indices))
        return np.array(sorted_indices[:pool_size], dtype=np.int64)
