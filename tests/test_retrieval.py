"""Tests for retrieval.py: ContextRetriever (brute-force cosine)."""

import numpy as np
import pytest

from insurance_credibility_transformer.retrieval import ContextRetriever


class TestContextRetriever:
    def test_basic_retrieval_shape(self):
        retriever = ContextRetriever(n_neighbors=5, use_faiss=False)
        train_emb = np.random.randn(100, 16).astype(np.float32)
        retriever.fit(train_emb)

        query = np.random.randn(10, 16).astype(np.float32)
        indices, sims = retriever.retrieve(query)
        assert indices.shape == (10, 5)
        assert sims.shape == (10, 5)

    def test_retrieves_nearest_neighbor(self):
        """Query identical to a training point should retrieve itself first."""
        retriever = ContextRetriever(n_neighbors=3, use_faiss=False)
        train_emb = np.eye(20, dtype=np.float32)  # orthogonal basis vectors
        retriever.fit(train_emb)

        query = np.eye(20, dtype=np.float32)
        indices, sims = retriever.retrieve(query)
        # Each row of query is identical to one training row
        for i in range(20):
            assert i in indices[i], f"Row {i} not in its own top-3"

    def test_similarities_in_range(self):
        retriever = ContextRetriever(n_neighbors=4, use_faiss=False)
        train_emb = np.random.randn(50, 8).astype(np.float32)
        retriever.fit(train_emb)
        query = np.random.randn(5, 8).astype(np.float32)
        _, sims = retriever.retrieve(query)
        assert (sims >= -1.01).all() and (sims <= 1.01).all()

    def test_fewer_neighbors_than_k(self):
        """When n_train < K, should return all training instances."""
        retriever = ContextRetriever(n_neighbors=20, use_faiss=False)
        train_emb = np.random.randn(5, 8).astype(np.float32)
        retriever.fit(train_emb)
        query = np.random.randn(3, 8).astype(np.float32)
        indices, sims = retriever.retrieve(query)
        # K is clipped to min(n_neighbors, n_train) = 5
        assert indices.shape[1] == 5

    def test_context_pool_size(self):
        retriever = ContextRetriever(n_neighbors=5, context_pool_size=10, use_faiss=False)
        train_emb = np.random.randn(50, 8).astype(np.float32)
        retriever.fit(train_emb)
        query = np.random.randn(4, 8).astype(np.float32)
        pool = retriever.build_context_pool(query)
        assert len(pool) <= 10
        # All indices valid
        assert (pool >= 0).all() and (pool < 50).all()

    def test_requires_fit_before_retrieve(self):
        retriever = ContextRetriever(use_faiss=False)
        with pytest.raises(AssertionError):
            retriever.retrieve(np.random.randn(2, 8).astype(np.float32))
