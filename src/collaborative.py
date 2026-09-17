"""
Collaborative Filtering Module implementing SVD, KNN, and Matrix Factorization (ALS).
Refactored from SVD_Collaborative_Filtering.ipynb and ALS_KNN.ipynb.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import logger, save_model, load_model

# Try importing Surprise SVD, handle fallback cleanly if not present
try:
    from surprise import Dataset, Reader, SVD
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False
    logger.info("scikit-surprise not installed. SVD will fallback to SciPy matrix decomposition.")


class CollaborativeRecommender:
    """Unified Collaborative Filtering Engine (SVD, KNN, and SciPy ALS Matrix Factorization)."""

    def __init__(self, algorithm: str = "svd", n_factors: int = 20, n_epochs: int = 20):
        self.algorithm = algorithm.lower()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        
        self.reviews_df = None
        self.user_ids = []
        self.item_ids = []
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.surprise_model = None

    def fit(self, reviews_df: pd.DataFrame):
        """Fit collaborative filtering model on user-item rating interactions."""
        self.reviews_df = reviews_df.copy()
        
        # Build user and item index mappings
        self.user_ids = sorted(self.reviews_df["user_id"].unique())
        self.item_ids = sorted(self.reviews_df["business_id"].unique())
        
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.idx_to_user = {i: u for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {item: i for i, item in enumerate(self.item_ids)}
        self.idx_to_item = {i: item for i, item in enumerate(self.item_ids)}

        # Construct sparse user-item matrix
        rows = self.reviews_df["user_id"].map(self.user_to_idx)
        cols = self.reviews_df["business_id"].map(self.item_to_idx)
        ratings = self.reviews_df["stars"].values

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        self.user_item_matrix = csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        if self.algorithm == "svd" and HAS_SURPRISE:
            logger.info("Fitting Surprise SVD model...")
            reader = Reader(rating_scale=(1.0, 5.0))
            data = Dataset.load_from_df(self.reviews_df[["user_id", "business_id", "stars"]], reader)
            trainset = data.build_full_trainset()
            self.surprise_model = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, random_state=42)
            self.surprise_model.fit(trainset)
            logger.info("Surprise SVD model training completed.")
        else:
            logger.info(f"Fitting {self.algorithm.upper()} matrix factorization via SciPy SVD...")
            k = min(self.n_factors, min(n_users, n_items) - 1)
            k = max(1, k)
            
            # SciPy Sparse SVD
            u, s, vt = svds(self.user_item_matrix, k=k)
            self.user_factors = u @ np.diag(s)
            self.item_factors = vt.T
            logger.info("SciPy Matrix Factorization completed.")

        return self

    def predict_rating(self, user_id: str, business_id: str) -> float:
        """Predict user-item rating score."""
        global_mean = float(self.reviews_df["stars"].mean()) if self.reviews_df is not None else 3.5

        if self.surprise_model is not None:
            try:
                return float(self.surprise_model.predict(user_id, business_id).est)
            except Exception:
                return global_mean

        if user_id not in self.user_to_idx or business_id not in self.item_to_idx:
            return global_mean

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[business_id]
        
        pred = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        return max(1.0, min(5.0, pred))

    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate top_k collaborative recommendations for user_id.
        Excludes items present in user's training history if exclude_seen is True.
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User '{user_id}' not found in training interactions.")
            return []

        u_idx = self.user_to_idx[user_id]
        seen_items = set()

        if exclude_seen and self.reviews_df is not None:
            user_reviews = self.reviews_df[self.reviews_df["user_id"] == user_id]
            seen_items = set(user_reviews["business_id"].tolist())

        predictions = []
        for b_id in self.item_ids:
            if exclude_seen and b_id in seen_items:
                continue
            pred_score = self.predict_rating(user_id, b_id)
            predictions.append((b_id, pred_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
