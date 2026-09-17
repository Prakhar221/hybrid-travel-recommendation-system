"""
Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.
Refactored from content_based_filtering_cosine_similarity.ipynb.
"""

import logging
from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import logger


class ContentBasedRecommender:
    """Content-based recommender calculating cosine similarity over TF-IDF item feature vectors."""

    def __init__(self, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.business_df = None
        self.id_to_idx = {}
        self.name_to_idx = {}

    def fit(self, business_df: pd.DataFrame):
        """Fit TF-IDF vectorizer and compute pairwise cosine similarity matrix."""
        self.business_df = business_df.copy().reset_index(drop=True)
        if "content" not in self.business_df.columns:
            raise KeyError("DataFrame missing 'content' feature column for TF-IDF vectorization.")

        logger.info(f"Fitting TF-IDF Vectorizer on {len(self.business_df)} businesses...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.business_df["content"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Mapping for fast lookup
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.business_df["business_id"])}
        self.name_to_idx = {name.lower(): idx for idx, name in enumerate(self.business_df["name"])}
        logger.info("Content-Based Recommender fitted successfully.")
        return self

    def recommend(self, item_id_or_name: str, top_k: int = 10, threshold: float = 0.0) -> pd.DataFrame:
        """
        Recommend top_k similar businesses based on cosine similarity.
        
        Parameters:
            item_id_or_name: business_id string or business name string
            top_k: number of recommendations to return
            threshold: minimum similarity score cutoff
        """
        if self.cosine_sim is None or self.business_df is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        idx = None
        if item_id_or_name in self.id_to_idx:
            idx = self.id_to_idx[item_id_or_name]
        elif item_id_or_name.lower() in self.name_to_idx:
            idx = self.name_to_idx[item_id_or_name.lower()]

        if idx is None:
            logger.warning(f"Business '{item_id_or_name}' not found in dataset.")
            return pd.DataFrame()

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Filter self and threshold
        sim_scores = [(i, score) for i, score in sim_scores if i != idx and score >= threshold]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_k]

        if not sim_scores:
            return pd.DataFrame()

        indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.business_df.iloc[indices].copy()
        results["similarity_score"] = scores
        return results[["business_id", "name", "categories", "city", "state", "stars", "review_count", "price_range", "latitude", "longitude", "similarity_score"]]
