"""
Hybrid Recommendation Engine combining Collaborative, Content-Based, Cold-Start, Geolocation, and Budget Signals.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.cold_start import ColdStartRecommender, calculate_bayesian_rating
from src.geo_utils import add_distance_column, compute_distance_score
from src.utils import logger


class HybridRecommender:
    """Multi-attribute Hybrid Recommender integrating collaborative, content, rating, geo, and budget components."""

    def __init__(
        self,
        business_df: pd.DataFrame,
        reviews_df: Optional[pd.DataFrame] = None,
        collab_recommender: Optional[CollaborativeRecommender] = None,
        content_recommender: Optional[ContentBasedRecommender] = None
    ):
        self.business_df = business_df.copy().reset_index(drop=True)
        self.reviews_df = reviews_df
        
        self.content_recommender = content_recommender or ContentBasedRecommender()
        if content_recommender is None:
            self.content_recommender.fit(self.business_df)

        self.collab_recommender = collab_recommender
        if collab_recommender is None and self.reviews_df is not None:
            self.collab_recommender = CollaborativeRecommender(algorithm="svd")
            self.collab_recommender.fit(self.reviews_df)

        self.cold_start = ColdStartRecommender(self.business_df)

    def recommend_hybrid(
        self,
        user_id: Optional[str] = None,
        target_item_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        target_budget: Optional[int] = None,
        min_rating: float = 3.0,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations.
        
        Default Weight Configuration:
        - collab_weight: 0.35
        - content_weight: 0.25
        - rating_weight: 0.20
        - distance_weight: 0.10
        - budget_weight: 0.10
        """
        default_weights = {
            "collab": 0.35 if (user_id and self.collab_recommender) else 0.0,
            "content": 0.30 if (target_item_id or categories) else 0.15,
            "rating": 0.30,
            "distance": 0.15 if (user_lat is not None and user_lon is not None) else 0.0,
            "budget": 0.10 if target_budget is not None else 0.0
        }

        if weights:
            default_weights.update(weights)

        # Normalize weights so sum == 1.0
        total_w = sum(default_weights.values())
        if total_w > 0:
            norm_weights = {k: v / total_w for k, v in default_weights.items()}
        else:
            norm_weights = {k: 0.2 for k in default_weights}

        df = self.business_df.copy()

        # Filter city/state
        if state and state != "All":
            df = df[df["state"].str.upper() == state.upper()]
        if city and city != "All":
            df = df[df["city"].str.lower() == city.lower()]
        if min_rating:
            df = df[df["stars"] >= min_rating]

        if len(df) == 0:
            return pd.DataFrame()

        # 1. Collaborative Scores (0.0 to 1.0)
        df["collab_score"] = 0.0
        if user_id and self.collab_recommender:
            seen_items = set()
            if self.reviews_df is not None:
                seen_items = set(self.reviews_df[self.reviews_df["user_id"] == user_id]["business_id"].tolist())
            
            # Predict ratings for unseen candidate businesses
            collab_preds = []
            for b_id in df["business_id"]:
                if b_id in seen_items:
                    collab_preds.append(0.0)
                else:
                    pred_rating = self.collab_recommender.predict_rating(user_id, b_id)
                    collab_preds.append(pred_rating / 5.0)  # scale to 0-1
            df["collab_score"] = collab_preds

        # 2. Content Similarity Scores (0.0 to 1.0)
        df["content_score"] = 0.0
        if target_item_id and self.content_recommender:
            content_recs = self.content_recommender.recommend(target_item_id, top_k=len(df))
            if not content_recs.empty:
                score_map = dict(zip(content_recs["business_id"], content_recs["similarity_score"]))
                df["content_score"] = df["business_id"].map(score_map).fillna(0.0)
        elif categories and len(categories) > 0:
            pattern = "|".join([cat.strip() for cat in categories if cat.strip()])
            if pattern:
                match_mask = df["categories"].str.contains(pattern, case=False, na=False)
                df["content_score"] = np.where(match_mask, 1.0, 0.2)

        # 3. Normalized Rating Score (0.0 to 1.0)
        bayesian_ratings = calculate_bayesian_rating(df)
        df["rating_score"] = (bayesian_ratings / 5.0).fillna(0.5)

        # 4. Geolocation Distance Score (0.0 to 1.0)
        if user_lat is not None and user_lon is not None:
            df = add_distance_column(df, user_lat, user_lon)
            if max_distance_km and max_distance_km > 0:
                df = df[df["distance_km"] <= max_distance_km]
            df["distance_score"] = df["distance_km"].apply(lambda d: compute_distance_score(d, max_distance_km or 50.0))
        else:
            df["distance_km"] = 0.0
            df["distance_score"] = 1.0

        if len(df) == 0:
            return pd.DataFrame()

        # 5. Budget Match Score (0.0 to 1.0)
        if target_budget is not None and target_budget in [1, 2, 3, 4]:
            # Exact match = 1.0, 1 level off = 0.7, 2 levels off = 0.4, 3 levels off = 0.1
            diff = (df["price_range"] - target_budget).abs()
            df["budget_score"] = np.where(diff == 0, 1.0, np.where(diff == 1, 0.7, np.where(diff == 2, 0.4, 0.1)))
        else:
            df["budget_score"] = 1.0

        # Calculate Final Composite Match Score (Explicitly 0.00 - 1.00 index)
        df["match_score"] = (
            norm_weights["collab"] * df["collab_score"] +
            norm_weights["content"] * df["content_score"] +
            norm_weights["rating"] * df["rating_score"] +
            norm_weights["distance"] * df["distance_score"] +
            norm_weights["budget"] * df["budget_score"]
        )

        df = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
        return df.head(top_k)
