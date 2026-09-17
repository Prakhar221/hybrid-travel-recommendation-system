"""
Cold-start recommendation module for new users without rating history.
Refactored from new_user_recommendation system.ipynb.
"""

import logging
from typing import Optional, List
import pandas as pd
import numpy as np

from src.geo_utils import filter_by_distance, compute_distance_score
from src.utils import logger


def calculate_bayesian_rating(df: pd.DataFrame, m_quantile: float = 0.5) -> pd.Series:
    """Calculate Bayesian smoothed average rating taking review count into account."""
    if len(df) == 0:
        return pd.Series(dtype=float)
    
    C = df["stars"].mean()
    m = df["review_count"].quantile(m_quantile) if len(df) > 1 else 5.0
    m = max(1.0, m)

    v = df["review_count"]
    R = df["stars"]
    bayesian_scores = (v / (v + m)) * R + (m / (v + m)) * C
    return bayesian_scores


class ColdStartRecommender:
    """Filters and ranks businesses for new users based on explicit preferences."""

    def __init__(self, business_df: pd.DataFrame):
        self.business_df = business_df.copy()

    def recommend(
        self,
        city: Optional[str] = None,
        state: Optional[str] = None,
        categories: Optional[List[str]] = None,
        max_budget: Optional[int] = None,
        min_rating: float = 3.5,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """Filter businesses by destination, categories, budget, rating, and distance."""
        df = self.business_df.copy()

        # Filter state & city if specified
        if state and state != "All":
            df = df[df["state"].str.upper() == state.upper()]

        if city and city != "All":
            df = df[df["city"].str.lower() == city.lower()]

        # Filter minimum rating
        if min_rating:
            df = df[df["stars"] >= min_rating]

        # Filter budget
        if max_budget and max_budget in [1, 2, 3, 4]:
            df = df[df["price_range"] <= max_budget]

        # Filter category preference matching
        if categories and len(categories) > 0:
            pattern = "|".join([cat.strip() for cat in categories if cat.strip()])
            if pattern:
                df = df[df["categories"].str.contains(pattern, case=False, na=False)]

        # Distance filtering if coordinates provided
        if user_lat is not None and user_lon is not None:
            df = filter_by_distance(df, user_lat, user_lon, max_distance_km)
            df["distance_score"] = df["distance_km"].apply(lambda d: compute_distance_score(d, max_distance_km or 50.0))
        else:
            df["distance_km"] = 0.0
            df["distance_score"] = 1.0

        if len(df) == 0:
            logger.warning("No businesses matched the specified cold-start preference filters.")
            return pd.DataFrame()

        # Compute Bayesian Rating Score
        df["bayesian_score"] = calculate_bayesian_rating(df)
        
        # Combine distance and rating into ranking score
        df["cold_start_score"] = 0.7 * (df["bayesian_score"] / 5.0) + 0.3 * df["distance_score"]
        
        df = df.sort_values(by="cold_start_score", ascending=False).reset_index(drop=True)
        return df.head(top_k)
