"""
Multi-attribute Ranking Module attaching explanations and structure to final recommendation candidate lists.
"""

from typing import List, Dict, Optional
import pandas as pd

from src.explanations import generate_recommendation_explanation


class RecommendationRanker:
    """Re-ranks recommendation candidates and attaches feature-backed explanations."""

    def rank_and_explain(
        self,
        candidates_df: pd.DataFrame,
        user_id: Optional[str] = None,
        preferred_categories: Optional[List[str]] = None,
        target_budget: Optional[int] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """Sort candidates by composite match score and attach formatted explanations."""
        if candidates_df.empty:
            return candidates_df

        df = candidates_df.copy()
        
        # Ensure match_score exists
        if "match_score" not in df.columns:
            if "stars" in df.columns:
                df["match_score"] = df["stars"] / 5.0
            else:
                df["match_score"] = 0.5

        df = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
        top_df = df.head(top_k).copy()

        # Attach explanations
        explanations = []
        for _, row in top_df.iterrows():
            reasons = generate_recommendation_explanation(
                item_row=row,
                user_id=user_id,
                preferred_categories=preferred_categories,
                target_budget=target_budget
            )
            explanations.append(reasons)

        top_df["explanation_reasons"] = explanations
        return top_df
