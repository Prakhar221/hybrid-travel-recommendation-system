"""
Explanation engine generating feature-backed recommendation rationale bullet points.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


def generate_recommendation_explanation(
    item_row: pd.Series,
    user_id: Optional[str] = None,
    preferred_categories: Optional[List[str]] = None,
    target_budget: Optional[int] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None
) -> List[str]:
    """
    Generate bullet points explaining why a business was recommended.
    Driven by actual computed feature attributes.
    """
    reasons = []

    # 1. Rating & Reputation
    stars = item_row.get("stars", 0.0)
    reviews = item_row.get("review_count", 0)
    if stars >= 4.5:
        reasons.append(f"Highly rated place ({stars:.1f}★ with {reviews}+ reviews)")
    elif stars >= 4.0:
        reasons.append(f"Strong community rating ({stars:.1f}★ with {reviews} reviews)")

    # 2. Category Fit
    item_cats = str(item_row.get("categories", "")).split(",")
    item_cats_clean = [c.strip() for c in item_cats]
    
    if preferred_categories:
        matching = [cat for cat in preferred_categories if any(cat.lower() in ic.lower() for ic in item_cats_clean)]
        if matching:
            reasons.append(f"Matches your requested category: '{matching[0]}'")
    elif len(item_cats_clean) > 0:
        reasons.append(f"Popular choice in {item_cats_clean[0]}")

    # 3. Location Proximity
    dist_km = item_row.get("distance_km", None)
    if dist_km is not None and dist_km > 0 and not pd.isna(dist_km):
        if dist_km <= 5.0:
            reasons.append(f"Very close to your location ({dist_km:.1f} km away)")
        elif dist_km <= 20.0:
            reasons.append(f"Within comfortable driving distance ({dist_km:.1f} km away)")

    # 4. Budget Compatibility
    price_range = item_row.get("price_range", 2)
    price_symbols = "$" * int(price_range)
    if target_budget is not None and int(price_range) <= target_budget:
        reasons.append(f"Fits your target budget range ({price_symbols})")
    elif price_range > 0:
        reasons.append(f"Price Tier: {price_symbols}")

    # 5. Collaborative Fit
    collab_score = item_row.get("collab_score", 0.0)
    if collab_score > 0.6 and user_id:
        reasons.append("Similar to places liked by travelers with similar tastes")

    if not reasons:
        reasons.append("Matches overall popularity and satisfaction criteria")

    return reasons
