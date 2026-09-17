"""
Geographic utilities for location-aware recommendations using the Haversine formula.
"""

import math
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface
    specified in decimal degrees using the Haversine formula.
    
    Returns:
        Distance in kilometers (km).
    """
    if math.isnan(lat1) or math.isnan(lon1) or math.isnan(lat2) or math.isnan(lon2):
        return float("inf")
    
    # Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    distance = R * c
    return distance


def compute_distance_score(distance_km: float, max_distance_km: float = 50.0) -> float:
    """
    Calculate continuous distance proximity score between 0.0 and 1.0.
    1.0 means 0 km distance; decays exponentially as distance increases.
    """
    if math.isinf(distance_km) or distance_km < 0:
        return 0.0
    scale = max(1.0, max_distance_km)
    score = float(np.exp(-distance_km / scale))
    return max(0.0, min(1.0, score))


def add_distance_column(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    """Calculate and attach 'distance_km' to business dataframe."""
    df = df.copy()
    distances = []
    for _, row in df.iterrows():
        dist = haversine_distance(user_lat, user_lon, row["latitude"], row["longitude"])
        distances.append(dist)
    df["distance_km"] = distances
    return df


def filter_by_distance(df: pd.DataFrame, user_lat: float, user_lon: float, max_distance_km: Optional[float] = None) -> pd.DataFrame:
    """Filter business dataframe within specified max distance radius (in km)."""
    df_dist = add_distance_column(df, user_lat, user_lon)
    if max_distance_km is not None and max_distance_km > 0:
        df_dist = df_dist[df_dist["distance_km"] <= max_distance_km]
    return df_dist
