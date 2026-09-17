"""
Automated pytest test suite verifying recommendation core modules, preprocessing, geolocations,
and evaluation metrics.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import DataPreprocessor, parse_price_range
from src.geo_utils import haversine_distance, compute_distance_score, add_distance_column
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.cold_start import ColdStartRecommender
from src.hybrid import HybridRecommender
from src.ranking import RecommendationRanker
from src.evaluation import (
    user_wise_train_test_split,
    calculate_rmse,
    calculate_mae,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    calculate_ndcg_at_k
)


@pytest.fixture
def sample_business_data():
    return pd.DataFrame([
        {
            "business_id": "b1",
            "name": "Los Agaves",
            "city": "Santa Barbara",
            "state": "CA",
            "categories": "Mexican, Restaurants",
            "stars": 4.5,
            "review_count": 3834,
            "latitude": 34.4258,
            "longitude": -119.6872,
            "price_range": 2,
            "attributes": "{'RestaurantsPriceRange2': '2'}",
            "content": "los agaves mexican restaurants santa barbara ca"
        },
        {
            "business_id": "b2",
            "name": "Toma Restaurant & Bar",
            "city": "Santa Barbara",
            "state": "CA",
            "categories": "Italian, Seafood, Bars",
            "stars": 4.5,
            "review_count": 1084,
            "latitude": 34.4095,
            "longitude": -119.6920,
            "price_range": 3,
            "attributes": "{'RestaurantsPriceRange2': '3'}",
            "content": "toma restaurant & bar italian seafood bars santa barbara ca"
        },
        {
            "business_id": "b3",
            "name": "Pho Street",
            "city": "Philadelphia",
            "state": "PA",
            "categories": "Vietnamese, Asian Fusion",
            "stars": 4.0,
            "review_count": 150,
            "latitude": 39.9526,
            "longitude": -75.1652,
            "price_range": 1,
            "attributes": "{'RestaurantsPriceRange2': '1'}",
            "content": "pho street vietnamese asian fusion philadelphia pa"
        }
    ])


@pytest.fixture
def sample_review_data():
    return pd.DataFrame([
        {"user_id": "u1", "business_id": "b1", "stars": 5.0},
        {"user_id": "u1", "business_id": "b2", "stars": 4.5},
        {"user_id": "u1", "business_id": "b3", "stars": 1.0},
        {"user_id": "u2", "business_id": "b1", "stars": 4.0},
        {"user_id": "u2", "business_id": "b2", "stars": 5.0},
        {"user_id": "u3", "business_id": "b3", "stars": 4.5},
    ])


def test_parse_price_range():
    assert parse_price_range("{'RestaurantsPriceRange2': '3'}") == 3
    assert parse_price_range(None) == 2


def test_haversine_distance():
    # Distance between Santa Barbara points
    dist = haversine_distance(34.4258, -119.6872, 34.4095, -119.6920)
    assert 1.0 <= dist <= 3.0
    score = compute_distance_score(dist, max_distance_km=50.0)
    assert 0.0 <= score <= 1.0


def test_content_based_recommender(sample_business_data):
    cb = ContentBasedRecommender()
    cb.fit(sample_business_data)
    recs = cb.recommend("Los Agaves", top_k=2)
    assert not recs.empty
    assert "similarity_score" in recs.columns


def test_collaborative_recommender(sample_review_data):
    cf = CollaborativeRecommender(algorithm="svd", n_factors=5)
    cf.fit(sample_review_data)
    pred = cf.predict_rating("u1", "b1")
    assert 1.0 <= pred <= 5.0
    recs = cf.recommend("u1", top_k=2, exclude_seen=True)
    assert isinstance(recs, list)


def test_cold_start_recommender(sample_business_data):
    cs = ColdStartRecommender(sample_business_data)
    recs = cs.recommend(city="Santa Barbara", min_rating=4.0, top_k=2)
    assert len(recs) <= 2
    assert "cold_start_score" in recs.columns


def test_hybrid_recommender(sample_business_data, sample_review_data):
    hybrid = HybridRecommender(sample_business_data, sample_review_data)
    recs = hybrid.recommend_hybrid(user_id="u1", city="Santa Barbara", top_k=2)
    assert not recs.empty
    assert "match_score" in recs.columns


def test_ranking_and_explanations(sample_business_data):
    ranker = RecommendationRanker()
    ranked = ranker.rank_and_explain(sample_business_data, preferred_categories=["Mexican"])
    assert "explanation_reasons" in ranked.columns
    assert len(ranked["explanation_reasons"].iloc[0]) > 0


def test_evaluation_metrics():
    y_true = np.array([5.0, 4.0, 3.0])
    y_pred = np.array([4.8, 4.2, 2.8])
    assert calculate_rmse(y_true, y_pred) > 0.0
    assert calculate_mae(y_true, y_pred) > 0.0

    recommended = ["b1", "b2", "b3"]
    relevant = {"b1", "b2"}
    assert calculate_precision_at_k(recommended, relevant, k=2) == 1.0
    assert calculate_recall_at_k(recommended, relevant, k=2) == 1.0
    assert calculate_hit_rate_at_k(recommended, relevant, k=2) == 1.0
    assert calculate_ndcg_at_k(recommended, relevant, k=2) > 0.0


def test_user_wise_split(sample_review_data):
    train, test = user_wise_train_test_split(sample_review_data, test_ratio=0.3, min_interactions=2)
    assert len(train) + len(test) == len(sample_review_data)
