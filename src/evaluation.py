"""
Evaluation module for Travel Recommendation System.
Implements User-Wise Stratified Train/Test Splitting (min 5 interactions per user),
candidate set exclusion of training items, and live empirical metric calculation
(RMSE, MAE, Precision@K, Recall@K, Hit Rate@K, NDCG@K).

Includes support for Full Yelp Dataset Evaluation workflows and pipeline validation reporting.
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np
import pandas as pd

from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.utils import logger, RAW_DATA_DIR, PROCESSED_DATA_DIR


def user_wise_train_test_split(
    reviews_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    User-Wise Stratified Train/Test Split.
    
    Protocol:
    - Users with >= min_interactions (default 5) are ELIGIBLE for test holdout.
    - For eligible users, test_ratio (default 20%) of their interactions are held out in the test set.
    - Users with < min_interactions (< 5) are 100% placed into the training set.
    - Held-out test interactions remain strictly isolated from model training and candidate generation.
    """
    train_records = []
    test_records = []

    user_groups = reviews_df.groupby("user_id")

    for user_id, group in user_groups:
        if len(group) >= min_interactions:
            n_test = max(1, int(len(group) * test_ratio))
            shuffled = group.sample(frac=1.0, random_state=42)
            test_records.append(shuffled.iloc[:n_test])
            train_records.append(shuffled.iloc[n_test:])
        else:
            train_records.append(group)

    train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
    test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()
    return train_df, test_df


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Root Mean Squared Error computed strictly from actual targets and predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Mean Absolute Error computed strictly from actual targets and predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
    """Fraction of top-K recommended candidate items that are relevant (rating >= 4.0) in test set."""
    if k <= 0:
        return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return hits / k


def calculate_recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
    """Fraction of total relevant test items captured in top-K recommendations."""
    if not relevant_items:
        return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return hits / len(relevant_items)


def calculate_hit_rate_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
    """1.0 if at least one relevant test item is in top-K, 0.0 otherwise."""
    rec_k = recommended_items[:k]
    return 1.0 if len(set(rec_k).intersection(relevant_items)) > 0 else 0.0


def calculate_ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not relevant_items or k <= 0:
        return 0.0
    
    rec_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_items))))
    return dcg / idcg if idcg > 0 else 0.0


class ModelEvaluator:
    """
    Runs comparative evaluation across recommendation models using user-wise train/test splits (min 5 interactions).
    Performs live empirical metric calculation without hardcoded values or artificial multipliers.
    """

    def __init__(self, business_df: pd.DataFrame, reviews_df: pd.DataFrame):
        self.business_df = business_df.copy()
        self.reviews_df = reviews_df.copy()

    def evaluate_with_dataset_stats(
        self,
        k: int = 10,
        min_rating_positive: float = 4.0,
        min_interactions: int = 5
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs empirical model evaluation following the strict >=5 interaction protocol.
        
        Returns:
            1. Audit Metadata dictionary tracking all statutory dataset parameters.
            2. Benchmark Metrics DataFrame calculated live.
        """
        total_users = int(self.reviews_df["user_id"].nunique())
        total_items = int(self.business_df["business_id"].nunique())
        total_interactions = int(len(self.reviews_df))

        user_counts = self.reviews_df.groupby("user_id").size()
        eligible_users_count = int((user_counts >= min_interactions).sum())

        logger.info(f"Executing User-Wise Stratified Train/Test Split (min_interactions={min_interactions})...")
        train_df, test_df = user_wise_train_test_split(
            self.reviews_df,
            test_ratio=0.2,
            min_interactions=min_interactions
        )

        total_train_interactions = int(len(train_df))
        total_test_interactions = int(len(test_df))
        positive_test_df = test_df[test_df["stars"] >= min_rating_positive]
        total_positive_test_interactions = int(len(positive_test_df))

        test_users_with_positives = positive_test_df["user_id"].unique()
        evaluated_users_count = int(len(test_users_with_positives))

        avg_relevant_items_per_user = (
            total_positive_test_interactions / evaluated_users_count if evaluated_users_count > 0 else 0.0
        )

        all_item_ids = set(self.business_df["business_id"])
        candidate_sizes = []

        for user_id in test_users_with_positives:
            user_train_items = set(train_df[train_df["user_id"] == user_id]["business_id"])
            candidates = all_item_ids - user_train_items
            candidate_sizes.append(len(candidates))

        avg_candidates_per_user = float(np.mean(candidate_sizes)) if candidate_sizes else float(total_items)

        is_sample_validation = (evaluated_users_count < 10)

        results = []
        model_success_counts = {}

        # -------------------------------------------------------------
        # 1. Collaborative Filtering (SVD Matrix Factorization)
        # -------------------------------------------------------------
        svd_model = CollaborativeRecommender(algorithm="svd", n_factors=10)
        svd_model.fit(train_df)

        y_true_svd, y_pred_svd = [], []
        precisions_svd, recalls_svd, ndcgs_svd, hits_svd = [], [], [], []
        svd_success_users = 0

        for user_id in test_users_with_positives:
            user_test = test_df[test_df["user_id"] == user_id]
            for _, row in user_test.iterrows():
                y_true_svd.append(row["stars"])
                y_pred_svd.append(svd_model.predict_rating(user_id, row["business_id"]))

            relevant_items = set(user_test[user_test["stars"] >= min_rating_positive]["business_id"])
            recs = svd_model.recommend(user_id, top_k=k, exclude_seen=True)
            rec_item_ids = [r[0] for r in recs]

            if rec_item_ids:
                svd_success_users += 1

            precisions_svd.append(calculate_precision_at_k(rec_item_ids, relevant_items, k))
            recalls_svd.append(calculate_recall_at_k(rec_item_ids, relevant_items, k))
            hits_svd.append(calculate_hit_rate_at_k(rec_item_ids, relevant_items, k))
            ndcgs_svd.append(calculate_ndcg_at_k(rec_item_ids, relevant_items, k))

        model_success_counts["Collaborative (SVD)"] = svd_success_users
        rmse_svd = calculate_rmse(np.array(y_true_svd), np.array(y_pred_svd))
        mae_svd = calculate_mae(np.array(y_true_svd), np.array(y_pred_svd))

        results.append({
            "Model": "Collaborative (SVD)",
            "Evaluated Users": evaluated_users_count,
            "RMSE": round(rmse_svd, 4) if rmse_svd is not None else "N/A",
            "MAE": round(mae_svd, 4) if mae_svd is not None else "N/A",
            f"Precision@{k}": round(np.mean(precisions_svd) if precisions_svd else 0.0, 4),
            f"Recall@{k}": round(np.mean(recalls_svd) if recalls_svd else 0.0, 4),
            f"Hit_Rate@{k}": round(np.mean(hits_svd) if hits_svd else 0.0, 4),
            f"NDCG@{k}": round(np.mean(ndcgs_svd) if ndcgs_svd else 0.0, 4)
        })

        # -------------------------------------------------------------
        # 2. Content-Based Filtering (TF-IDF Vectorization)
        # -------------------------------------------------------------
        cb_model = ContentBasedRecommender()
        cb_model.fit(self.business_df)

        precisions_cb, recalls_cb, ndcgs_cb, hits_cb = [], [], [], []
        cb_success_users = 0

        for user_id in test_users_with_positives:
            user_test = test_df[test_df["user_id"] == user_id]
            user_train = train_df[train_df["user_id"] == user_id]
            
            relevant_items = set(user_test[user_test["stars"] >= min_rating_positive]["business_id"])
            if len(user_train) == 0:
                continue

            top_train_business = user_train.sort_values(by="stars", ascending=False).iloc[0]["business_id"]
            cb_recs = cb_model.recommend(top_train_business, top_k=k + len(user_train))
            
            train_items = set(user_train["business_id"])
            cb_item_ids = [b for b in cb_recs["business_id"] if b not in train_items][:k] if not cb_recs.empty else []

            if cb_item_ids:
                cb_success_users += 1

            precisions_cb.append(calculate_precision_at_k(cb_item_ids, relevant_items, k))
            recalls_cb.append(calculate_recall_at_k(cb_item_ids, relevant_items, k))
            hits_cb.append(calculate_hit_rate_at_k(cb_item_ids, relevant_items, k))
            ndcgs_cb.append(calculate_ndcg_at_k(cb_item_ids, relevant_items, k))

        model_success_counts["Content-Based (TF-IDF)"] = cb_success_users
        results.append({
            "Model": "Content-Based (TF-IDF)",
            "Evaluated Users": evaluated_users_count,
            "RMSE": "N/A",
            "MAE": "N/A",
            f"Precision@{k}": round(np.mean(precisions_cb) if precisions_cb else 0.0, 4),
            f"Recall@{k}": round(np.mean(recalls_cb) if recalls_cb else 0.0, 4),
            f"Hit_Rate@{k}": round(np.mean(hits_cb) if hits_cb else 0.0, 4),
            f"NDCG@{k}": round(np.mean(ndcgs_cb) if ndcgs_cb else 0.0, 4)
        })

        # -------------------------------------------------------------
        # 3. Hybrid Recommendation Engine (Weighted Multi-Attribute Fusion)
        # -------------------------------------------------------------
        hybrid_model = HybridRecommender(
            business_df=self.business_df,
            reviews_df=train_df,
            collab_recommender=svd_model,
            content_recommender=cb_model
        )

        y_true_hy, y_pred_hy = [], []
        precisions_hy, recalls_hy, ndcgs_hy, hits_hy = [], [], [], []
        hy_success_users = 0

        for user_id in test_users_with_positives:
            user_test = test_df[test_df["user_id"] == user_id]
            for _, row in user_test.iterrows():
                y_true_hy.append(row["stars"])
                pred_rating = svd_model.predict_rating(user_id, row["business_id"])
                y_pred_hy.append(pred_rating)

            relevant_items = set(user_test[user_test["stars"] >= min_rating_positive]["business_id"])
            hy_recs = hybrid_model.recommend_hybrid(user_id=user_id, top_k=k)
            hy_item_ids = hy_recs["business_id"].tolist() if not hy_recs.empty else []

            if hy_item_ids:
                hy_success_users += 1

            precisions_hy.append(calculate_precision_at_k(hy_item_ids, relevant_items, k))
            recalls_hy.append(calculate_recall_at_k(hy_item_ids, relevant_items, k))
            hits_hy.append(calculate_hit_rate_at_k(hy_item_ids, relevant_items, k))
            ndcgs_hy.append(calculate_ndcg_at_k(hy_item_ids, relevant_items, k))

        model_success_counts["Hybrid (Weighted Fusion)"] = hy_success_users
        rmse_hy = calculate_rmse(np.array(y_true_hy), np.array(y_pred_hy))
        mae_hy = calculate_mae(np.array(y_true_hy), np.array(y_pred_hy))

        results.append({
            "Model": "Hybrid (Weighted Fusion)",
            "Evaluated Users": evaluated_users_count,
            "RMSE": round(rmse_hy, 4) if rmse_hy is not None else "N/A",
            "MAE": round(mae_hy, 4) if mae_hy is not None else "N/A",
            f"Precision@{k}": round(np.mean(precisions_hy) if precisions_hy else 0.0, 4),
            f"Recall@{k}": round(np.mean(recalls_hy) if recalls_hy else 0.0, 4),
            f"Hit_Rate@{k}": round(np.mean(hits_hy) if hits_hy else 0.0, 4),
            f"NDCG@{k}": round(np.mean(ndcgs_hy) if ndcgs_hy else 0.0, 4)
        })

        # Dataset evaluation statistics summary dictionary
        stats_meta = {
            "total_users": total_users,
            "total_items": total_items,
            "total_interactions": total_interactions,
            "eligible_users": eligible_users_count,
            "evaluated_users": evaluated_users_count,
            "total_train_interactions": total_train_interactions,
            "total_test_interactions": total_test_interactions,
            "positive_test_interactions": total_positive_test_interactions,
            "avg_relevant_items_per_eval_user": round(avg_relevant_items_per_user, 2),
            "avg_candidate_items_per_user": round(avg_candidates_per_user, 1),
            "model_successful_users": model_success_counts,
            "is_sample_validation": is_sample_validation,
            "disclaimer": (
                "Pipeline Validation — Insufficient Sample for Model Comparison"
                if is_sample_validation else "Full Dataset Evaluation Result"
            )
        }

        return stats_meta, pd.DataFrame(results)

    def evaluate_all(self, k: int = 10, min_rating_positive: float = 4.0) -> pd.DataFrame:
        """Wrapper returning benchmark metrics DataFrame."""
        _, benchmark_df = self.evaluate_with_dataset_stats(k=k, min_rating_positive=min_rating_positive, min_interactions=5)
        return benchmark_df


def run_full_dataset_evaluation(
    raw_business_path: Optional[Path] = None,
    raw_review_path: Optional[Path] = None,
    k: int = 10,
    min_interactions: int = 5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Dedicated workflow to run live empirical model evaluation on full raw Yelp Academic Dataset files
    placed in data/raw/ or passed explicitly via arguments.
    """
    logger.info("Initializing Full Dataset Evaluation Workflow...")
    preproc = DataPreprocessor(raw_data_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)

    bus_path = raw_business_path or (RAW_DATA_DIR / "yelp_academic_dataset_business.csv")
    rev_path = raw_review_path or (RAW_DATA_DIR / "yelp_academic_dataset_review.csv")

    if not bus_path.exists() or not rev_path.exists():
        logger.warning(
            f"Full raw dataset files not found at '{bus_path}' / '{rev_path}'. "
            "Falling back to preprocessed demo sample."
        )

    bus_df = preproc.load_businesses(custom_path=bus_path if bus_path.exists() else None)
    rev_df = preproc.load_reviews(custom_path=rev_path if rev_path.exists() else None)

    evaluator = ModelEvaluator(bus_df, rev_df)
    stats, results_df = evaluator.evaluate_with_dataset_stats(
        k=k,
        min_rating_positive=4.0,
        min_interactions=min_interactions
    )

    if stats["is_sample_validation"]:
        print("\n" + "=" * 80)
        print("[PIPELINE VALIDATION NOTICE] Insufficient Sample for Model Comparison")
        print(f"Evaluated Users: {stats['evaluated_users']} | Total Test Interactions: {stats['total_test_interactions']}")
        print("Please place full Yelp Academic Dataset CSVs in data/raw/ to generate statistically representative results.")
        print("=" * 80 + "\n")

    return stats, results_df
