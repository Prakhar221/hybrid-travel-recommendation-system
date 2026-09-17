"""
Preprocessing module for Travel Recommendation System.
Ingests Yelp Academic Dataset (full or sample subset) and cleans businesses, categories,
attributes, price ranges, geolocations, and ratings.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.utils import PROCESSED_DATA_DIR, RAW_DATA_DIR, logger


def parse_price_range(attr_str) -> int:
    """Extract price range integer (1-4) from Yelp attributes field."""
    if pd.isna(attr_str) or attr_str is None:
        return 2  # default moderate price range
    
    if isinstance(attr_str, dict):
        attr_dict = attr_str
    else:
        try:
            attr_dict = ast.literal_eval(str(attr_str))
        except Exception:
            try:
                attr_dict = json.loads(str(attr_str).replace("'", '"'))
            except Exception:
                attr_dict = {}

    if not isinstance(attr_dict, dict):
        return 2

    price = attr_dict.get("RestaurantsPriceRange2", 2)
    try:
        price_int = int(price)
        return max(1, min(4, price_int))
    except (ValueError, TypeError):
        return 2


def create_content_string(row: pd.Series) -> str:
    """Combine name, categories, city, state, and key attributes into content text for TF-IDF."""
    name = str(row.get("name", ""))
    categories = str(row.get("categories", ""))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    attributes = str(row.get("attributes", ""))

    content_tokens = [name, categories, city, state, attributes]
    clean_text = " ".join([token for token in content_tokens if token and token != "nan"])
    return clean_text.lower()


class DataPreprocessor:
    """Pipeline for loading, cleaning, and structuring Yelp dataset files."""

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)

    def load_businesses(self, custom_path: Optional[Path] = None) -> pd.DataFrame:
        """Load business data from raw or processed CSV."""
        if custom_path and Path(custom_path).exists():
            filepath = Path(custom_path)
        elif (self.raw_data_dir / "yelp_academic_dataset_business.csv").exists():
            filepath = self.raw_data_dir / "yelp_academic_dataset_business.csv"
        elif (self.processed_dir / "yelp_sample_businesses.csv").exists():
            filepath = self.processed_dir / "yelp_sample_businesses.csv"
        else:
            raise FileNotFoundError("No business dataset found in raw or processed directory.")

        logger.info(f"Loading business data from {filepath}")
        df = pd.read_csv(filepath)
        return self.clean_businesses(df)

    def clean_businesses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and structure raw business dataframe."""
        df = df.copy()

        # Essential columns fillna
        df["name"] = df["name"].fillna("Unknown Business")
        df["city"] = df["city"].fillna("Unknown City")
        df["state"] = df["state"].fillna("Unknown State")
        df["categories"] = df["categories"].fillna("Uncategorized")
        df["stars"] = df["stars"].fillna(df["stars"].mean() if len(df) > 0 else 3.5)
        df["review_count"] = df["review_count"].fillna(0).astype(int)
        df["latitude"] = df["latitude"].fillna(0.0)
        df["longitude"] = df["longitude"].fillna(0.0)

        # Ensure price_range exists
        if "price_range" not in df.columns or df["price_range"].isnull().all():
            df["price_range"] = df["attributes"].apply(parse_price_range)
        else:
            df["price_range"] = df["price_range"].fillna(2).astype(int)

        # Generate combined content text for TF-IDF
        df["content"] = df.apply(create_content_string, axis=1)

        # Reset index
        df = df.drop_duplicates(subset=["business_id"]).reset_index(drop=True)
        return df

    def load_reviews(self, custom_path: Optional[Path] = None) -> pd.DataFrame:
        """Load review interaction data."""
        if custom_path and Path(custom_path).exists():
            filepath = Path(custom_path)
        elif (self.raw_data_dir / "yelp_academic_dataset_review.csv").exists():
            filepath = self.raw_data_dir / "yelp_academic_dataset_review.csv"
        elif (self.processed_dir / "yelp_sample_reviews.csv").exists():
            filepath = self.processed_dir / "yelp_sample_reviews.csv"
        else:
            raise FileNotFoundError("No review dataset found in raw or processed directory.")

        logger.info(f"Loading review interactions from {filepath}")
        df = pd.read_csv(filepath)
        return self.clean_reviews(df)

    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean user rating interactions."""
        df = df.copy()
        required_cols = ["user_id", "business_id", "stars"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required review column: '{col}'")

        df = df.dropna(subset=["user_id", "business_id", "stars"])
        df["stars"] = df["stars"].astype(float)
        df = df.drop_duplicates(subset=["user_id", "business_id"]).reset_index(drop=True)
        return df
