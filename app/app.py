"""
Travel Recommendation Platform — Interactive Streamlit Application.
Multi-page Web App with Discover Explorer, User Personalization Portal, and Model Benchmarks.
"""

import sys
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.cold_start import ColdStartRecommender
from src.hybrid import HybridRecommender
from src.ranking import RecommendationRanker
from src.evaluation import ModelEvaluator
from src.utils import log_user_feedback

# Streamlit Page Configuration
st.set_page_config(
    page_title="TravelRecommender | AI Travel Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .rec-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .badge-cat {
        background-color: #0284c7;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-price {
        background-color: #16a34a;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-score {
        background-color: #9333ea;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pipeline-warning {
        background: rgba(234, 179, 8, 0.1);
        border-left: 4px solid #eab308;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_and_models():
    """Load preprocessed data and fit core recommender models."""
    prep = DataPreprocessor()
    try:
        bus_df = prep.load_businesses()
        rev_df = prep.load_reviews()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None, None, None, None, None

    cb_model = ContentBasedRecommender()
    cb_model.fit(bus_df)

    cf_model = CollaborativeRecommender(algorithm="svd", n_factors=10)
    cf_model.fit(rev_df)

    hybrid_model = HybridRecommender(
        business_df=bus_df,
        reviews_df=rev_df,
        collab_recommender=cf_model,
        content_recommender=cb_model
    )
    ranker = RecommendationRanker()

    return bus_df, rev_df, cb_model, cf_model, hybrid_model, ranker


# Main Initialization
data_bundle = load_data_and_models()
if data_bundle[0] is None:
    st.stop()

bus_df, rev_df, cb_model, cf_model, hybrid_model, ranker = data_bundle

# Navigation Sidebar
st.sidebar.title("✈️ Travel Recommender")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Navigation Page:",
    ["🔍 Discover & Filter", "👤 Personalized Portal", "📊 Model Insights & Evaluation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Hybrid ML Fusion (SVD + TF-IDF + Haversine)")

# -------------------------------------------------------------------
# PAGE 1 — DISCOVER & FILTER (COLD START / HYBRID EXPLORER)
# -------------------------------------------------------------------
if page == "🔍 Discover & Filter":
    st.title("🔍 Discover Travel Recommendations")
    st.subheader("Filter places by destination, preferences, budget, and distance radius")

    col1, col2, col3 = st.columns(3)

    with col1:
        cities = ["All"] + sorted(bus_df["city"].unique().tolist())
        selected_city = st.selectbox("Destination City:", cities)
        min_rating = st.slider("Minimum Rating:", 1.0, 5.0, 4.0, step=0.5)

    with col2:
        all_categories = ["Restaurants", "Tours", "Seafood", "Mexican", "Sushi Bars", "Vegan", "Bars", "Fast Food"]
        selected_categories = st.multiselect("Preferred Categories:", all_categories, default=["Restaurants"])

        budget_map = {"$ (Budget)": 1, "$$ (Moderate)": 2, "$$$ (Expensive)": 3, "$$$$ (Luxury)": 4}
        selected_budget_str = st.selectbox("Max Budget Tier:", list(budget_map.keys()), index=1)
        selected_budget = budget_map[selected_budget_str]

    with col3:
        distance_map = {"5 km": 5.0, "10 km": 10.0, "20 km": 20.0, "50 km": 50.0, "Unlimited": None}
        selected_distance_str = st.radio("Max Distance Radius:", list(distance_map.keys()), index=3)
        max_dist_km = distance_map[selected_distance_str]

        # Dynamically compute city center coordinates from dataset
        city_coords = bus_df[bus_df["latitude"] != 0.0].groupby("city")[["latitude", "longitude"]].mean().reset_index()
        loc_options = ["None (No Geolocation Filter)"] + [f"Demo User Location: {row['city']}" for _, row in city_coords.iterrows()]
        
        selected_loc_str = st.selectbox("Current User Location:", loc_options, index=1 if len(loc_options) > 1 else 0)

        if selected_loc_str != "None (No Geolocation Filter)":
            loc_city = selected_loc_str.replace("Demo User Location: ", "").strip()
            match_row = city_coords[city_coords["city"] == loc_city]
            if not match_row.empty:
                user_lat = float(match_row.iloc[0]["latitude"])
                user_lon = float(match_row.iloc[0]["longitude"])
            else:
                user_lat, user_lon = None, None
        else:
            user_lat, user_lon = None, None

    st.markdown("---")
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Computing multi-attribute hybrid recommendations..."):
            recs_df = hybrid_model.recommend_hybrid(
                city=selected_city if selected_city != "All" else None,
                categories=selected_categories,
                target_budget=selected_budget,
                min_rating=min_rating,
                user_lat=user_lat,
                user_lon=user_lon,
                max_distance_km=max_dist_km,
                top_k=10
            )

            ranked_recs = ranker.rank_and_explain(
                candidates_df=recs_df,
                preferred_categories=selected_categories,
                target_budget=selected_budget
            )

        if ranked_recs.empty:
            st.warning("No businesses matched your exact filter criteria. Try expanding distance or budget.")
        else:
            st.success(f"Found top {len(ranked_recs)} recommended places:")

            for idx, row in ranked_recs.iterrows():
                with st.container():
                    price_str = "$" * int(row.get("price_range", 2))
                    dist_str = f"{row.get('distance_km', 0.0):.1f} km away" if row.get("distance_km", 0) > 0 else "Location N/A"
                    match_score = float(row.get("match_score", 0.0))

                    st.markdown(f"""
                    <div class="rec-card">
                        <h3>{idx + 1}. {row['name']}</h3>
                        <p><strong>City:</strong> {row['city']}, {row['state']} | <strong>Address:</strong> {row['address']}</p>
                        <p>
                            <span class="badge-cat">{row['categories'].split(',')[0]}</span>
                            <span class="badge-price">{price_str}</span>
                            <span class="badge-score">Match Score: {match_score:.2f} / 1.00</span>
                        </p>
                        <p>⭐ <strong>{row['stars']} / 5.0</strong> ({row['review_count']} reviews) | 📍 <strong>{dist_str}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                    reasons = row.get("explanation_reasons", [])
                    if reasons:
                        st.markdown("**Why Recommended:**")
                        for reason in reasons:
                            st.markdown(f"- {reason}")

                    fb_col1, fb_col2, _ = st.columns([1, 1, 4])
                    with fb_col1:
                        if st.button(f"👍 Like", key=f"like_{row['business_id']}"):
                            log_user_feedback("anonymous_user", row["business_id"], "like")
                            st.toast(f"Liked {row['name']}!")
                    with fb_col2:
                        if st.button(f"👎 Dislike", key=f"dislike_{row['business_id']}"):
                            log_user_feedback("anonymous_user", row["business_id"], "dislike")
                            st.toast(f"Feedback saved for {row['name']}.")

                    st.markdown("---")

# -------------------------------------------------------------------
# PAGE 2 — PERSONALIZED USER PORTAL
# -------------------------------------------------------------------
elif page == "👤 Personalized Portal":
    st.title("👤 Personalized User Portal")
    st.subheader("Select a user to view interaction history and collaborative recommendations")

    available_users = sorted(rev_df["user_id"].unique().tolist())
    selected_user = st.selectbox("Select Active User ID:", available_users, index=0)

    user_reviews = rev_df[rev_df["user_id"] == selected_user]

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📜 Past Interactions")
        st.caption(f"Total reviews submitted: {len(user_reviews)}")
        
        merged_user_revs = pd.merge(user_reviews, bus_df, on="business_id", how="left", suffixes=("", "_bus"))
        for _, row in merged_user_revs.iterrows():
            b_name = row.get("name", row["business_id"])
            stars_val = row.get("stars", row.get("stars_bus", "N/A"))
            cat_str = str(row.get("categories", "N/A")).split(",")[0]
            st.markdown(f"- **{b_name}**: ⭐ **{stars_val}** ({cat_str})")

    with col_right:
        st.markdown("### 🎯 Personalized Recommendations")
        st.caption("Collaborative SVD & Hybrid Candidate Scoring (Excludes Already Rated Places)")

        user_recs = hybrid_model.recommend_hybrid(user_id=selected_user, top_k=5)
        ranked_user_recs = ranker.rank_and_explain(candidates_df=user_recs, user_id=selected_user)

        if ranked_user_recs.empty:
            st.info("No personalized recommendations generated.")
        else:
            for idx, row in ranked_user_recs.iterrows():
                match_score = float(row.get("match_score", 0.0))
                st.markdown(f"**{idx + 1}. {row['name']}**")
                st.caption(f"Category: {row['categories'].split(',')[0]} | Rating: ⭐ {row['stars']} | Match Score: {match_score:.2f} / 1.00")
                reasons = row.get("explanation_reasons", [])
                if reasons:
                    st.markdown(f"_Reason: {reasons[0]}_")
                st.markdown("---")

# -------------------------------------------------------------------
# PAGE 3 — MODEL INSIGHTS & BENCHMARKS
# -------------------------------------------------------------------
elif page == "📊 Model Insights & Evaluation":
    st.title("📊 Model Insights & Scientific Evaluation Benchmarks")
    st.caption("Empirical performance evaluation & dataset sample audit statistics breakdown")

    st.markdown("""
    <div class="pipeline-warning">
        ⚠️ <strong>Pipeline Validation Disclaimer:</strong><br>
        The evaluation metrics displayed below are computed live on the bundled <strong>Yelp Demo Sample Dataset</strong> 
        to validate the scientific evaluation pipeline, train/test splitting logic, and candidate set exclusion.
        These sample results demonstrate software correctness and should <strong>not</strong> be interpreted as 
        representative final model performance on the full multi-GB Yelp Academic Dataset.
    </div>
    """, unsafe_allow_html=True)

    evaluator = ModelEvaluator(bus_df, rev_df)
    stats, eval_results = evaluator.evaluate_with_dataset_stats(k=10, min_rating_positive=4.0, min_interactions=5)

    st.markdown("### 🔍 Dataset & Splitting Audit Parameters")
    
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Users in Dataset", f"{stats['total_users']:,}")
        st.metric("Eligible Users (≥5 reviews)", f"{stats['eligible_users']:,}")
        st.metric("Evaluated Users (≥1 pos test)", f"{stats['evaluated_users']:,}")
    with s2:
        st.metric("Total Businesses / Items", f"{stats['total_items']:,}")
        st.metric("Total Interactions", f"{stats['total_interactions']:,}")
        st.metric("Avg Candidates / User", f"{stats['avg_candidate_items_per_user']:.1f}")
    with s3:
        st.metric("Training Interactions", f"{stats['total_train_interactions']:,}")
        st.metric("Test Interactions", f"{stats['total_test_interactions']:,}")
        st.metric("Positive Test Interactions (≥4.0★)", f"{stats['positive_test_interactions']:,}")
    with s4:
        st.metric("Avg Relevant Items / User", f"{stats['avg_relevant_items_per_eval_user']:.2f}")
        for m_name, count in stats['model_successful_users'].items():
            st.caption(f"{m_name}: {count} users recs generated")

    st.markdown("---")

    st.markdown("### 🏆 Live Empirical Benchmark Results")
    st.dataframe(eval_results, use_container_width=True)

    st.markdown("### 📈 Metric Visualizations")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_prec = px.bar(
            eval_results,
            x="Model",
            y="Precision@10",
            title="Precision@10 Comparison",
            color="Model",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"]
        )
        st.plotly_chart(fig_prec, use_container_width=True)

    with col_chart2:
        fig_ndcg = px.bar(
            eval_results,
            x="Model",
            y="NDCG@10",
            title="NDCG@10 Rank Order Quality Comparison",
            color="Model",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"]
        )
        st.plotly_chart(fig_ndcg, use_container_width=True)
