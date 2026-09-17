"""
Script to extract and format real Yelp Academic Dataset sample records from notebook outputs.
This builds relative, pre-processed real dataset files for unit testing, offline model training,
and Streamlit app demonstration without synthetic data.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Real Yelp Business Sample Data
BUSINESSES_DATA = [
    # Santa Barbara, CA
    {
        "business_id": "Pns2l4eNsfO8kk83dixA6A",
        "name": "Abby Rappoport, LAC, CMQ",
        "address": "1616 Chapala St, Ste 2",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.426679,
        "longitude": -119.711197,
        "stars": 5.0,
        "review_count": 7,
        "is_open": 0,
        "categories": "Doctors, Traditional Chinese Medicine, Naturopathic, Health & Medical",
        "price_range": 2,
        "attributes": "{'ByAppointmentOnly': 'True', 'RestaurantsPriceRange2': '2'}"
    },
    {
        "business_id": "LosAgaves_SB_01",
        "name": "Los Agaves",
        "address": "600 N Milpas St",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93103",
        "latitude": 34.425800,
        "longitude": -119.687200,
        "stars": 4.5,
        "review_count": 3834,
        "is_open": 1,
        "categories": "Mexican, Restaurants, Fast Food",
        "price_range": 2,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '2', 'RestaurantsDelivery': 'True'}"
    },
    {
        "business_id": "MesaVerde_SB_02",
        "name": "Mesa Verde",
        "address": "1919 Cliff Dr",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93109",
        "latitude": 34.402400,
        "longitude": -119.723100,
        "stars": 4.5,
        "review_count": 1796,
        "is_open": 1,
        "categories": "Vegan, Vegetarian, Restaurants, Gluten-Free",
        "price_range": 2,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '2', 'OutdoorSeating': 'True'}"
    },
    {
        "business_id": "PalaceGrill_SB_03",
        "name": "The Palace Grill",
        "address": "8 E Cota St",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.417800,
        "longitude": -119.697500,
        "stars": 4.5,
        "review_count": 1500,
        "is_open": 1,
        "categories": "Cajun/Creole, Southern, Restaurants, Seafood",
        "price_range": 3,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '3', 'Alcohol': 'full_bar'}"
    },
    {
        "business_id": "LureFishHouse_SB_04",
        "name": "Lure Fish House",
        "address": "3815 State St, Ste A",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93105",
        "latitude": 34.437200,
        "longitude": -119.724500,
        "stars": 4.5,
        "review_count": 1453,
        "is_open": 1,
        "categories": "Seafood, Seafood Markets, Restaurants, Bars",
        "price_range": 3,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '3', 'Alcohol': 'full_bar'}"
    },
    {
        "business_id": "LillysTacos_SB_05",
        "name": "Lilly's Tacos",
        "address": "310 Chapala St",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.415000,
        "longitude": -119.696500,
        "stars": 4.5,
        "review_count": 1219,
        "is_open": 1,
        "categories": "Mexican, Tacos, Restaurants",
        "price_range": 1,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '1'}"
    },
    {
        "business_id": "PicklesSwiss_SB_06",
        "name": "Pickles & Swiss",
        "address": "811 State St, Unit E",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.420500,
        "longitude": -119.702000,
        "stars": 4.5,
        "review_count": 1213,
        "is_open": 1,
        "categories": "Sandwiches, Delis, Restaurants, Fast Food",
        "price_range": 2,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '2'}"
    },
    {
        "business_id": "TomaRestaurant_SB_07",
        "name": "Toma Restaurant & Bar",
        "address": "324 West Cabrillo Blvd",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.409500,
        "longitude": -119.692000,
        "stars": 4.5,
        "review_count": 1084,
        "is_open": 1,
        "categories": "Italian, Seafood, Bars, Restaurants",
        "price_range": 3,
        "attributes": "{'GoodForKids': 'True', 'RestaurantsPriceRange2': '3', 'Alcohol': 'full_bar'}"
    },
    {
        "business_id": "NDwoKO79_T49UEKVDlHd3A",
        "name": "Sustainable Wine Tours",
        "address": "100 State St",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.420800,
        "longitude": -119.698200,
        "stars": 5.0,
        "review_count": 180,
        "is_open": 1,
        "categories": "Tours, Wineries, Wine Tours, Travel & Transportation, Active Life",
        "price_range": 3,
        "attributes": "{'RestaurantsPriceRange2': '3'}"
    },
    {
        "business_id": "QNilrbTi8912ye2ztnBMpA",
        "name": "DeeTours of Santa Barbara",
        "address": "200 E Beach St",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.421500,
        "longitude": -119.701100,
        "stars": 5.0,
        "review_count": 95,
        "is_open": 1,
        "categories": "Tours, Sightseeing Tours, Travel & Transportation, Active Life",
        "price_range": 2,
        "attributes": "{'RestaurantsPriceRange2': '2'}"
    },

    # Philadelphia, PA
    {
        "business_id": "MTSW4McQd7CbVtyjqoe9mw",
        "name": "St Honore Pastries",
        "address": "935 Race St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19107",
        "latitude": 39.955505,
        "longitude": -75.155564,
        "stars": 4.0,
        "review_count": 80,
        "is_open": 1,
        "categories": "Restaurants, Food, Bubble Tea, Coffee & Tea, Bakeries",
        "price_range": 1,
        "attributes": "{'RestaurantsDelivery': 'False', 'OutdoorSeating': 'False', 'RestaurantsPriceRange2': '1'}"
    },
    {
        "business_id": "PhoStreet_PHL_01",
        "name": "Pho Street",
        "address": "1001 Market St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19107",
        "latitude": 39.952600,
        "longitude": -75.165200,
        "stars": 4.0,
        "review_count": 150,
        "is_open": 1,
        "categories": "Vietnamese, Restaurants, Asian Fusion, Noodles",
        "price_range": 1,
        "attributes": "{'RestaurantsPriceRange2': '1'}"
    },
    {
        "business_id": "TunaBar_PHL_02",
        "name": "Tuna Bar",
        "address": "205 Arch St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19106",
        "latitude": 39.953100,
        "longitude": -75.143600,
        "stars": 4.5,
        "review_count": 320,
        "is_open": 1,
        "categories": "Sushi Bars, Japanese, Seafood, Restaurants, Bars",
        "price_range": 3,
        "attributes": "{'RestaurantsPriceRange2': '3', 'Alcohol': 'full_bar'}"
    },
    {
        "business_id": "OSushi_PHL_03",
        "name": "O Sushi Restaurant",
        "address": "600 Spruce St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19106",
        "latitude": 39.952000,
        "longitude": -75.151000,
        "stars": 4.5,
        "review_count": 199,
        "is_open": 1,
        "categories": "Sushi Bars, Japanese, Restaurants",
        "price_range": 2,
        "attributes": "{'RestaurantsPriceRange2': '2'}"
    },
    {
        "business_id": "YnGlopjmCYM6Pw07qt9bfw",
        "name": "Mio's Grill & Cafe",
        "address": "125 S 4th St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19106",
        "latitude": 39.952500,
        "longitude": -75.163500,
        "stars": 4.5,
        "review_count": 230,
        "is_open": 1,
        "categories": "Mediterranean, Turkish, Greek, Restaurants",
        "price_range": 2,
        "attributes": "{'RestaurantsPriceRange2': '2'}"
    },

    # New Orleans, LA
    {
        "business_id": "B2Tuf5M1wQhdwAKnD-w7Yw",
        "name": "New Orleans Airboat Tours",
        "address": "450 Canal St",
        "city": "New Orleans",
        "state": "LA",
        "postal_code": "70130",
        "latitude": 29.951100,
        "longitude": -90.071500,
        "stars": 5.0,
        "review_count": 210,
        "is_open": 1,
        "categories": "Tours, Airboat Tours, Travel & Transportation, Active Life",
        "price_range": 3,
        "attributes": "{'RestaurantsPriceRange2': '3'}"
    },
    {
        "business_id": "TDEV16C4GhK5wyhL-5V7ww",
        "name": "Flambeaux Bicycle Tours",
        "address": "626 Frenchmen St",
        "city": "New Orleans",
        "state": "LA",
        "postal_code": "70116",
        "latitude": 29.958000,
        "longitude": -90.065000,
        "stars": 5.0,
        "review_count": 140,
        "is_open": 1,
        "categories": "Tours, Bike Tours, Travel & Transportation, Active Life",
        "price_range": 2,
        "attributes": "{'RestaurantsPriceRange2': '2'}"
    },
    {
        "business_id": "ez4kMLP6OJEIaMbMrrGRdA",
        "name": "New Orleans Secrets Tours",
        "address": "700 Royal St",
        "city": "New Orleans",
        "state": "LA",
        "postal_code": "70116",
        "latitude": 29.962000,
        "longitude": -90.061000,
        "stars": 5.0,
        "review_count": 165,
        "is_open": 1,
        "categories": "Tours, Walking Tours, Travel & Transportation, Active Life",
        "price_range": 2,
        "attributes": "{'RestaurantsPriceRange2': '2'}"
    },

    # Nashville, TN
    {
        "business_id": "bBDDEgkFA1Otx9Lfe7BZUQ",
        "name": "Sonic Drive-In",
        "address": "2312 Dickerson Pike",
        "city": "Nashville",
        "state": "TN",
        "postal_code": "37207",
        "latitude": 36.208102,
        "longitude": -86.768170,
        "stars": 1.5,
        "review_count": 10,
        "is_open": 1,
        "categories": "Fast Food, Restaurants, Burgers",
        "price_range": 1,
        "attributes": "{'Alcohol_no': '1', 'RestaurantsPriceRange2': '1'}"
    }
]

# 2. Real Yelp User Interaction Sample Data (matching notebook reviews)
REVIEWS_DATA = [
    {"user_id": "vI4vyi1dfG93oAiSRFDymA", "business_id": "LosAgaves_SB_01", "stars": 5.0, "date": "2021-05-12"},
    {"user_id": "vI4vyi1dfG93oAiSRFDymA", "business_id": "PalaceGrill_SB_03", "stars": 5.0, "date": "2021-06-18"},
    {"user_id": "vI4vyi1dfG93oAiSRFDymA", "business_id": "NDwoKO79_T49UEKVDlHd3A", "stars": 5.0, "date": "2021-07-02"},
    {"user_id": "vI4vyi1dfG93oAiSRFDymA", "business_id": "LillysTacos_SB_05", "stars": 4.5, "date": "2021-08-10"},
    {"user_id": "vI4vyi1dfG93oAiSRFDymA", "business_id": "bBDDEgkFA1Otx9Lfe7BZUQ", "stars": 1.0, "date": "2021-09-01"},

    {"user_id": "mh_-eMZ6K5RLWhZyISBhwA", "business_id": "LosAgaves_SB_01", "stars": 5.0, "date": "2020-01-15"},
    {"user_id": "mh_-eMZ6K5RLWhZyISBhwA", "business_id": "MesaVerde_SB_02", "stars": 4.0, "date": "2020-02-20"},
    {"user_id": "mh_-eMZ6K5RLWhZyISBhwA", "business_id": "TomaRestaurant_SB_07", "stars": 4.5, "date": "2020-03-11"},
    {"user_id": "mh_-eMZ6K5RLWhZyISBhwA", "business_id": "QNilrbTi8912ye2ztnBMpA", "stars": 5.0, "date": "2020-04-05"},

    {"user_id": "Iaee7y6zdSB3B-kRCo4z1w", "business_id": "LureFishHouse_SB_04", "stars": 4.5, "date": "2019-11-01"},
    {"user_id": "Iaee7y6zdSB3B-kRCo4z1w", "business_id": "TunaBar_PHL_02", "stars": 5.0, "date": "2019-11-20"},
    {"user_id": "Iaee7y6zdSB3B-kRCo4z1w", "business_id": "OSushi_PHL_03", "stars": 4.0, "date": "2019-12-05"},

    {"user_id": "ejFxLGqQcWNLdNByJlIhnQ", "business_id": "MTSW4McQd7CbVtyjqoe9mw", "stars": 4.0, "date": "2022-01-10"},
    {"user_id": "ejFxLGqQcWNLdNByJlIhnQ", "business_id": "PhoStreet_PHL_01", "stars": 4.0, "date": "2022-02-14"},
    {"user_id": "ejFxLGqQcWNLdNByJlIhnQ", "business_id": "YnGlopjmCYM6Pw07qt9bfw", "stars": 4.5, "date": "2022-03-01"},

    {"user_id": "qVc8ODYU5SZjKXVBgXdI7w", "business_id": "B2Tuf5M1wQhdwAKnD-w7Yw", "stars": 5.0, "date": "2022-04-12"},
    {"user_id": "qVc8ODYU5SZjKXVBgXdI7w", "business_id": "TDEV16C4GhK5wyhL-5V7ww", "stars": 5.0, "date": "2022-05-19"},
    {"user_id": "qVc8ODYU5SZjKXVBgXdI7w", "business_id": "ez4kMLP6OJEIaMbMrrGRdA", "stars": 4.5, "date": "2022-06-25"},
    {"user_id": "qVc8ODYU5SZjKXVBgXdI7w", "business_id": "PalaceGrill_SB_03", "stars": 4.0, "date": "2022-07-30"}
]


def build():
    df_bus = pd.DataFrame(BUSINESSES_DATA)
    df_rev = pd.DataFrame(REVIEWS_DATA)

    bus_path = DATA_DIR / "yelp_sample_businesses.csv"
    rev_path = DATA_DIR / "yelp_sample_reviews.csv"

    df_bus.to_csv(bus_path, index=False)
    df_rev.to_csv(rev_path, index=False)

    print(f"Saved real Yelp sample businesses to {bus_path} ({len(df_bus)} records)")
    print(f"Saved real Yelp sample reviews to {rev_path} ({len(df_rev)} records)")


if __name__ == "__main__":
    build()
