import sqlite3
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
import joblib
from src.01_data_processing.data_collector import (
    get_census_data, 
    clean_census_data, 
    get_airnow_current_observation, 
    get_airnow_forecast, 
    get_airnow_historical,
    clean_airnow_data
    get_latest_available_year
)

DB_NAME = 'urbanpulse.db'

def connect_to_database():
    """Connect to the SQLite database and return the connection object."""
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def load_geographic_data(conn):
    """Load geographic data from the database."""
    query = "SELECT * FROM geographic_data"
    return pd.read_sql_query(query, conn)

def load_demographic_data(conn):
    """Load demographic data from the database."""
    query = "SELECT * FROM demographic_data"
    return pd.read_sql_query(query, conn)

def load_air_quality_data(conn):
    """Load air quality data from the database."""
    query = "SELECT * FROM air_quality_data"
    return pd.read_sql_query(query, conn)

def analyze_demographic_trends(demographic_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze demographic trends from the collected multi-year data.
    """
    trends = {
        "population_growth_rate": calculate_growth_rate(demographic_data.set_index('year')['total_population'].to_dict()),
        "total_population": {
            "start": demographic_data['total_population'].iloc[0],
            "end": demographic_data['total_population'].iloc[-1],
            "change_percent": ((demographic_data['total_population'].iloc[-1] - demographic_data['total_population'].iloc[0]) / demographic_data['total_population'].iloc[0]) * 100
        },
        "median_household_income": {
            "start": demographic_data['median_household_income'].iloc[0],
            "end": demographic_data['median_household_income'].iloc[-1],
            "change_percent": ((demographic_data['median_household_income'].iloc[-1] - demographic_data['median_household_income'].iloc[0]) / demographic_data['median_household_income'].iloc[0]) * 100
        },
        "unemployment_rate": {
            "start": (demographic_data['unemployment_count'].iloc[0] / demographic_data['total_population'].iloc[0]) * 100,
            "end": (demographic_data['unemployment_count'].iloc[-1] / demographic_data['total_population'].iloc[-1]) * 100,
            "change": ((demographic_data['unemployment_count'].iloc[-1] / demographic_data['total_population'].iloc[-1]) - 
                       (demographic_data['unemployment_count'].iloc[0] / demographic_data['total_population'].iloc[0])) * 100
        },
        "bachelors_degree_rate": {
            "start": (demographic_data['bachelors_degree_count'].iloc[0] / demographic_data['total_population'].iloc[0]) * 100,
            "end": (demographic_data['bachelors_degree_count'].iloc[-1] / demographic_data['total_population'].iloc[-1]) * 100,
            "change": ((demographic_data['bachelors_degree_count'].iloc[-1] / demographic_data['total_population'].iloc[-1]) - 
                       (demographic_data['bachelors_degree_count'].iloc[0] / demographic_data['total_population'].iloc[0])) * 100
        }
    }
    return trends

def calculate_growth_rate(population_data: Dict[int, int]) -> float:
    """
    Calculate the average annual population growth rate.
    """
    years = sorted(population_data.keys())
    initial_pop = population_data[years[0]]
    final_pop = population_data[years[-1]]
    num_years = years[-1] - years[0]
    
    growth_rate = (final_pop / initial_pop) ** (1/num_years) - 1
    return growth_rate * 100  # Return as a percentage

def assess_environmental_impact(air_quality_data: List[Dict]) -> Dict[str, float]:
    """
    Assess environmental impact based on air quality data.
    """
    df = pd.DataFrame(air_quality_data)
    impact = {
        "avg_aqi": df['aqi'].mean(),
        "max_aqi": df['aqi'].max(),
        "days_unhealthy": len(df[df['category'] == 'Unhealthy']),
        "primary_pollutant": df['parameter'].mode().iloc[0]
    }
    return impact

def analyze_housing_market(demographic_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze housing market trends.
    """
    return {
        "median_home_value": demographic_data['median_home_value'].median(),
        "home_value_to_income_ratio": demographic_data['median_home_value'].median() / demographic_data['median_household_income'].median(),
    }

def analyze_education_level(demographic_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze education levels in the population.
    """
    total_population = demographic_data['total_population'].sum()
    bachelors_count = demographic_data['bachelors_degree_count'].sum()
    return {
        "bachelors_degree_rate": (bachelors_count / total_population) * 100,
    }

def analyze_economic_indicators(demographic_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze economic indicators.
    """
    return {
        "median_household_income": demographic_data['median_household_income'].median(),
        "unemployment_rate": (demographic_data['unemployment_count'].sum() / demographic_data['total_population'].sum()) * 100,
    }

def analyze_air_quality_trends(air_quality_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze trends in air quality data.
    """
    df = pd.DataFrame(air_quality_data)
    return {
        "avg_aqi_by_parameter": df.groupby('parameter')['aqi'].mean().to_dict(),
        "aqi_trend": df.groupby('date')['aqi'].mean().to_dict(),
        "most_common_category": df['category'].mode().iloc[0],
    }

def load_or_collect_data():
    try:
        demographic_data = pd.read_csv('processed_demographic_data.csv')
        air_quality_data = pd.read_csv('processed_air_quality_data.csv')
        adaptive_grid = gpd.read_file('adaptive_grid.geojson')
        print("Loaded processed data from files.")
    except FileNotFoundError:
        print("Processed data files not found. Collecting new data...")
        start_year = 2010
        end_year = get_latest_available_year()
        dataset = "acs/acs5"
        variables = [
            "NAME",
            "B01003_001E",  # Total population
            "B19013_001E",  # Median household income
            "B25077_001E",  # Median home value
            "B23025_005E",  # Unemployment count
            "B15003_022E",  # Bachelor's degree count
        ]
        geography = "place:05320"  # Bayonne city, NJ
        zip_code = "07002"

        census_df = get_census_data(start_year, end_year, dataset, variables, geography)
        demographic_data = clean_census_data(census_df)

        current_air_data = get_airnow_current_observation(zip_code)
        forecast_air_data = get_airnow_forecast(zip_code)
        historical_air_data = get_airnow_historical(zip_code, pd.Timestamp.now() - pd.Timedelta(days=1))

        air_quality_data = []
        for data, data_type in [(current_air_data, 'current'), (forecast_air_data, 'forecast'), (historical_air_data, 'historical')]:
            if data:
                air_quality_data.extend(clean_airnow_data(data, data_type))
        
        air_quality_data = pd.DataFrame(air_quality_data)

        # Note: We don't have the adaptive_grid here as it's created in the data collection process.
        # For this example, we'll create a dummy GeoDataFrame.
        adaptive_grid = gpd.GeoDataFrame({'geometry': [None]})

        # Save the collected data
        demographic_data.to_csv('processed_demographic_data.csv', index=False)
        air_quality_data.to_csv('processed_air_quality_data.csv', index=False)
        adaptive_grid.to_file('adaptive_grid.geojson', driver='GeoJSON')

    return demographic_data, air_quality_data, adaptive_grid

def split_data(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

def analyze_population_density(demographic_data):
    """Analyze and visualize population density."""
    plt.figure(figsize=(10, 6))
    plt.bar(demographic_data['city'], demographic_data['total_population'], color='skyblue')
    plt.title('Total Population by City')
    plt.xlabel('City')
    plt.ylabel('Total Population')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_air_quality(air_quality_data):
    """Analyze and visualize air quality data."""
    air_quality_data['date'] = pd.to_datetime(air_quality_data['date'])
    plt.figure(figsize=(10, 6))
    plt.plot(air_quality_data['date'], air_quality_data['aqi'], marker='o', linestyle='-', color='orange')
    plt.title('Air Quality Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function for testing the data analysis module with real data from collectors.
    """
    # Connect to the database
    conn = connect_to_database()
    if conn is None:
        return  # Exit if the connection failed

    # Load data
    geographic_data = load_geographic_data(conn)
    demographic_data = load_demographic_data(conn)
    air_quality_data = load_air_quality_data(conn)

    # Perform analysis
    analyze_population_density(demographic_data)
    analyze_air_quality(air_quality_data)

    # Close the database connection
    conn.close()

    if demographic_data is None or air_quality_data.empty:
        print("Failed to retrieve data.")
        return

    # Apply 80/20 split
    demographic_train, demographic_test = split_data(demographic_data)
    air_quality_train, air_quality_test = split_data(air_quality_data)

    # Run analysis functions
    print("Demographic Trends:")
    print(analyze_demographic_trends(demographic_train))

    print("\nEnvironmental Impact:")
    print(assess_environmental_impact(air_quality_train.to_dict('records')))

    print("\nHousing Market:")
    print(analyze_housing_market(demographic_train))

    print("\nEducation Level:")
    print(analyze_education_level(demographic_train))

    print("\nEconomic Indicators:")
    print(analyze_economic_indicators(demographic_train))

    print("\nAir Quality Trends:")
    print(analyze_air_quality_trends(air_quality_train.to_dict('records')))

    print("\nAdaptive Grid Information:")
    print(f"Number of cells: {len(adaptive_grid)}")

    # Prepare data for ML model (to be used in 02_ml_model)
    ml_data = {
        'demographic_train': demographic_train,
        'demographic_test': demographic_test,
        'air_quality_train': air_quality_train,
        'air_quality_test': air_quality_test,
        'adaptive_grid': adaptive_grid
    }
    
    # Save the data
    joblib.dump(ml_data, 'ml_data.joblib')

if __name__ == "__main__":
    main()