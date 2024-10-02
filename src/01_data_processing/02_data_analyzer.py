import pandas as pd
import numpy as np
from typing import Dict, List, Any

def analyze_demographic_trends(demographic_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze demographic trends from the collected data.
    """
    population_data = demographic_data.set_index('year')['total_population'].to_dict()
    trends = {
        "population_growth_rate": calculate_growth_rate(population_data),
        "current_population": demographic_data['total_population'].iloc[-1],
        "median_age": demographic_data['median_age'].median(),
        "population_density": demographic_data['total_population'].sum() / demographic_data['land_area'].sum()
    }
    return trends

def calculate_growth_rate(population_data: Dict[str, int]) -> float:
    """
    Calculate the population growth rate.
    """
    years = sorted(population_data.keys())
    initial_pop = population_data[years[0]]
    final_pop = population_data[years[-1]]
    num_years = len(years) - 1
    
    growth_rate = (final_pop / initial_pop) ** (1/num_years) - 1
    return growth_rate

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

def main():
    """
    Main function for testing the data analysis module.
    """
    # Sample demographic data
    demographic_data = pd.DataFrame({
        'year': [2010, 2015, 2020],
        'total_population': [100000, 105000, 110000],
        'median_age': [35, 36, 37],
        'land_area': [50, 50, 50],
        'median_home_value': [200000, 220000, 240000],
        'median_household_income': [50000, 55000, 60000],
        'bachelors_degree_count': [20000, 22000, 24000],
        'unemployment_count': [5000, 4800, 4600]
    })

    # Sample air quality data
    air_quality_data = [
        {'date': '2023-01-01', 'aqi': 50, 'category': 'Good', 'parameter': 'PM2.5'},
        {'date': '2023-01-02', 'aqi': 75, 'category': 'Moderate', 'parameter': 'Ozone'},
        {'date': '2023-01-03', 'aqi': 100, 'category': 'Unhealthy for Sensitive Groups', 'parameter': 'PM2.5'}
    ]

    # Run analysis functions
    print("Demographic Trends:")
    print(analyze_demographic_trends(demographic_data))

    print("\nEnvironmental Impact:")
    print(assess_environmental_impact(air_quality_data))

    print("\nHousing Market:")
    print(analyze_housing_market(demographic_data))

    print("\nEducation Level:")
    print(analyze_education_level(demographic_data))

    print("\nEconomic Indicators:")
    print(analyze_economic_indicators(demographic_data))

    print("\nAir Quality Trends:")
    print(analyze_air_quality_trends(air_quality_data))

if __name__ == "__main__":
    main()