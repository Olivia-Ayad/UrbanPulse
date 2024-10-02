import os
import requests
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import json
import backoff

# Load environment variables
load_dotenv("env_file.txt")

TRIMBLE_API_KEY = os.getenv('TRIMBLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
AIRNOW_API_KEY = os.getenv('AIRNOW_API_KEY')

# Database setup
DB_NAME = 'urbanpulse.db'

def create_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS geographic_data
                 (city TEXT, lat REAL, lon REAL, timestamp TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS demographic_data
                 (city TEXT, total_population INTEGER, median_household_income REAL, 
                  median_home_value REAL, unemployment_count INTEGER, 
                  bachelors_degree_count INTEGER, timestamp TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS air_quality_data
                 (city TEXT, date TEXT, parameter TEXT, aqi INTEGER, category TEXT, 
                 concentration REAL, unit TEXT, type TEXT, timestamp TEXT)''')
    
    conn.commit()
    conn.close()

def get_headers(api_key):
    return {
        'Authorization': api_key,
        'Accept': 'application/json',
        'Content-type': 'application/json'
    }

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=10)
def make_trimble_request(url, headers, params=None, data=None, method='GET'):
    if method == 'GET':
        response = requests.get(url, headers=headers, params=params, timeout=10)
    elif method == 'POST':
        response = requests.post(url, headers=headers, params=params, data=json.dumps(data), timeout=10)
    response.raise_for_status()
    return response.json()

def get_trimble_location_data(query: str) -> Optional[Dict]:
    """
    Fetch location data from the Trimble API.

    This function sends a GET request to the Trimble SingleSearch API to retrieve 
    geographic information based on the provided query (typically a city name).

    Args:
        query (str): The location query string, e.g., "Bayonne, NJ"

    Returns:
        Optional[Dict]: A dictionary containing the API response if successful, None otherwise.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """
    region = 'na'  # 'na' stands for North America
    url = f'https://singlesearch.alk.com/{region}/api/search'
    
    params = {
        'query': query,
        'includeTrimblePlaceIds': 'true'
    }
    
    headers = get_headers(TRIMBLE_API_KEY)
    
    try:
        result = make_trimble_request(url, headers, params)
        return result
    except requests.exceptions.RequestException as e:
        print(f"Trimble API request failed: {e}")
        return None

def clean_trimble_data(data: Optional[Dict]) -> Optional[Dict[str, float]]:
    """
    Clean and extract relevant information from the Trimble API response.

    This function parses the raw API response, extracting the city name, latitude, and longitude.
    It handles potential missing data and returns None if the required information is not available.

    Args:
        data (Optional[Dict]): The raw API response from the Trimble API.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing cleaned data with keys 'city', 'lat', and 'lon',
                                    or None if the data is invalid or missing.
    """
    if not data or 'single' not in data or not data['single']:
        return None
    
    city_data = data['single'][0]
    return {
        'city': city_data.get('address', {}).get('city', 'Unknown'),
        'lat': float(city_data.get('lat', 0)),
        'lon': float(city_data.get('lon', 0))
    }

def store_geographic_data(geo_data: Dict[str, float]) -> None:
    """
    Store geographic data in the SQLite database.

    Args:
        geo_data (Dict[str, float]): A dictionary containing 'city', 'lat', and 'lon' keys.

    Raises:
        sqlite3.Error: If there's an issue with the database operation.
    """
    query = '''INSERT INTO geographic_data (city, lat, lon, timestamp)
               VALUES (?, ?, ?, ?)'''
    
    timestamp = datetime.now().isoformat()
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute(query, (geo_data['city'], geo_data['lat'], geo_data['lon'], timestamp))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise

def get_census_data(year: str, dataset: str, variables: List[str], geography: str) -> Optional[pd.DataFrame]:
    """
    Retrieve demographic data from the U.S. Census Bureau API.

    This function constructs a request to the Census API based on the provided parameters,
    fetches the data, and returns it as a pandas DataFrame.

    Args:
        year (str): The year for which to retrieve data, e.g., "2019"
        dataset (str): The Census dataset to query, e.g., "acs/acs5" for 5-year American Community Survey
        variables (List[str]): List of Census variable codes to retrieve
        geography (str): The geographic scope of the query, e.g., "place:05320" for Bayonne city, NJ

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the requested Census data, or None if the request fails.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """
    base_url = f"https://api.census.gov/data/{year}/{dataset}"
    
    params = {
        "get": ",".join(variables),
        "for": geography,
        "key": CENSUS_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except requests.exceptions.RequestException as e:
        print(f"Census API request failed: {e}")
        return None

def clean_census_data(df):
    if df is None or df.empty:
        return None
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['B01003_001E', 'B19013_001E', 'B25077_001E', 'B23025_005E', 'B15003_022E']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename columns for clarity
    df = df.rename(columns={
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B25077_001E': 'median_home_value',
        'B23025_005E': 'unemployment_count',
        'B15003_022E': 'bachelors_degree_count'
    })
    
    return df

def store_demographic_data(demo_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    c.execute('''INSERT INTO demographic_data 
                 (city, total_population, median_household_income, median_home_value, 
                  unemployment_count, bachelors_degree_count, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (demo_data['NAME'].iloc[0], 
               demo_data['total_population'].iloc[0],
               demo_data['median_household_income'].iloc[0],
               demo_data['median_home_value'].iloc[0],
               demo_data['unemployment_count'].iloc[0],
               demo_data['bachelors_degree_count'].iloc[0],
               timestamp))
    
    conn.commit()
    conn.close()

def get_airnow_current_observation(zip_code):
    base_url = "http://www.airnowapi.org/aq/observation/zipCode/current/"
    
    params = {
        "format": "application/json",
        "zipCode": zip_code,
        "distance": 25,
        "API_KEY": AIRNOW_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"AirNow API request failed: {e}")
        return None

def get_airnow_forecast(zip_code):
    base_url = "http://www.airnowapi.org/aq/forecast/zipCode/"
    
    params = {
        "format": "application/json",
        "zipCode": zip_code,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "distance": 25,
        "API_KEY": AIRNOW_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"AirNow API request failed: {e}")
        return None

def get_airnow_historical(zip_code, date):
    base_url = "http://www.airnowapi.org/aq/observation/zipCode/historical/"
    
    params = {
        "format": "application/json",
        "zipCode": zip_code,
        "date": date.strftime("%Y-%m-%dT00-0000"),
        "distance": 25,
        "API_KEY": AIRNOW_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"AirNow API request failed: {e}")
        return None

def clean_airnow_data(data, data_type):
    if not data:
        return None
    
    cleaned_data = []
    for item in data:
        cleaned_item = {
            'city': item.get('ReportingArea', ''),
            'date': item.get('DateObserved', '') if data_type != 'forecast' else item.get('DateForecast', ''),
            'parameter': item.get('ParameterName', ''),
            'aqi': item.get('AQI', 0),
            'category': item.get('Category', {}).get('Name', ''),
            'concentration': item.get('Concentration', 0),
            'unit': item.get('Unit', ''),
            'type': data_type
        }
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

def store_air_quality_data(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    for item in data:
        c.execute('''INSERT INTO air_quality_data 
                     (city, date, parameter, aqi, category, concentration, unit, type, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (item['city'], item['date'], item['parameter'], 
                   item['aqi'], item['category'], item['concentration'], 
                   item['unit'], item['type'], timestamp))
    
    conn.commit()
    conn.close()

def main():
    create_database()
    
    city = "Bayonne, NJ"
    zip_code = "07002"
    
    # Get and clean geographic data from Trimble API
    trimble_data = get_trimble_location_data(city)
    cleaned_geo_data = clean_trimble_data(trimble_data)
    
    if cleaned_geo_data:
        print("Cleaned Trimble Geographic Data:")
        print(json.dumps(cleaned_geo_data, indent=2))
        store_geographic_data(cleaned_geo_data)
    else:
        print("No valid geographic data available.")
        return
    
    # Get demographic and economic data from Census API
    year = "2019"
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

    census_df = get_census_data(year, dataset, variables, geography)
    cleaned_census_df = clean_census_data(census_df)
    
    if cleaned_census_df is not None:
        print("\nCleaned Census Demographic and Economic Data:")
        print(cleaned_census_df)
        store_demographic_data(cleaned_census_df)
        
        # Calculate some derived statistics
        total_pop = cleaned_census_df['total_population'].iloc[0]
        unemployed = cleaned_census_df['unemployment_count'].iloc[0]
        bachelors = cleaned_census_df['bachelors_degree_count'].iloc[0]
        
        unemployment_rate = (unemployed / total_pop) * 100
        education_rate = (bachelors / total_pop) * 100
        
        print(f"\nDerived Statistics for {city}:")
        print(f"Unemployment Rate: {unemployment_rate:.2f}%")
        print(f"Percentage with Bachelor's Degree: {education_rate:.2f}%")
    else:
        print("No valid census data available.")
    
    # Get current air quality data
    current_data = get_airnow_current_observation(zip_code)
    if current_data:
        cleaned_current_data = clean_airnow_data(current_data, 'current')
        print("\nCleaned Current Air Quality Data:")
        print(json.dumps(cleaned_current_data, indent=2))
        store_air_quality_data(cleaned_current_data)
    
    # Get air quality forecast
    forecast_data = get_airnow_forecast(zip_code)
    if forecast_data:
        cleaned_forecast_data = clean_airnow_data(forecast_data, 'forecast')
        print("\nCleaned Air Quality Forecast:")
        print(json.dumps(cleaned_forecast_data, indent=2))
        store_air_quality_data(cleaned_forecast_data)
    
    # Get historical air quality data (for yesterday)
    yesterday = datetime.now() - timedelta(days=1)
    historical_data = get_airnow_historical(zip_code, yesterday)
    if historical_data:
        cleaned_historical_data = clean_airnow_data(historical_data, 'historical')
        print("\nCleaned Historical Air Quality Data:")
        print(json.dumps(cleaned_historical_data, indent=2))
        store_air_quality_data(cleaned_historical_data)
    
    print("\nThis product uses the Trimble Maps API, Census Bureau Data API, and EPA AirNow API but is not endorsed or certified by these agencies.")

if __name__ == "__main__":
    main()    