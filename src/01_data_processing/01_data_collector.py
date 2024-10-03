import os
import requests
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import json
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler
import shapely.geometry
from urllib.request import urlretrieve
import zipfile
import rasterio
import osmnx as ox
from rasterio.mask import mask
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from shapely.geometry import box
from rasterstats import zonal_stats

# Load environment variables
load_dotenv("env_file.txt")

TRIMBLE_API_KEY = os.getenv('TRIMBLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
AIRNOW_API_KEY = os.getenv('AIRNOW_API_KEY')

# Database setup
DB_NAME = 'urbanpulse.db'

def create_database():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Create tables if they do not exist
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
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
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
    try:
        query = '''INSERT INTO geographic_data (city, lat, lon, timestamp)
                   VALUES (?, ?, ?, ?)'''
        
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute(query, (geo_data['city'], geo_data['lat'], geo_data['lon'], timestamp))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error storing geographic data: {e}")

def get_census_data(start_year: int, end_year: int, dataset: str, variables: List[str], geography: str) -> Optional[pd.DataFrame]:
    """
    Retrieve demographic data from the U.S. Census Bureau API for multiple years.

    Args:
        start_year (int): The start year for data retrieval
        end_year (int): The end year for data retrieval
        dataset (str): The Census dataset to query, e.g., "acs/acs5" for 5-year American Community Survey
        variables (List[str]): List of Census variable codes to retrieve
        geography (str): The geographic scope of the query, e.g., "place:05320" for Bayonne city, NJ

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the requested Census data for all years, or None if all requests fail.
    """
    def fetch_year_data(year):
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
            df['year'] = year
            return df
        except requests.exceptions.RequestException as e:
            print(f"Census API request failed for year {year}: {e}")
            return None

    all_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_year = {executor.submit(fetch_year_data, year): year for year in range(start_year, end_year + 1)}
        for future in as_completed(future_to_year):
            year_data = future.result()
            if year_data is not None:
                all_data.append(year_data)

    if not all_data:
        print("Failed to retrieve data for all years.")
        return None

    return pd.concat(all_data, ignore_index=True)

def clean_census_data(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
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
    
    # Ensure 'year' column is integer type
    df['year'] = df['year'].astype(int)
    
    return df

def store_demographic_data(demo_data):
    try:
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
    except sqlite3.Error as e:
        print(f"Error storing demographic data: {e}")
    finally:
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

def create_adaptive_grid(city_boundary: gpd.GeoDataFrame, population_data: pd.DataFrame, min_cell_size: float, max_cell_size: float) -> gpd.GeoDataFrame:
    # Calculate population density
    latest_year = population_data['year'].max()
    latest_population = population_data[population_data['year'] == latest_year]['total_population'].iloc[0]
    area_km2 = city_boundary.to_crs(epsg=3857).area.iloc[0] / 1e6  # Convert to km2
    density = latest_population / area_km2
    
    # Normalize density to 0-1 range for cell size calculation
    max_density = 10000  # Assumes a max urban density of 10,000 people per km2
    normalized_density = min(density / max_density, 1)
    
    # Calculate cell size based on density
    cell_size = min_cell_size + (max_cell_size - min_cell_size) * (1 - normalized_density)
    
    # Create grid
    minx, miny, maxx, maxy = city_boundary.total_bounds
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)
    cells = [shapely.geometry.box(x, y, x + cell_size, y + cell_size)
             for x in x_coords for y in y_coords]
    
    # Create GeoDataFrame from cells
    grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=city_boundary.crs)
    
    # Only keep cells that intersect with the city boundary
    grid_gdf = grid_gdf[grid_gdf.intersects(city_boundary.geometry.iloc[0])]
    
    return grid_gdf

def get_latest_available_year():
    current_year = datetime.now().year
    # The ACS 5-year estimates are typically released in December for the previous year
    return current_year - 2  # This ensures we're using the most recent available data

def get_city_boundary(state_fips, county_fips, place_fips):
    year = str(get_latest_available_year())
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/PLACE/tl_{year}_{state_fips}_place.zip"
    zip_file = "place_shapefile.zip"
    urlretrieve(url, zip_file)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("place_shapefile")
    
    shapefile = f"place_shapefile/tl_{year}_{state_fips}_place.shp"
    places = gpd.read_file(shapefile)
    city_boundary = places[places['PLACEFP'] == place_fips]
    
    # Clean up
    os.remove(zip_file)
    for file in os.listdir("place_shapefile"):
        os.remove(os.path.join("place_shapefile", file))
    os.rmdir("place_shapefile")
    
    return city_boundary

def download_usgs_data(city_name: str) -> None:
    """
    Automate the process of downloading data from the USGS National Map Viewer.

    Args:
        city_name (str): The name of the city to search for.
    """
    # Set up the Selenium WebDriver (make sure to specify the correct path to your driver)
    driver = webdriver.Chrome(executable_path='/path/to/chromedriver')
    
    try:
        # Navigate to the USGS National Map Downloader
        driver.get("https://apps.nationalmap.gov/downloader/")
        
        # Wait for the page to load
        time.sleep(5)
        
        # Enter the city name in the search box
        search_box = driver.find_element(By.NAME, "search")
        search_box.send_keys(city_name)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for search results to load
        time.sleep(5)
        
        # Select the datasets
        # Assuming the datasets are available in a specific panel, you may need to adjust the selectors
        land_cover_checkbox = driver.find_element(By.XPATH, "//label[contains(text(), 'Land Cover')]/preceding-sibling::input")
        elevation_checkbox = driver.find_element(By.XPATH, "//label[contains(text(), 'Elevation')]/preceding-sibling::input")
        
        land_cover_checkbox.click()
        elevation_checkbox.click()
        
        # Click "Search Products"
        search_products_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Search Products')]")
        search_products_button.click()
        
        # Wait for the products to load
        time.sleep(5)
        
        # Download the relevant files (you may need to adjust the selectors based on the page structure)
        download_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Download')]")
        for button in download_buttons:
            button.click()
            time.sleep(2)  # Wait for the download to start

    finally:
        driver.quit()

def get_geographic_data(city_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Assuming city_boundary is already obtained
    return city_boundary

def get_demographic_data(start_year: int, end_year: int, dataset: str, variables: List[str], geography: str) -> Optional[pd.DataFrame]:
    # Use the existing function to get demographic data
    return get_census_data(start_year, end_year, dataset, variables, geography)

def get_economic_data(start_year: int, end_year: int, dataset: str, variables: List[str], geography: str) -> Optional[pd.DataFrame]:
    # Use the existing function to get economic data
    return get_census_data(start_year, end_year, dataset, variables, geography)

def get_infrastructure_data(city_name: str) -> gpd.GeoDataFrame:
    # Get road network data using OSMnx
    graph = ox.graph_from_place(city_name, network_type='all')
    roads = ox.graph_to_gdfs(graph, nodes=False)
    return roads

def download_nlcd_data(city_name: str, state: str, output_folder: str) -> str:
    """
    Download NLCD data for the specified city and state, and save it to the output folder.

    Args:
        city_name (str): The name of the city.
        state (str): The state abbreviation.
        output_folder (str): The folder where the NLCD data will be saved.

    Returns:
        str: The path to the downloaded NLCD raster file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the URL for the NLCD data (this is an example URL; adjust as needed)
    nlcd_url = f"https://www.mrlc.gov/data/nlcd-{state.lower()}-{city_name.lower().replace(' ', '-')}-land-cover.zip"

    # Download the NLCD data
    response = requests.get(nlcd_url)
    if response.status_code == 200:
        zip_file_path = os.path.join(output_folder, "nlcd_data.zip")
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)
        
        # Unzip the downloaded file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        
        # Return the path to the extracted NLCD raster file (adjust based on the actual extracted file name)
        return os.path.join(output_folder, "nlcd_data.tif")  # Adjust based on the actual extracted file name
    else:
        print(f"Failed to download NLCD data: {response.status_code}")
        return None

def get_environmental_data(city_boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    # Define the path to the NLCD data folder relative to the project structure
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nlcd_folder = os.path.join(project_folder, "src", "01_data_processing", "nlcd_2001_2021_land_cover_change_index_l48_20230630")
    
    # Define the expected NLCD raster file path
    nlcd_raster_path = os.path.join(nlcd_folder, "nlcd_2001_2021_land_cover_change_index_l48_20230630.img")

    # Check if the NLCD raster file exists
    if not os.path.exists(nlcd_raster_path):
        print("The specified NLCD raster file does not exist.")
        return pd.DataFrame()  # Return empty DataFrame if the file is not found

    # Read the NLCD raster data
    try:
        with rasterio.open(nlcd_raster_path) as src:
            nlcd_data = src.read(1)  # Read the first band (land cover)
            nlcd_meta = src.meta
        
        # Ensure the city boundary is in the same CRS as the raster
        city_boundary = city_boundary.to_crs(nlcd_meta['crs'])
        
        # Calculate zonal statistics for land cover within the city boundary
        stats = zonal_stats(city_boundary, nlcd_data, stats="mode", geojson_out=True)
        
        # Extract land cover categories from the stats
        land_cover_data = pd.DataFrame([feature['properties'] for feature in stats])
        
        return land_cover_data
    except Exception as e:
        print(f"Error reading NLCD data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if an error occurs

def get_air_quality_data(zip_code: str) -> pd.DataFrame:
    # Use the existing function to get air quality data
    current_data = get_airnow_current_observation(zip_code)
    return clean_airnow_data(current_data, 'current')

def create_urban_grid(city_boundary: gpd.GeoDataFrame, grid_size: int = 50) -> gpd.GeoDataFrame:
    # Create a grid of the specified size over the city boundary
    minx, miny, maxx, maxy = city_boundary.total_bounds
    x_coords = np.linspace(minx, maxx, grid_size + 1)
    y_coords = np.linspace(miny, maxy, grid_size + 1)
    
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
            cells.append(cell)
    
    grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=city_boundary.crs)
    return grid_gdf

def aggregate_data_to_grid(grid: gpd.GeoDataFrame, demographic_data: pd.DataFrame, economic_data: pd.DataFrame, infrastructure_data: gpd.GeoDataFrame, environmental_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
    # Initialize columns for the grid
    grid['land_use'] = None
    grid['population_density'] = None
    grid['median_income'] = None
    grid['road_density'] = None
    grid['green_space'] = None
    
    for idx, cell in grid.iterrows():
        # Get the geometry of the cell
        cell_geom = cell.geometry
        
        # Land Use: Get the most common land use category in the cell
        land_use = environmental_data.loc[environmental_data.intersects(cell_geom), 'land_cover'].mode()
        grid.at[idx, 'land_use'] = land_use[0] if not land_use.empty else None
        
        # Population Density: Calculate based on demographic data
        population_in_cell = demographic_data.loc[demographic_data['geometry'].intersects(cell_geom), 'total_population'].sum()
        area_of_cell = cell_geom.area
        grid.at[idx, 'population_density'] = population_in_cell / area_of_cell if area_of_cell > 0 else 0
        
        # Median Income: Calculate based on economic data
        median_income = economic_data.loc[economic_data['geometry'].intersects(cell_geom), 'median_household_income'].median()
        grid.at[idx, 'median_income'] = median_income if not pd.isna(median_income) else None
        
        # Road Density: Calculate based on infrastructure data
        roads_in_cell = infrastructure_data.loc[infrastructure_data.intersects(cell_geom)]
        road_length = roads_in_cell.length.sum()
        grid.at[idx, 'road_density'] = road_length / area_of_cell if area_of_cell > 0 else 0
        
        # Green Space: Calculate based on environmental data
        green_space_area = environmental_data.loc[environmental_data['geometry'].intersects(cell_geom), 'green_space_area'].sum()
        grid.at[idx, 'green_space'] = green_space_area / area_of_cell if area_of_cell > 0 else 0

    return grid

def main():
    create_database()
    
    # Prompt user for city name and state
    state = input("Enter the state abbreviation (e.g., NJ): ")
    city = input("Enter the city name (e.g., Bayonne): ")
    full_city_name = f"{city}, {state}"
    zip_code = input("Enter the zip code (e.g., 07002): ")
    
    state_fips = "34"  # New Jersey
    county_fips = "017"  # Hudson County
    place_fips = "04180"  # Bayonne city
    
    # Get and clean geographic data from Trimble API
    trimble_data = get_trimble_location_data(full_city_name)
    cleaned_geo_data = clean_trimble_data(trimble_data)
    
    if cleaned_geo_data:
        print("Cleaned Trimble Geographic Data:")
        print(json.dumps(cleaned_geo_data, indent=2))
        store_geographic_data(cleaned_geo_data)
    else:
        print("No valid geographic data available.")
        return
    
    # Get city boundary
    city_boundary = get_city_boundary(state_fips, county_fips, place_fips)
    if city_boundary.empty:
        print("Failed to retrieve city boundary.")
        return
    print(f"City boundary retrieved. Area: {city_boundary.to_crs(epsg=3857).area.iloc[0] / 1e6:.2f} km²")

    # Get demographic and economic data from Census API
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
    geography = f"place:{place_fips}"

    census_df = get_census_data(start_year, end_year, dataset, variables, geography)
    cleaned_census_df = clean_census_data(census_df)
    
    if cleaned_census_df is not None:
        print("\nCleaned Census Demographic and Economic Data:")
        print(cleaned_census_df)
        store_demographic_data(cleaned_census_df)
        
        # Calculate some derived statistics for the most recent year
        latest_year = cleaned_census_df['year'].max()
        latest_data = cleaned_census_df[cleaned_census_df['year'] == latest_year]
        
        total_pop = latest_data['total_population'].iloc[0]
        unemployed = latest_data['unemployment_count'].iloc[0]
        bachelors = latest_data['bachelors_degree_count'].iloc[0]
        
        unemployment_rate = (unemployed / total_pop) * 100
        education_rate = (bachelors / total_pop) * 100
        
        print(f"\nDerived Statistics for {full_city_name} (Year {latest_year}):")
        print(f"Unemployment Rate: {unemployment_rate:.2f}%")
        print(f"Percentage with Bachelor's Degree: {education_rate:.2f}%")
        
        # Create adaptive grid
        adaptive_grid = create_adaptive_grid(city_boundary, cleaned_census_df, min_cell_size=0.001, max_cell_size=0.005)
        print(f"\nAdaptive Grid Created:")
        print(f"Number of cells: {len(adaptive_grid)}")
        
        # Save the grid to a GeoJSON file
        adaptive_grid.to_file("adaptive_grid.geojson", driver="GeoJSON")
        print("Adaptive grid saved to 'adaptive_grid.geojson'")
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

    # Download USGS data
    download_usgs_data(full_city_name)

    # Get geographic data
    city_boundary = get_city_boundary(state_fips, county_fips, place_fips)
    
    # Get demographic and economic data
    demographic_data = get_demographic_data(start_year, end_year, dataset, variables, geography)
    economic_data = get_economic_data(start_year, end_year, dataset, variables, geography)
    
    # Get infrastructure data
    infrastructure_data = get_infrastructure_data(full_city_name)
    
    # Get environmental data
    environmental_data = get_environmental_data(city_boundary)
    
    # Get air quality data
    air_quality_data = get_air_quality_data(zip_code)
    
    # Create urban grid
    urban_grid = create_urban_grid(city_boundary)
    
    # Aggregate data to the grid
    aggregated_data = aggregate_data_to_grid(urban_grid, demographic_data, economic_data, infrastructure_data, environmental_data, air_quality_data)
    
    # Save the aggregated data to a file
    aggregated_data.to_file("urban_grid_data.geojson", driver="GeoJSON")
    print("Aggregated urban grid data saved to 'urban_grid_data.geojson'")

if __name__ == "__main__":
    main()