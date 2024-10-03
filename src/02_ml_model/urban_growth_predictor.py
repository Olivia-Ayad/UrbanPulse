import joblib
import pandas as pd
import geopandas as gpd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_NAME = 'urbanpulse.db'

def connect_to_database():
    """Connect to the SQLite database and return the connection object."""
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

def load_data(conn):
    """Load demographic and air quality data from the database."""
    demographic_query = "SELECT * FROM demographic_data"
    air_quality_query = "SELECT * FROM air_quality_data"

    demographic_data = pd.read_sql_query(demographic_query, conn)
    air_quality_data = pd.read_sql_query(air_quality_query, conn)

    return demographic_data, air_quality_data

def prepare_features(demographic_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for the ML model."""
    # Merge demographic and air quality data
    features = pd.merge(demographic_data, air_quality_data, on='city', how='left')

    # Calculate additional features
    features['population_density'] = features['total_population'] / features['area']  # Assuming 'area' is available
    features['unemployment_rate'] = features['unemployment_count'] / features['total_population']
    
    # Drop unnecessary columns
    features = features.drop(columns=['city', 'timestamp'])  # Adjust based on your schema

    return features

def create_target(features: pd.DataFrame) -> pd.Series:
    """Create the target variable: population growth rate."""
    features['population_growth_rate'] = features['total_population'].pct_change()
    target = features['population_growth_rate'].dropna()  # Remove NaN values
    return target

def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestRegressor:
    """Train the Random Forest model."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler

def evaluate_model(model: RandomForestRegressor, scaler: StandardScaler, features: pd.DataFrame, target: pd.Series) -> dict:
    """Evaluate the model's performance."""
    X = scaler.transform(features)
    y_pred = model.predict(X)

    mse = mean_squared_error(target, y_pred)
    r2 = r2_score(target, y_pred)

    return {
        'MSE': mse,
        'R2': r2,
        'RMSE': np.sqrt(mse)
    }

def predict_future_growth(model: RandomForestRegressor, scaler: StandardScaler, last_year_data: pd.DataFrame, num_years: int = 5) -> pd.DataFrame:
    """Predict future urban growth."""
    future_predictions = []
    current_data = last_year_data.copy()

    for _ in range(num_years):
        X = scaler.transform(current_data.drop(['year', 'population_growth_rate'], axis=1))
        growth_rate = model.predict(X)[0]
        
        next_year = current_data['year'].iloc[0] + 1
        next_population = current_data['total_population'].iloc[0] * (1 + growth_rate)
        
        prediction = {
            'year': next_year,
            'predicted_population': next_population,
            'predicted_growth_rate': growth_rate
        }
        future_predictions.append(prediction)
        
        # Update current_data for next prediction
        current_data['year'] = next_year
        current_data['total_population'] = next_population
        # You might want to add logic to update other features based on your domain knowledge

    return pd.DataFrame(future_predictions)

def main():
    # Connect to the database
    conn = connect_to_database()
    if conn is None:
        return  # Exit if the connection failed

    # Load data
    demographic_data, air_quality_data = load_data(conn)

    # Prepare features
    features = prepare_features(demographic_data, air_quality_data)

    # Create target variable
    target = create_target(features)

    # Train model
    model, scaler = train_model(features, target)

    # Evaluate model
    metrics = evaluate_model(model, scaler, features, target)
    logging.info("Model Performance:")
    logging.info(f"Metrics: {metrics}")

    # Save the model and scaler
    joblib.dump((model, scaler), 'urban_growth_model.joblib')
    logging.info("Model and scaler saved to 'urban_growth_model.joblib'")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()