import pandas as pd
import geopandas as gpd
import json

def prepare_data_for_grasshopper(growth_predictions, city_boundary):
    # Load growth predictions
    predictions_df = pd.read_csv(growth_predictions)
    
    # Load city boundary
    boundary_gdf = gpd.read_file(city_boundary)
    
    # Prepare growth data
    growth_data = predictions_df[['year', 'predicted_population', 'predicted_growth_rate']].to_dict('records')
    
    # Prepare boundary data
    boundary_json = json.loads(boundary_gdf.to_json())
    
    # Combine data
    grasshopper_data = {
        'growth_predictions': growth_data,
        'city_boundary': boundary_json
    }
    
    # Save to JSON file
    with open('grasshopper_input_data.json', 'w') as f:
        json.dump(grasshopper_data, f)
    
    print("Data prepared for Grasshopper and saved to 'grasshopper_input_data.json'")

if __name__ == "__main__":
    prepare_data_for_grasshopper('growth_predictions.csv', 'city_boundary.geojson')