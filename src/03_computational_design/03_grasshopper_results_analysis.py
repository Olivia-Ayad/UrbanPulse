import json
import geopandas as gpd
import matplotlib.pyplot as plt

def analyze_grasshopper_results(results_file):
    # Load results from Grasshopper
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(results['features'])
    
    # Analyze results
    land_use_distribution = gdf['land_use'].value_counts(normalize=True)
    density = gdf['population'].sum() / gdf.total_bounds[2]  # Assuming total_bounds[2] is total area
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    gdf.plot(column='land_use', categorical=True, legend=True, ax=ax1)
    ax1.set_title('Optimized Urban Design')
    
    land_use_distribution.plot(kind='bar', ax=ax2)
    ax2.set_title('Land Use Distribution')
    ax2.set_ylabel('Proportion')
    
    plt.tight_layout()
    plt.savefig('urban_design_analysis.png')
    
    print(f"Population Density: {density:.2f} people per square unit")
    print("Land Use Distribution:")
    print(land_use_distribution)
    print("Visualization saved as 'urban_design_analysis.png'")

if __name__ == "__main__":
    analyze_grasshopper_results('grasshopper_output.json')