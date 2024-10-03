from src.01_data_processing.data_collector import main as collect_data
from src.01_data_processing.data_analyzer import main as analyze_data
from src.02_ml_model.urban_growth_predictor import main as train_model

def main():
    # Collect and process data
    collect_data()
    
    # Analyze data
    analyze_data()
    
    # Train ML model
    train_model()

if __name__ == "__main__":
    main()