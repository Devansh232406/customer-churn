import os
from src.generate_data import generate_synthetic_churn_data
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_all_models
from src.evaluate_models import evaluate_all_models

def main():
    print("=== Customer Churn Prediction Pipeline ===")
    
    data_path = 'data/raw_data.csv'
    
    # Step 1: Data Generation (if not already present)
    if not os.path.exists(data_path):
        print("\n[Step 1] Initializing Data...")
        generate_synthetic_churn_data(n_samples=5000, output_path=data_path)
    else:
        print("\n[Step 1] Data already exists. Skipping generation.")
        
    # Step 2: Data Preprocessing
    print("\n[Step 2] Preprocessing Data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Step 3: Model Training
    print("\n[Step 3] Training Models...")
    models = train_all_models(X_train, y_train)
    
    # Step 4: Model Evaluation
    print("\n[Step 4] Evaluating Models...")
    evaluate_all_models(models, X_test, y_test)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
