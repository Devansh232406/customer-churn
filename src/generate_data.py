import pandas as pd
import numpy as np
import os

def generate_synthetic_churn_data(n_samples=5000, output_path='data/raw_data.csv'):
    """
    Generates a synthetic customer churn dataset.
    """
    np.random.seed(42)
    
    # Generate features
    customer_id = [f"CUST_{i:04d}" for i in range(n_samples)]
    age = np.random.randint(18, 80, n_samples)
    tenure_months = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, n_samples)
    
    # Categorical features
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract = np.random.choice(contract_types, n_samples, p=[0.55, 0.25, 0.20])
    
    internet_services = ['DSL', 'Fiber optic', 'No']
    internet_service = np.random.choice(internet_services, n_samples, p=[0.35, 0.45, 0.20])
    
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        n_samples
    )
    
    # Create a baseline churn probability
    base_prob = np.zeros(n_samples)
    
    # Logic to introduce correlation with features
    base_prob += np.where(contract == 'Month-to-month', 0.8, 0.1) # higher churn for M2M
    base_prob -= np.where(contract == 'Two year', 0.5, 0)
    base_prob += np.where(internet_service == 'Fiber optic', 0.3, 0) # higher churn for Fiber optic maybe
    base_prob -= (tenure_months / 72) * 1.5  # Longer tenure reduces churn
    base_prob += (monthly_charges / 120) * 0.5 # Higher charges increase churn
    
    # Sigmoid function to keep probs in [0,1]
    prob_churn = 1 / (1 + np.exp(-base_prob))
    
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)
    churn_str = ['Yes' if c == 1 else 'No' for c in churn]
    
    df = pd.DataFrame({
        'CustomerID': customer_id,
        'Age': age,
        'TenureMonths': tenure_months,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'Churn': churn_str
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {n_samples} samples at {output_path}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_churn_data()
