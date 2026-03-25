import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess_data(data_path='data/raw_data.csv'):
    """
    Loads raw data, performs preprocessing (encoding, scaling, train-test split), 
    and returns features and targets.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop CustomerID as it doesn't have predictive power
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode Target (Yes/No to 1/0)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    print(f"Numerical columns: {list(num_cols)}")
    print(f"Categorical columns: {list(cat_cols)}")
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"Preprocessing complete.")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
