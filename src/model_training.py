import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import os

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    print("Training SVM...")
    # probability=True is needed for ROC-AUC
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    models = {
        'LogisticRegression': train_logistic_regression(X_train, y_train),
        'RandomForest': train_random_forest(X_train, y_train),
        'GradientBoosting': train_gradient_boosting(X_train, y_train),
        'SVM': train_svm(X_train, y_train)
    }
    
    for name, model in models.items():
        joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
        print(f"Saved {name} to {save_dir}/{name}.pkl")
        
    return models
