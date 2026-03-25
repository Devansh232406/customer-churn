# Customer Churn Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict customer churn using classical machine learning algorithms.

## Overview
The goal of this project is to identify customers who are likely to stop using a company's service (churn). By accurately predicting churn, businesses can take targeted actions to retain valuable customers.

## Project Structure
- `data/`: Contains the generated raw and processed datasets (ignored in git if applicable).
- `models/`: Contains the serialized trained machine learning models (`.pkl` files) and the fitted standard scaler.
- `outputs/`: Output directory for evaluation metrics (CSV) and automatically generated charts (Confusion Matrices, ROC curves).
- `src/`: Source code directory containing modular python scripts:
  - `generate_data.py`: Script to generate a realistic synthetic telecom customer churn dataset.
  - `data_preprocessing.py`: Handles missing values (if any), encodes categorical variables, balances features, scales numerical features, and performs train-test splitting.
  - `model_training.py`: Trains Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine models.
  - `evaluate_models.py`: Evaluates the trained models and generates performance charts and metrics.
- `main.py`: The entry point script to run the entire pipeline seamlessly.
- `requirements.txt`: Python package dependencies.
- `description.md`: Detailed theoretical background and algorithm descriptions.

## Setup Instructions

1. **Install dependencies:**
   Ensure you have Python 3.8+ installed. Run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline:**
   Execute the `main.py` script to run data generation, preprocessing, training, and evaluation in one go.
   ```bash
   python main.py
   ```

3. **View Results:**
   Check the `outputs/` directory for the model performance metrics CSV and the generated diagnostic plots.
