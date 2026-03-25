# Customer Churn Prediction: Technical Description

## 1. Problem Statement
Customer Churn refers to the phenomenon where customers or subscribers stop doing business with a company or service. The ability to predict which customers are at high risk of churning is highly valuable, as acquiring new customers is often significantly more expensive than retaining existing ones.

This project approaches churn prediction as a **Binary Classification** problem.

## 2. Dataset
A synthetic but realistic telecom dataset is generated for this project. The features include:
- **Demographics & Usage**: Age, Tenure (months), Monthly Charges.
- **Categorical factors**: Contract type, Internet service type, Paperless billing, Payment method.
- **Target Variable**: `Churn` (Yes/No), which indicates whether the customer left within the last month.

## 3. Data Preprocessing
- **Encoding Target Variable**: The `Churn` (Yes/No) label is encoded to Binary (1/0) using `LabelEncoder`.
- **One-Hot Encoding**: Used to convert multi-class categorical features (Contract, PaymentMethod, InternetService) into binary vectors, avoiding ordinal bias.
- **Feature Scaling**: Numerical variables (`Age`, `TenureMonths`, `MonthlyCharges`) are standardized using `StandardScaler` to have a mean of 0 and variance of 1. This is especially important for distance-based models like SVM and gradient-descent-based models like Logistic Regression.

## 4. Machine Learning Algorithms
The project utilizes several classical machine learning classifiers:

### 4.1 Logistic Regression
A linear model that estimates the probability of a binary response based on one or more predictor variables. It applies a sigmoid function to the linear combination of inputs. It is heavily used as a baseline model due to its simplicity, interpretability, and speed.

### 4.2 Random Forest Classifier
An ensemble learning method that operates by constructing a multitude of decision trees at training time. It outputs the class that is the mode of the classes of the individual trees. By averaging multiple trees trained on diverse subsets of data, it significantly reduces the overfitting typically seen in highly complex single decision trees.

### 4.3 Gradient Boosting Classifier
Another ensemble technique that builds trees sequentially. Each new tree helps to correct errors made by previously trained trees. It optimizes a differentiable loss function, leading to highly accurate models, though it can be sensitive to hyperparameter choices.

### 4.4 Support Vector Machine (SVM)
Constructs a hyperplane or set of hyperplanes in a high-dimensional space. A good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class. It is effective in high dimensional spaces and versatile due to the use of different kernel functions.

## 5. Evaluation Metrics
Models are evaluated on a held-out test set using the following metrics:
- **Accuracy**: The overall correct prediction rate.
- **Precision**: The proportion of positive identifications that were actually correct (reducing false positives).
- **Recall (Sensitivity)**: The proportion of actual churners that were identified correctly (reducing false negatives). In churn prediction, Recall is often prioritized to ensure high-risk customers aren't missed.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between classes across all classification thresholds. 
