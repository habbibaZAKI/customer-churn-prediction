# Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecommunications company using various classification algorithms. The project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and an interactive prediction interface.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)

## Overview

Customer churn is a critical business metric for telecommunications companies. This project aims to predict which customers are likely to churn (cancel their service) based on various customer attributes and service usage patterns. The model helps identify at-risk customers, enabling proactive retention strategies.

## Dataset

The dataset used in this project is the **Telco Customer Churn** dataset, which contains information about 7,043 customers with 21 features including:

- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Services**: Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Account Information**: Tenure (months), Contract type, Paperless Billing, Payment Method
- **Charges**: Monthly Charges, Total Charges
- **Target Variable**: Churn (Yes/No)

### Dataset Statistics

- **Total Records**: 7,043 customers
- **Churn Rate**: 26.58% (1,869 customers)
- **Retention Rate**: 73.42% (5,163 customers)
- **Features**: 21 columns (20 features + 1 target)

## Features

- **Comprehensive Data Preprocessing**: Handles missing values, data type conversions, and feature encoding
- **Exploratory Data Analysis**: Visualizations and statistical analysis of customer behavior patterns
- **Multiple ML Models**: Comparison of Logistic Regression and Random Forest Classifier
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique) for balanced training
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Feature Importance Analysis**: Identifies key factors driving customer churn
- **Model Persistence**: Saved trained model for future predictions
- **Interactive Interface**: User-friendly prediction interface for new customer data

## Project Structure

```
Telco Customer Churn/
|
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── Telco Customer Churn.ipynb        # Main analysis and model training notebook
├── small interactive interface.ipynb # Interactive prediction interface
├── churn_model.pkl                    # Trained Random Forest model (saved)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
|
└── [Generated outputs]
    ├── Visualizations (correlation heatmaps, bar plots, feature importance)
    └── Model evaluation metrics
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip or conda package manager

### Setup

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or if using conda:
   ```bash
   conda install --file requirements.txt
   ```

3. **Ensure the dataset file is in the project directory**:
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Usage

### Running the Main Analysis

1. Open `Telco Customer Churn.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train and evaluate models
   - Generate visualizations
   - Save the trained model

### Making Predictions

#### Option 1: Using the Interactive Notebook

1. Open `small interactive interface.ipynb`
2. Run all cells to load the saved model
3. Follow the prompts to enter customer information:
   - Tenure (months)
   - Monthly Charges
   - Total Charges
   - Contract type
   - Payment Method
   - Internet Service
   - Gender, Partner, Dependents, Phone Service, Paperless Billing

4. The model will output the churn prediction and probability

#### Option 2: Programmatic Usage

```python
import pickle
import pandas as pd

# Load the saved model
with open("churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
model_columns = model_data["columns"]

# Prepare your customer data (must match model_columns format)
# ... preprocessing code ...

# Make prediction
prediction = model.predict(customer_data)
probability = model.predict_proba(customer_data)[:, 1]
```

## Methodology

### 1. Data Preprocessing

- **Missing Values**: Identified and handled empty strings in `TotalCharges` column
- **Data Type Conversion**: Converted `TotalCharges` from string to float
- **Feature Encoding**:
  - Binary features (Yes/No) converted to 1/0
  - Categorical features one-hot encoded using `pd.get_dummies()`
- **Feature Selection**: Removed `customerID` (identifier, not predictive)

### 2. Exploratory Data Analysis

- Distribution analysis of churn rates
- Correlation analysis between numerical features
- Visualization of relationships between contract types, charges, and churn
- Feature distribution analysis

### 3. Model Training

#### Models Evaluated:
1. **Logistic Regression** (Baseline)
   - Class weight balancing for imbalanced data
   - Accuracy: ~74%

2. **Random Forest Classifier**
   - Initial model with class weight balancing
   - Accuracy: ~78%
   - Improved with SMOTE oversampling
   - Final tuned model with GridSearchCV

#### Hyperparameter Tuning:
- **Method**: GridSearchCV with 5-fold cross-validation
- **Scoring Metric**: F1-score (balanced for imbalanced classes)
- **Parameters Tuned**:
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2', None]

### 4. Class Imbalance Handling

- **SMOTE (Synthetic Minority Oversampling Technique)**: Applied to training data to balance classes
- Result: Balanced dataset with 4,130 samples per class

## Model Performance

### Final Model (Random Forest with GridSearchCV)

- **Best Hyperparameters**:
  - `n_estimators`: 500
  - `max_depth`: None
  - `max_features`: 'log2'
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2

- **Performance Metrics**:
  - **Accuracy**: 76.33%
  - **F1-Score**: 0.76 (weighted average)
  - **Precision (Churn)**: 0.55
  - **Recall (Churn)**: 0.56
  - **Precision (No Churn)**: 0.84
  - **Recall (No Churn)**: 0.84

### Confusion Matrix
```
                Predicted
              No Churn  Churn
Actual No Churn   864     169
       Churn      164     210
```

## Key Insights

### Top Features Affecting Churn (by importance):

1. **PaymentMethod_Electronic check** (21.06%) - Highest churn risk
2. **Tenure** (13.41%) - Longer tenure = lower churn
3. **InternetService_Fiber optic** (10.91%) - Higher churn risk
4. **Contract_Two year** (10.29%) - Lower churn risk
5. **TotalCharges** (7.63%) - Higher total charges = lower churn
6. **Partner** (5.91%) - Having a partner reduces churn
7. **Dependents** (5.66%) - Having dependents reduces churn
8. **MonthlyCharges** (3.51%)

### Business Recommendations:

1. **Payment Method**: Customers using electronic checks are at highest risk - consider incentives for automatic payment methods
2. **Contract Type**: Two-year contracts significantly reduce churn - promote longer-term contracts
3. **Tenure**: Focus retention efforts on newer customers (lower tenure)
4. **Fiber Optic Service**: Investigate why fiber optic customers churn more - may need better support or pricing
5. **Family Status**: Customers with partners and dependents are more stable - target family-oriented retention programs

## Technologies Used

- **Python 3.x**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Model Persistence**: pickle
- **Interactive Interface**: streamlit (for interface notebook)

## Notes

- The model was trained on a specific dataset and may require retraining for different customer populations
- Model performance metrics are based on a 80/20 train-test split
- The saved model (`churn_model.pkl`) includes both the trained model and the feature column names for proper data alignment
- Version compatibility warnings may appear when loading the model if scikit-learn versions differ between training and inference environments

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

This project is for educational purposes. The dataset is publicly available and commonly used for machine learning practice.

## Acknowledgments

- Dataset: Telco Customer Churn dataset (commonly used in ML tutorials)
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn

---

**Last Updated**: 2025
