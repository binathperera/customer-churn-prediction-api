# Customer Churn Prediction API

A Flask-based REST API for predicting customer churn using a LightGBM machine learning model, based on the Bank Churn dataset.

## Overview

This application implements the complete machine learning pipeline from `bank_churn_project.py`:

- **Data Preprocessing**: Feature scaling, categorical encoding, outlier removal
- **Baseline Model**: LightGBM with default parameters for comparison
- **Hyperparameter Tuning**: Automatic optimization using Optuna (100 trials, 5-fold CV)
- **Final Model**: LightGBM trained with best hyperparameters
- **Deployment**: Flask REST API for single and batch predictions

## Project Structure

```
customer-churn-prediction/
├── main.py              # Flask application with prediction endpoints
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── lgbm_model.pkl      # Trained LightGBM model (generated after training)
├── scaler.pkl          # Feature scaler (generated after training)
├── encoders.pkl        # Category encoders (generated after training)
└── README.md           # This file
```

## Installation

### 1. Clone/Setup Repository

```bash
cd customer-churn-prediction
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- Flask
- scikit-learn
- lightgbm
- pandas
- numpy
- imbalanced-learn (for SMOTE)

## Usage

### Step 1: Train the Model

First, you need to train the model with the Churn_Modelling.csv dataset:

```bash
python train_model.py path/to/Churn_Modelling.csv
```

**Example:**

```bash
python train_model.py data/Churn_Modelling.csv
```

This will:

1. Load the dataset
2. Apply preprocessing (drop columns, remove outliers, encode categories, scale features)
3. Handle class imbalance with SMOTE
4. **Train baseline model** with default parameters
5. **Run hyperparameter optimization** using Optuna:
   - Tests 100 different hyperparameter combinations
   - Uses 5-fold stratified cross-validation
   - Optimizes for maximum F1 score
   - Takes ~5-10 minutes depending on hardware
6. **Train final model** with best hyperparameters found
7. Display improvement metrics (accuracy, F1, ROC-AUC)
8. Save artifacts:
   - `lgbm_model.pkl` - Trained model with optimized hyperparameters
   - `scaler.pkl` - StandardScaler
   - `encoders.pkl` - LabelEncoders

**Sample Output:**

```
============================================================
MODEL IMPROVEMENT SUMMARY
============================================================

Baseline Model (Default Parameters):
  Accuracy : 84.50 %
  F1 Score : 75.20 %
  ROC-AUC  : 86.30 %

Optimized Model (Best Hyperparameters):
  Accuracy : 86.75 %
  F1 Score : 78.56 %
  ROC-AUC  : 88.45 %

Improvement:
  Accuracy : +2.25 %
  F1 Score : +3.36 %
  ROC-AUC  : +2.15 %
```

### Step 2: Run the Flask API

```bash
python main.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. **Health Check** - GET `/`

Returns API information.

**Response:**

```json
{
  "message": "Customer Churn Prediction API",
  "endpoint": "/predict",
  "method": "POST"
}
```

### 2. **Single Prediction** - POST `/predict`

Predict churn for a single customer.

**Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 750,
    "Age": 42,
    "Tenure": 8,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 150000,
    "Geography": "Germany",
    "Gender": "Male"
  }'
```

**Response:**

```json
{
  "prediction": {
    "churn": 0,
    "churn_probability": 0.2847,
    "status": "Will stay",
    "confidence": 71.53
  },
  "customer_input": {
    "CreditScore": 750,
    "Age": 42,
    "Tenure": 8,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 150000,
    "Geography": "Germany",
    "Gender": "Male"
  }
}
```

### 3. **Batch Prediction** - POST `/batch_predict`

Predict churn for multiple customers at once.

**Request:**

```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "CreditScore": 750,
        "Age": 42,
        "Tenure": 8,
        "Balance": 125000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 150000,
        "Geography": "Germany",
        "Gender": "Male"
      },
      {
        "CreditScore": 600,
        "Age": 55,
        "Tenure": 2,
        "Balance": 50000,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 100000,
        "Geography": "France",
        "Gender": "Female"
      }
    ]
  }'
```

**Response:**

```json
{
  "total_customers": 2,
  "predictions": [
    {
      "customer_index": 0,
      "prediction": {
        "churn": 0,
        "churn_probability": 0.2847,
        "status": "Will stay",
        "confidence": 71.53
      }
    },
    {
      "customer_index": 1,
      "prediction": {
        "churn": 1,
        "churn_probability": 0.7156,
        "status": "Will leave",
        "confidence": 71.56
      }
    }
  ]
}
```

### 4. **Model Information** - GET `/model_info`

Get information about the trained model and its encodings.

**Request:**

```bash
curl http://localhost:5000/model_info
```

**Response:**

```json
{
  "model_type": "LightGBM",
  "n_estimators": 200,
  "learning_rate": 0.1,
  "max_depth": 5,
  "features": [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography",
    "Gender"
  ],
  "geography_encoding": {
    "France": 0,
    "Germany": 1,
    "Spain": 2
  },
  "gender_encoding": {
    "Female": 0,
    "Male": 1
  },
  "target": {
    "0": "Customer stayed",
    "1": "Customer left (churned)"
  }
}
```

## CRUD Endpoints for Data Management

The API provides complete CRUD operations for managing customer data with persistent storage.

### 5. **Get All Customers** - GET `/data`

Retrieve all stored customer records.

**Request:**
```bash
curl -X GET http://localhost:5000/data
```

**Response:**
```json
{
  "total_customers": 2,
  "customers": [
    {
      "id": "a1b2c3d4",
      "data": {
        "CreditScore": 750,
        "Age": 42,
        ...
      },
      "created_at": "2024-01-15T10:30:45.123456",
      "updated_at": "2024-01-15T10:30:45.123456"
    }
  ]
}
```

### 6. **Create Customer** - POST `/data`

Create a new customer record with auto-generated ID.

**Request:**
```bash
curl -X POST http://localhost:5000/data \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 750,
    "Age": 42,
    "Tenure": 8,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 150000,
    "Geography": "Germany",
    "Gender": "Male"
  }'
```

**Response (201 Created):**
```json
{
  "message": "Customer created successfully",
  "customer_id": "a1b2c3d4",
  "customer": {
    "id": "a1b2c3d4",
    "data": { ... },
    "created_at": "2024-01-15T10:30:45.123456",
    "updated_at": "2024-01-15T10:30:45.123456"
  }
}
```

### 7. **Get Specific Customer** - GET `/data/<customer_id>`

Retrieve a specific customer by ID.

**Request:**
```bash
curl -X GET http://localhost:5000/data/a1b2c3d4
```

**Response:**
```json
{
  "customer_id": "a1b2c3d4",
  "customer": { ... }
}
```

### 8. **Update Customer** - PUT `/data/<customer_id>`

Update specific fields of a customer (supports partial updates).

**Request:**
```bash
curl -X PUT http://localhost:5000/data/a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 43,
    "Balance": 130000
  }'
```

**Response:**
```json
{
  "message": "Customer updated successfully",
  "customer_id": "a1b2c3d4",
  "customer": { ... }
}
```

### 9. **Delete Customer** - DELETE `/data/<customer_id>`

Delete a specific customer record.

**Request:**
```bash
curl -X DELETE http://localhost:5000/data/a1b2c3d4
```

**Response:**
```json
{
  "message": "Customer deleted successfully",
  "customer_id": "a1b2c3d4",
  "deleted_customer": { ... }
}
```

### 10. **Search Customers** - POST `/data/search`

Search for customers by specific criteria.

**Request:**
```bash
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{
    "field": "Geography",
    "value": "Germany"
  }'
```

**Response:**
```json
{
  "search_criteria": { "field": "Geography", "value": "Germany" },
  "results_count": 1,
  "results": [ ... ]
}
```

### 11. **Get Data Statistics** - GET `/data/stats`

Get aggregated statistics about all stored customers.

**Request:**
```bash
curl -X GET http://localhost:5000/data/stats
```

**Response:**
```json
{
  "total_customers": 2,
  "geography_distribution": { "Germany": 1, "France": 1 },
  "gender_distribution": { "Male": 1, "Female": 1 },
  "average_age": 48.5,
  "average_tenure": 8.5,
  "average_balance": 87500.0,
  "average_salary": 150000.0,
  "oldest_customer_age": 55,
  "youngest_customer_age": 42
}
```

### 12. **Delete All Customers** - DELETE `/data`

Delete all stored customer records.

**Request:**
```bash
curl -X DELETE http://localhost:5000/data
```

**Response:**
```json
{
  "message": "All customers deleted successfully",
  "deleted_count": 2
}
```

## Data Persistence

All customer records are automatically saved to `customers_data.json` and restored when the API restarts. This ensures data persistence across application restarts.

For complete CRUD endpoint documentation, see [CRUD_ENDPOINTS.md](CRUD_ENDPOINTS.md).

## Input Features

All features are required for predictions:

| Feature           | Type  | Description                     | Example |
| ----------------- | ----- | ------------------------------- | ------- |
| `CreditScore`     | float | Customer's credit score         | 750     |
| `Age`             | float | Customer's age in years         | 42      |
| `Tenure`          | float | Years as customer with bank     | 8       |
| `Balance`         | float | Account balance in currency     | 125000  |
| `NumOfProducts`   | int   | Number of products held         | 2       |
| `HasCrCard`       | int   | Has credit card (0/1)           | 1       |
| `IsActiveMember`  | int   | Is active member (0/1)          | 1       |
| `EstimatedSalary` | float | Estimated annual salary         | 150000  |
| `Geography`       | str   | Country: France, Germany, Spain | Germany |
| `Gender`          | str   | Gender: Male, Female            | Male    |

## Output Explanation

The prediction response includes:

- **`churn`** (0 or 1): Binary prediction
  - 0 = Customer will stay
  - 1 = Customer will leave (churn)

- **`churn_probability`** (0.0-1.0): Probability of churning
  - Higher values indicate higher churn likelihood

- **`status`**: Human-readable prediction
  - "Will stay" or "Will leave"

- **`confidence`** (0-100%): Model confidence in the prediction
  - Confidence = max(probability, 1-probability) × 100

## Preprocessing Pipeline

The model follows these preprocessing steps (matching bank_churn_project.py):

1. **Drop Irrelevant Columns**: RowNumber, CustomerId, Surname
2. **Outlier Removal**: IQR method on numerical features
3. **Categorical Encoding**:
   - Geography: France→0, Germany→1, Spain→2
   - Gender: Female→0, Male→1
4. **Feature Scaling**: StandardScaler (mean=0, std=1)
5. **Class Imbalance**: SMOTE (training only)

## Model Configuration

### Hyperparameter Tuning with Optuna

The model training includes automatic hyperparameter optimization using Optuna:

**Search Space:**

- **n_estimators**: 100-500 (number of boosting stages)
- **learning_rate**: 0.01-0.3 (step size for boosting)
- **max_depth**: 3-10 (maximum tree depth)
- **num_leaves**: 20-100 (maximum leaves per tree)
- **min_data_in_leaf**: 5-50 (minimum samples per leaf)
- **lambda_l1**: 0.0-10.0 (L1 regularization)
- **lambda_l2**: 0.0-10.0 (L2 regularization)
- **feature_fraction**: 0.4-1.0 (feature sampling rate)
- **bagging_fraction**: 0.4-1.0 (data sampling rate)
- **bagging_freq**: 1-10 (bagging frequency)
- **min_sum_hessian_in_leaf**: 1e-3-10.0 (minimum Hessian sum in leaf)

**Optimization Details:**

- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Trials**: 100 (configurable)
- **Evaluation Metric**: F1 Score (for imbalanced classification)
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Random State**: 42 (for reproducibility)

### Example Best Hyperparameters

After optimization, a typical best parameter set might look like:

```
n_estimators: 350
learning_rate: 0.052
max_depth: 7
num_leaves: 75
min_data_in_leaf: 12
lambda_l1: 2.34
lambda_l2: 0.56
feature_fraction: 0.92
bagging_fraction: 0.85
bagging_freq: 5
min_sum_hessian_in_leaf: 0.0015
```

## Testing with Python Requests

```python
import requests

# Single prediction
url = "http://localhost:5000/predict"
customer = {
    "CreditScore": 750,
    "Age": 42,
    "Tenure": 8,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 150000,
    "Geography": "Germany",
    "Gender": "Male"
}

response = requests.post(url, json=customer)
print(response.json())

# Batch prediction
url = "http://localhost:5000/batch_predict"
data = {"customers": [customer1, customer2, customer3]}
response = requests.post(url, json=data)
print(response.json())
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400**: Bad Request (missing fields, invalid data types)
- **500**: Internal Server Error (processing error)

**Example Error Response:**

```json
{
  "error": "Missing fields: Age, Geography"
}
```

## Troubleshooting

### Model Not Found

If you see "Model not found. Created placeholder model":

- Run `python train_model.py data/Churn_Modelling.csv` first
- Ensure CSV file path is correct

### Wrong Geography/Gender Values

Accepted values are case-sensitive:

- **Geography**: "France", "Germany", "Spain"
- **Gender**: "Male", "Female"

### Port Already in Use

If port 5000 is busy, modify main.py:

```python
if __name__ == '__main__':
    load_or_create_model()
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

## Dataset Schema (Churn_Modelling.csv)

The training dataset should have these columns:

```
RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure,
Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
```

## Performance Metrics (Example)

After training on the Bank Churn dataset:

- **Accuracy**: ~86%
- **F1 Score**: ~78%
- **ROC-AUC**: ~88%

## License

This project is based on the Bank Churn analysis and uses open-source libraries (scikit-learn, LightGBM, Flask).

## References

- Original Project: https://colab.research.google.com/drive/13teGShAamXWCouw3ZiomqWAmGBWCLTK8
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- scikit-learn Documentation: https://scikit-learn.org/
- SMOTE: https://imbalanced-learn.org/
