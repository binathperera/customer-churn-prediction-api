# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Train the Model

Download the `Churn_Modelling.csv` dataset, then run:

```bash
python train_model.py path/to/Churn_Modelling.csv
```

This will:

1. Load and preprocess the data
2. Train a **baseline model** with default parameters
3. **Optimize hyperparameters** using Optuna (100 trials)
4. Train the **final optimized model**
5. Compare and display improvements
6. Create:
   - `lgbm_model.pkl` (trained model with best hyperparameters)
   - `scaler.pkl` (feature scaler)
   - `encoders.pkl` (category encoders)

**Note:** Hyperparameter optimization takes ~5-10 minutes depending on your hardware.

## 3. Start the API Server

```bash
python main.py
```

Server runs on: `http://localhost:5000`

## 4. Make Predictions

### Single Customer Prediction

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

### Batch Prediction

```bash
python example_usage.py
```

### Get Model Info

```bash
curl http://localhost:5000/model_info
```

## 5. Manage Customer Data with CRUD Operations

The API also provides complete CRUD endpoints for managing customer data.

### Create a Customer
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

### Get All Customers
```bash
curl http://localhost:5000/data
```

### Get Specific Customer
```bash
curl http://localhost:5000/data/a1b2c3d4
```

### Update Customer
```bash
curl -X PUT http://localhost:5000/data/a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{"Age": 43, "Balance": 130000}'
```

### Search Customers
```bash
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "Geography", "value": "Germany"}'
```

### Get Statistics
```bash
curl http://localhost:5000/data/stats
```

### Delete Customer
```bash
curl -X DELETE http://localhost:5000/data/a1b2c3d4
```

### Run CRUD Tests
```bash
python test_crud_operations.py
```

See [CRUD_ENDPOINTS.md](CRUD_ENDPOINTS.md) for complete documentation.

## Key Points

- **Geography**: France, Germany, Spain (case-sensitive)
- **Gender**: Male, Female (case-sensitive)
- **Response** includes:
  - `churn`: 0 (will stay) or 1 (will leave)
  - `churn_probability`: likelihood of churn (0.0-1.0)
  - `confidence`: model confidence (%)

## Features Required for Prediction

All 10 features must be provided:

1. CreditScore (float)
2. Age (float)
3. Tenure (float)
4. Balance (float)
5. NumOfProducts (int)
6. HasCrCard (0 or 1)
7. IsActiveMember (0 or 1)
8. EstimatedSalary (float)
9. Geography (string)
10. Gender (string)

## Example Response

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
    ...
  }
}
```

## Training Output Example

The training script will show:

```
============================================================
CUSTOMER CHURN PREDICTION - MODEL TRAINING
With Hyperparameter Optimization
============================================================

[... data loading and preprocessing ...]

============================================================
BASELINE MODEL TRAINING
============================================================
✓ Baseline model trained successfully

BASELINE MODEL EVALUATION RESULTS
---
  Accuracy  : 84.50 %
  F1 Score  : 75.20 %
  ROC-AUC   : 86.30 %

============================================================
HYPERPARAMETER TUNING: LightGBM with Optuna
============================================================
Optimizing over 100 trials...
(This may take a few minutes)

[... optimization progress ...]

============================================================
OPTIMIZATION COMPLETE
============================================================

✓ Best F1 Score from Tuning: 0.7856

Best Hyperparameters Found:
  n_estimators: 350
  learning_rate: 0.052000
  max_depth: 7
  num_leaves: 75
  ... (more hyperparameters)

============================================================
FINAL MODEL TRAINING
With Best Hyperparameters from Optuna
============================================================
✓ Final model trained successfully

FINAL MODEL EVALUATION RESULTS
---
  Accuracy  : 86.75 %
  F1 Score  : 78.56 %
  ROC-AUC   : 88.45 %

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

## Troubleshooting

**Q: "Model not found. Created placeholder model"**
A: Run `python train_model.py data/Churn_Modelling.csv` first

**Q: Port 5000 already in use?**
A: Change port in main.py (last line)

**Q: Invalid geography/gender error?**
A: Use exact case: "France" not "france", "Male" not "male"

**Q: Hyperparameter tuning is taking too long**
A: Reduce n_trials in optimize_hyperparameters() call (default is 100)
