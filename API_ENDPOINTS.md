# Complete API Endpoints Reference

This document provides a comprehensive reference for all available endpoints in the Customer Churn Prediction API.

## Endpoint Categories

1. **Utility Endpoints** - Health check and info
2. **Prediction Endpoints** - Make predictions
3. **CRUD Endpoints** - Manage customer data
4. **Search & Analytics** - Search and statistics

---

## Utility Endpoints

### GET `/`
**Health Check** - Verify API is running

```bash
curl http://localhost:5000/
```

**Response (200):**
```json
{
  "message": "Customer Churn Prediction API",
  "endpoint": "/predict",
  "method": "POST"
}
```

### GET `/model_info`
**Get Model Information** - Retrieve model configuration and feature encodings

```bash
curl http://localhost:5000/model_info
```

**Response (200):**
```json
{
  "model_type": "LightGBM",
  "n_estimators": 200,
  "learning_rate": 0.1,
  "max_depth": 5,
  "features": [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography", "Gender"
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

---

## Prediction Endpoints

### POST `/predict`
**Single Customer Prediction** - Predict churn for one customer

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

**Response (200):**
```json
{
  "prediction": {
    "churn": 0,
    "churn_probability": 0.2847,
    "status": "Will stay",
    "confidence": 71.53
  },
  "customer_input": { ... }
}
```

**Error (400):**
```json
{
  "error": "Missing fields: Age, Geography"
}
```

### POST `/batch_predict`
**Batch Predictions** - Predict churn for multiple customers

**Request:**
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { ... customer1 data ... },
      { ... customer2 data ... }
    ]
  }'
```

**Response (200):**
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

---

## CRUD Endpoints

### POST `/data`
**Create Customer** - Add new customer record

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

**Response (201):**
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

### GET `/data`
**Get All Customers** - Retrieve all stored customer records

**Request:**
```bash
curl http://localhost:5000/data
```

**Response (200):**
```json
{
  "total_customers": 3,
  "customers": [
    {
      "id": "a1b2c3d4",
      "data": { ... },
      "created_at": "2024-01-15T10:30:45.123456",
      "updated_at": "2024-01-15T10:30:45.123456"
    }
  ]
}
```

### GET `/data/<customer_id>`
**Get Specific Customer** - Retrieve a customer by ID

**Request:**
```bash
curl http://localhost:5000/data/a1b2c3d4
```

**Response (200):**
```json
{
  "customer_id": "a1b2c3d4",
  "customer": { ... }
}
```

**Error (404):**
```json
{
  "error": "Customer a1b2c3d4 not found"
}
```

### PUT `/data/<customer_id>`
**Update Customer** - Modify customer record (supports partial updates)

**Request:**
```bash
curl -X PUT http://localhost:5000/data/a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 43,
    "Balance": 130000
  }'
```

**Response (200):**
```json
{
  "message": "Customer updated successfully",
  "customer_id": "a1b2c3d4",
  "customer": { ... }
}
```

**Error (404):**
```json
{
  "error": "Customer a1b2c3d4 not found"
}
```

### DELETE `/data/<customer_id>`
**Delete Customer** - Remove a customer record

**Request:**
```bash
curl -X DELETE http://localhost:5000/data/a1b2c3d4
```

**Response (200):**
```json
{
  "message": "Customer deleted successfully",
  "customer_id": "a1b2c3d4",
  "deleted_customer": { ... }
}
```

### DELETE `/data`
**Delete All Customers** - Remove all customer records

**Request:**
```bash
curl -X DELETE http://localhost:5000/data
```

**Response (200):**
```json
{
  "message": "All customers deleted successfully",
  "deleted_count": 3
}
```

---

## Search & Analytics Endpoints

### POST `/data/search`
**Search Customers** - Find customers by criteria

**Request:**
```bash
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{
    "field": "Geography",
    "value": "Germany"
  }'
```

**Response (200):**
```json
{
  "search_criteria": {
    "field": "Geography",
    "value": "Germany"
  },
  "results_count": 2,
  "results": [
    {
      "customer_id": "a1b2c3d4",
      "customer": { ... }
    }
  ]
}
```

**Example Searches:**
```bash
# By Gender
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "Gender", "value": "Female"}'

# By NumOfProducts
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "NumOfProducts", "value": 2}'

# By IsActiveMember
curl -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "IsActiveMember", "value": 1}'
```

### GET `/data/stats`
**Get Statistics** - Retrieve aggregated statistics about stored customers

**Request:**
```bash
curl http://localhost:5000/data/stats
```

**Response (200):**
```json
{
  "total_customers": 3,
  "geography_distribution": {
    "Germany": 2,
    "France": 1
  },
  "gender_distribution": {
    "Male": 2,
    "Female": 1
  },
  "average_age": 45.67,
  "average_tenure": 7.33,
  "average_balance": 118333.33,
  "average_salary": 150000.0,
  "oldest_customer_age": 55,
  "youngest_customer_age": 35
}
```

---

## HTTP Status Codes

| Code | Meaning | Examples |
|------|---------|----------|
| 200 | OK | GET successful, PUT successful, DELETE successful |
| 201 | Created | POST customer created successfully |
| 400 | Bad Request | Missing required fields, invalid data format |
| 404 | Not Found | Customer ID doesn't exist |
| 500 | Server Error | Internal processing error |

---

## Common Workflows

### Workflow 1: Create and Predict
```bash
# 1. Create customer
CUSTOMER=$(curl -s -X POST http://localhost:5000/data \
  -H "Content-Type: application/json" \
  -d '{...customer data...}')
CUSTOMER_ID=$(echo $CUSTOMER | jq -r '.customer_id')

# 2. Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{...same customer data...}'
```

### Workflow 2: Search and Batch Predict
```bash
# 1. Search for customers
RESULTS=$(curl -s -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "Geography", "value": "Germany"}')

# 2. Extract customer data and make batch predictions
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{...customers data...}'
```

### Workflow 3: Data Analysis
```bash
# 1. Get all customers
ALL=$(curl -s http://localhost:5000/data)

# 2. Get statistics
STATS=$(curl -s http://localhost:5000/data/stats)

# 3. Search specific subset
SUBSET=$(curl -s -X POST http://localhost:5000/data/search \
  -H "Content-Type: application/json" \
  -d '{"field": "IsActiveMember", "value": 1}')
```

### Workflow 4: Data Management
```bash
# 1. Create customer
CUSTOMER=$(curl -s -X POST http://localhost:5000/data ...)
ID=$(echo $CUSTOMER | jq -r '.customer_id')

# 2. Update customer
curl -X PUT http://localhost:5000/data/$ID \
  -H "Content-Type: application/json" \
  -d '{"Age": 43}'

# 3. Verify update
curl http://localhost:5000/data/$ID

# 4. Delete customer
curl -X DELETE http://localhost:5000/data/$ID
```

---

## Data Model

### Customer Object
```json
{
  "id": "a1b2c3d4",
  "data": {
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
  "created_at": "2024-01-15T10:30:45.123456",
  "updated_at": "2024-01-15T10:30:45.123456"
}
```

### Prediction Object
```json
{
  "churn": 0,
  "churn_probability": 0.2847,
  "status": "Will stay",
  "confidence": 71.53
}
```

---

## Field Reference

### Customer Fields
| Field | Type | Range | Example |
|-------|------|-------|---------|
| CreditScore | float | 300-850 | 750 |
| Age | float | 18-100 | 42 |
| Tenure | float | 0-40 | 8 |
| Balance | float | 0-1000000 | 125000 |
| NumOfProducts | int | 1-4 | 2 |
| HasCrCard | int | 0-1 | 1 |
| IsActiveMember | int | 0-1 | 1 |
| EstimatedSalary | float | 0-1000000 | 150000 |
| Geography | string | France, Germany, Spain | Germany |
| Gender | string | Male, Female | Male |

---

## Error Handling

### Example Error Responses

**Missing Fields (400):**
```json
{
  "error": "Missing fields: Age, Geography"
}
```

**Invalid Fields (400):**
```json
{
  "error": "Invalid fields: InvalidField"
}
```

**Not Found (404):**
```json
{
  "error": "Customer a1b2c3d4 not found"
}
```

**Server Error (500):**
```json
{
  "error": "Error creating customer: Internal server error"
}
```

---

## Rate Limiting & Performance

- No built-in rate limiting
- All operations are synchronous
- Average response time: < 100ms
- Supports batch predictions up to 1000 customers
- JSON file persistence: < 10ms

---

## Authentication

Currently, no authentication is implemented. For production use, consider adding:
- API key authentication
- JWT tokens
- OAuth2
- Rate limiting

---

## CORS & Headers

- Content-Type: application/json (required for POST/PUT)
- CORS: All origins allowed (localhost only recommended for production)

---

See full documentation in:
- [CRUD_ENDPOINTS.md](CRUD_ENDPOINTS.md) - Detailed CRUD operations
- [README.md](README.md) - Complete guide
- [QUICKSTART.md](QUICKSTART.md) - Quick setup
