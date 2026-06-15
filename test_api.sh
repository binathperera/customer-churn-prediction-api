#!/bin/bash
# API Test Script using curl

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:5000"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Customer Churn Prediction API${NC}"
echo -e "${BLUE}CURL Test Script${NC}"
echo -e "${BLUE}================================${NC}\n"

# Test 1: Health Check
echo -e "${GREEN}Test 1: Health Check${NC}"
curl -X GET ${BASE_URL}/ \
  -H "Content-Type: application/json"
echo -e "\n\n"

# Test 2: Model Info
echo -e "${GREEN}Test 2: Model Information${NC}"
curl -X GET ${BASE_URL}/model_info \
  -H "Content-Type: application/json"
echo -e "\n\n"

# Test 3: Single Prediction - Customer Likely to Stay
echo -e "${GREEN}Test 3: Single Prediction (Customer Likely to Stay)${NC}"
curl -X POST ${BASE_URL}/predict \
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
echo -e "\n\n"

# Test 4: Single Prediction - Customer Likely to Leave
echo -e "${GREEN}Test 4: Single Prediction (Customer Likely to Leave)${NC}"
curl -X POST ${BASE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Age": 55,
    "Tenure": 2,
    "Balance": 5000,
    "NumOfProducts": 1,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000,
    "Geography": "Germany",
    "Gender": "Female"
  }'
echo -e "\n\n"

# Test 5: Batch Prediction
echo -e "${GREEN}Test 5: Batch Prediction (Multiple Customers)${NC}"
curl -X POST ${BASE_URL}/batch_predict \
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
      },
      {
        "CreditScore": 850,
        "Age": 35,
        "Tenure": 10,
        "Balance": 200000,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 200000,
        "Geography": "Spain",
        "Gender": "Male"
      }
    ]
  }'
echo -e "\n\n"

# Test 6: Error Test - Missing Fields
echo -e "${GREEN}Test 6: Error Test - Missing Required Fields${NC}"
curl -X POST ${BASE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 750,
    "Age": 42
  }'
echo -e "\n\n"

# Test 7: Error Test - Invalid Geography
echo -e "${GREEN}Test 7: Error Test - Invalid Geography${NC}"
curl -X POST ${BASE_URL}/predict \
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
    "Geography": "InvalidCountry",
    "Gender": "Male"
  }'
echo -e "\n\n"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}✓ Tests Complete${NC}"
echo -e "${BLUE}================================${NC}\n"
