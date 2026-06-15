"""
Example Usage of the Customer Churn Prediction API

This script demonstrates how to use the Flask API for making predictions.
Run main.py first to start the server, then run this script.
"""

import requests
import json
from typing import Dict, List

# API Base URL
BASE_URL = "http://localhost:5000"


def print_response(response: requests.Response, title: str):
    """Pretty print API response."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Status Code: {response.status_code}")
        print(f"Error: {response.json()}")


def check_health():
    """Check API health."""
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response, "API Health Check")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("   Make sure to run 'python main.py' first")
        return False
    return True


def get_model_info():
    """Get model information and encodings."""
    response = requests.get(f"{BASE_URL}/model_info")
    print_response(response, "Model Information")


def predict_single_customer():
    """Make a prediction for a single customer."""
    customer_data = {
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
    
    print("\n" + "="*70)
    print("  Prediction Request (Single Customer)")
    print("="*70)
    print("Input:")
    print(json.dumps(customer_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print_response(response, "Prediction Response")


def predict_batch_customers():
    """Make predictions for multiple customers."""
    customers = [
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
    
    print("\n" + "="*70)
    print("  Batch Prediction Request (Multiple Customers)")
    print("="*70)
    print(f"Input: {len(customers)} customers")
    
    batch_data = {"customers": customers}
    response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
    print_response(response, "Batch Prediction Response")


def test_edge_cases():
    """Test edge cases and error handling."""
    
    # Test 1: Missing fields
    print("\n" + "="*70)
    print("  Test 1: Missing Required Fields (Error Test)")
    print("="*70)
    incomplete_customer = {
        "CreditScore": 750,
        "Age": 42,
        "Tenure": 8
        # Missing other fields
    }
    print("Input: Incomplete customer data")
    response = requests.post(f"{BASE_URL}/predict", json=incomplete_customer)
    print_response(response, "Error Response")
    
    # Test 2: Invalid Geography
    print("\n" + "="*70)
    print("  Test 2: Invalid Geography Value (Error Test)")
    print("="*70)
    invalid_customer = {
        "CreditScore": 750,
        "Age": 42,
        "Tenure": 8,
        "Balance": 125000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 150000,
        "Geography": "InvalidCountry",  # Invalid
        "Gender": "Male"
    }
    print("Input: Invalid geography")
    response = requests.post(f"{BASE_URL}/predict", json=invalid_customer)
    print_response(response, "Error Response")
    
    # Test 3: Invalid Gender
    print("\n" + "="*70)
    print("  Test 3: Invalid Gender Value (Error Test)")
    print("="*70)
    invalid_gender = {
        "CreditScore": 750,
        "Age": 42,
        "Tenure": 8,
        "Balance": 125000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 150000,
        "Geography": "Germany",
        "Gender": "Other"  # Invalid
    }
    print("Input: Invalid gender")
    response = requests.post(f"{BASE_URL}/predict", json=invalid_gender)
    print_response(response, "Error Response")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  CUSTOMER CHURN PREDICTION API - EXAMPLES")
    print("="*70)
    
    # Check health
    if not check_health():
        return
    
    # Get model info
    get_model_info()
    
    # Single prediction
    predict_single_customer()
    
    # Batch prediction
    predict_batch_customers()
    
    # Error handling
    test_edge_cases()
    
    print("\n" + "="*70)
    print("  ✓ All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
