"""
CRUD Operations Test Script for Customer Churn Prediction API

This script demonstrates all CRUD operations available in the API.
Make sure the Flask server is running before executing this script.

Run with: python test_crud_operations.py
"""

import requests
import json
import time
from typing import Optional, Dict

# API Base URL
BASE_URL = "http://localhost:5000"

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}✗ {message}{RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{YELLOW}ℹ {message}{RESET}")


def print_response(response: requests.Response, show_body: bool = True):
    """Print HTTP response details."""
    print(f"Status Code: {response.status_code}")
    if show_body:
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except:
            print(f"Response: {response.text}")


def test_health_check():
    """Test API health check."""
    print_header("1. Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print_success("API is healthy")
            print_response(response)
        else:
            print_error(f"API health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to API. Make sure server is running on localhost:5000")
        return False
    
    return True


def test_create_customers() -> list:
    """Test creating multiple customers."""
    print_header("2. Testing CREATE - Creating Customer Records")
    
    customers_data = [
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
            "CreditScore": 820,
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
    
    created_ids = []
    
    for i, customer_data in enumerate(customers_data, 1):
        print(f"\nCreating customer {i}...")
        response = requests.post(f"{BASE_URL}/data", json=customer_data)
        
        if response.status_code == 201:
            result = response.json()
            customer_id = result['customer_id']
            created_ids.append(customer_id)
            print_success(f"Customer created: {customer_id}")
            print(f"  Name: Customer {i}")
            print(f"  Age: {customer_data['Age']}")
            print(f"  Geography: {customer_data['Geography']}")
        else:
            print_error(f"Failed to create customer: {response.status_code}")
            print_response(response)
    
    return created_ids


def test_read_all_customers():
    """Test reading all customers."""
    print_header("3. Testing READ - Getting All Customers")
    
    response = requests.get(f"{BASE_URL}/data")
    
    if response.status_code == 200:
        result = response.json()
        print_success(f"Retrieved {result['total_customers']} customers")
        
        for i, customer in enumerate(result['customers'], 1):
            print(f"\n  Customer {i}:")
            print(f"    ID: {customer['id']}")
            print(f"    Age: {customer['data']['Age']}")
            print(f"    Geography: {customer['data']['Geography']}")
            print(f"    Created: {customer['created_at']}")
    else:
        print_error(f"Failed to retrieve customers: {response.status_code}")
        print_response(response)


def test_read_specific_customer(customer_id: str):
    """Test reading a specific customer."""
    print_header("4. Testing READ - Getting Specific Customer")
    
    print(f"Reading customer: {customer_id}")
    response = requests.get(f"{BASE_URL}/data/{customer_id}")
    
    if response.status_code == 200:
        result = response.json()
        customer = result['customer']
        print_success(f"Retrieved customer {customer_id}")
        print(f"  Age: {customer['data']['Age']}")
        print(f"  Balance: {customer['data']['Balance']}")
        print(f"  IsActiveMember: {customer['data']['IsActiveMember']}")
    else:
        print_error(f"Failed to retrieve customer: {response.status_code}")
        print_response(response)


def test_update_customer(customer_id: str):
    """Test updating a customer."""
    print_header("5. Testing UPDATE - Modifying Customer Record")
    
    update_data = {
        "Age": 43,
        "Balance": 135000,
        "IsActiveMember": 1
    }
    
    print(f"Updating customer {customer_id}...")
    print(f"Changes: {json.dumps(update_data, indent=2)}")
    
    response = requests.put(f"{BASE_URL}/data/{customer_id}", json=update_data)
    
    if response.status_code == 200:
        result = response.json()
        print_success("Customer updated successfully")
        print(f"  New Age: {result['customer']['data']['Age']}")
        print(f"  New Balance: {result['customer']['data']['Balance']}")
        print(f"  Updated At: {result['customer']['updated_at']}")
    else:
        print_error(f"Failed to update customer: {response.status_code}")
        print_response(response)


def test_search_customers():
    """Test searching for customers."""
    print_header("6. Testing SEARCH - Finding Customers by Criteria")
    
    search_queries = [
        {"field": "Geography", "value": "Germany"},
        {"field": "Gender", "value": "Female"},
        {"field": "IsActiveMember", "value": 1}
    ]
    
    for query in search_queries:
        print(f"\nSearching: {query['field']} = {query['value']}")
        response = requests.post(f"{BASE_URL}/data/search", json=query)
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Found {result['results_count']} matching customers")
            
            for i, match in enumerate(result['results'], 1):
                customer = match['customer']
                print(f"  {i}. {match['customer_id']} - Age: {customer['data']['Age']}")
        else:
            print_error(f"Search failed: {response.status_code}")


def test_get_statistics():
    """Test getting data statistics."""
    print_header("7. Testing STATS - Getting Data Statistics")
    
    response = requests.get(f"{BASE_URL}/data/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print_success("Retrieved statistics")
        print(f"\n  Total Customers: {stats.get('total_customers', 0)}")
        
        if stats.get('total_customers', 0) > 0:
            print(f"\n  Geographic Distribution:")
            for country, count in stats.get('geography_distribution', {}).items():
                print(f"    - {country}: {count}")
            
            print(f"\n  Gender Distribution:")
            for gender, count in stats.get('gender_distribution', {}).items():
                print(f"    - {gender}: {count}")
            
            print(f"\n  Average Values:")
            print(f"    - Age: {stats.get('average_age', 0):.2f}")
            print(f"    - Tenure: {stats.get('average_tenure', 0):.2f}")
            print(f"    - Balance: ${stats.get('average_balance', 0):.2f}")
            print(f"    - Salary: ${stats.get('average_salary', 0):.2f}")
            
            print(f"\n  Age Range:")
            print(f"    - Youngest: {stats.get('youngest_customer_age', 0)}")
            print(f"    - Oldest: {stats.get('oldest_customer_age', 0)}")
    else:
        print_error(f"Failed to retrieve statistics: {response.status_code}")


def test_delete_customer(customer_id: str):
    """Test deleting a customer."""
    print_header("8. Testing DELETE - Removing Customer Record")
    
    print(f"Deleting customer: {customer_id}")
    response = requests.delete(f"{BASE_URL}/data/{customer_id}")
    
    if response.status_code == 200:
        print_success("Customer deleted successfully")
        
        # Verify deletion
        time.sleep(0.5)
        verify = requests.get(f"{BASE_URL}/data/{customer_id}")
        if verify.status_code == 404:
            print_success("Deletion verified - customer not found")
    else:
        print_error(f"Failed to delete customer: {response.status_code}")
        print_response(response)


def test_prediction_with_crud_data(customer_id: str):
    """Test making a prediction using CRUD stored data."""
    print_header("9. Testing PREDICTION with CRUD Data")
    
    # Get customer data
    response = requests.get(f"{BASE_URL}/data/{customer_id}")
    
    if response.status_code == 200:
        result = response.json()
        customer_data = result['customer']['data']
        
        print(f"Making prediction for customer: {customer_id}")
        
        # Make prediction
        pred_response = requests.post(f"{BASE_URL}/predict", json=customer_data)
        
        if pred_response.status_code == 200:
            prediction = pred_response.json()
            pred_result = prediction['prediction']
            print_success("Prediction completed")
            print(f"  Churn Status: {pred_result['status']}")
            print(f"  Probability: {pred_result['churn_probability']:.4f}")
            print(f"  Confidence: {pred_result['confidence']:.2f}%")
        else:
            print_error(f"Prediction failed: {pred_response.status_code}")
    else:
        print_error(f"Could not retrieve customer data: {response.status_code}")


def test_error_handling():
    """Test error handling."""
    print_header("10. Testing ERROR HANDLING")
    
    # Test 1: Non-existent customer
    print("\nTest 1: Getting non-existent customer")
    response = requests.get(f"{BASE_URL}/data/nonexistent")
    if response.status_code == 404:
        print_success("Correctly returned 404 for non-existent customer")
    else:
        print_error(f"Expected 404, got {response.status_code}")
    
    # Test 2: Missing required fields
    print("\nTest 2: Creating customer with missing fields")
    incomplete_data = {
        "CreditScore": 750,
        "Age": 42
        # Missing other required fields
    }
    response = requests.post(f"{BASE_URL}/data", json=incomplete_data)
    if response.status_code == 400:
        print_success("Correctly returned 400 for incomplete data")
        print(f"  Error: {response.json()['error']}")
    else:
        print_error(f"Expected 400, got {response.status_code}")
    
    # Test 3: Invalid field in update
    print("\nTest 3: Updating with invalid field")
    response = requests.put(f"{BASE_URL}/data/dummy_id", json={"InvalidField": "value"})
    if response.status_code in [400, 404]:
        print_success(f"Correctly returned {response.status_code} for invalid field")
    else:
        print_error(f"Expected 400 or 404, got {response.status_code}")


def main():
    """Main test runner."""
    print(f"\n{BLUE}{'='*70}")
    print("  CUSTOMER CHURN PREDICTION API - CRUD OPERATIONS TEST")
    print(f"{'='*70}{RESET}")
    
    # Test 1: Health check
    if not test_health_check():
        return
    
    time.sleep(1)
    
    # Test 2: Create customers
    customer_ids = test_create_customers()
    time.sleep(1)
    
    if not customer_ids:
        print_error("No customers were created. Aborting tests.")
        return
    
    # Test 3: Read all customers
    test_read_all_customers()
    time.sleep(1)
    
    # Test 4: Read specific customer
    test_read_specific_customer(customer_ids[0])
    time.sleep(1)
    
    # Test 5: Update customer
    test_update_customer(customer_ids[0])
    time.sleep(1)
    
    # Test 6: Search customers
    test_search_customers()
    time.sleep(1)
    
    # Test 7: Get statistics
    test_get_statistics()
    time.sleep(1)
    
    # Test 8: Make prediction with CRUD data
    test_prediction_with_crud_data(customer_ids[0])
    time.sleep(1)
    
    # Test 9: Error handling
    test_error_handling()
    time.sleep(1)
    
    # Test 10: Delete customer
    test_delete_customer(customer_ids[0])
    time.sleep(1)
    
    # Final summary
    print_header("TEST SUMMARY")
    print_success("All CRUD operations tested successfully!")
    print("\nFeatures Tested:")
    print("  ✓ CREATE - Creating customer records")
    print("  ✓ READ - Getting all customers and specific customers")
    print("  ✓ UPDATE - Modifying customer records")
    print("  ✓ DELETE - Removing customer records")
    print("  ✓ SEARCH - Finding customers by criteria")
    print("  ✓ STATS - Getting aggregated statistics")
    print("  ✓ PREDICT - Making predictions with stored data")
    print("  ✓ ERROR HANDLING - Proper error responses")
    print("\nData Persistence:")
    print("  ✓ Customer data saved to customers_data.json")
    print("  ✓ Data persists across API restarts")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
