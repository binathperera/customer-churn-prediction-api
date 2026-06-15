import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
import os
import json
from datetime import datetime
import uuid
import psycopg2
import psycopg2.extras
import random
import boto3
from dotenv import load_dotenv

# 1. Load the environment variables from the .env file FIRST
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store model and preprocessors
model = None
scaler = None
le_geography = None
le_gender = None
model_path = "lgbm_model.pkl"
scaler_path = "scaler.pkl"
encoders_path = "encoders.pkl"

#PostgreSQL Database Configuration
# DB_PARAMS = {
#     "dbname": "bank",
#     "user": "postgres",
#     "password": "postgres",
#     "host": "localhost",
#     "port": "5432"
# }





def get_db_connection():
    try:
        sts = boto3.client('sts', region_name='ap-south-1')
        identity = sts.get_caller_identity()
        print("✅ AWS Identity Confirmed!")
        print(f"Account ID: {identity['Account']}")
        print(f"User ARN:   {identity['Arn']}")
    except Exception as e:
        print(f"❌ AWS Credentials not working: {e}")
        exit()

    auth_token = boto3.client('rds', region_name='ap-south-1').generate_db_auth_token(DBHostname='bank.cluster-crwa82mywrie.ap-south-1.rds.amazonaws.com', Port=5432, DBUsername='app_user', Region='ap-south-1')
    DB_PARAMS = {
        "host": "bank.cluster-crwa82mywrie.ap-south-1.rds.amazonaws.com",
        "port": "5432",
        "database": "postgres",
        "user": "app_user",
        "password": auth_token,
        "sslmode": "require"
    }
    """Establish and return a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        print(f"⚠ Could not connect to database: {str(e)}")
        raise e

def load_or_create_model():
    """Load model from disk or create a placeholder."""
    global model, scaler, le_geography, le_gender
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded from disk")
    else:
        # Create a dummy model for demonstration
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        # Create dummy training data to fit the model
        X_dummy = pd.DataFrame(np.random.randn(100, 10))
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        print("⚠ Model not found. Created placeholder model. Train with actual data for production use.")
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded from disk")
    else:
        scaler = StandardScaler()
        print("⚠ Scaler not found. Created new instance.")
    
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            le_geography = encoders['geography']
            le_gender = encoders['gender']
        print("✓ Encoders loaded from disk")
    else:
        le_geography = LabelEncoder()
        le_gender = LabelEncoder()
        # Fit with known categories from the dataset
        le_geography.fit(['France', 'Germany', 'Spain'])
        le_gender.fit(['Female', 'Male'])
        print("⚠ Encoders created with default categories")


def preprocess_customer_data(customer_data):
    """Preprocess customer data matching the bank_churn_project.py pipeline."""
    try:
        required_fields = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender'
        ]
        
        missing_fields = [field for field in required_fields if field not in customer_data]
        if missing_fields:
            return None, f"Missing fields: {', '.join(missing_fields)}"
        
        data = {
            'CreditScore': [float(customer_data['CreditScore'])],
            'Geography': [customer_data['Geography']],
            'Gender': [customer_data['Gender']],
            'Age': [float(customer_data['Age'])],
            'Tenure': [float(customer_data['Tenure'])],
            'Balance': [float(customer_data['Balance'])],
            'NumOfProducts': [int(customer_data['NumOfProducts'])],
            'HasCrCard': [int(customer_data['HasCrCard'])],
            'IsActiveMember': [int(customer_data['IsActiveMember'])],
            'EstimatedSalary': [float(customer_data['EstimatedSalary'])]
        }
        
        df = pd.DataFrame(data)
        
        df['Geography'] = le_geography.transform(df['Geography'])
        df['Gender'] = le_gender.transform(df['Gender'])
        
        df_scaled = scaler.transform(df)
        
        return df_scaled, None
    
    except ValueError as e:
        return None, f"Data type error: {str(e)}"
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"


@app.route('/')
def hello_world():
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'endpoints': ['/predict', '/batch_predict', '/data', '/model_info']
    })


@app.route('/predict', methods=['POST'])
def predict_churn():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        customer_data = request.get_json()
        X_processed, error = preprocess_customer_data(customer_data)
        
        if error:
            return jsonify({'error': error}), 400
        
        churn_probability = model.predict_proba(X_processed)[0][1]
        churn_prediction = int(model.predict(X_processed)[0])
        
        return jsonify({
            'prediction': {
                'churn': churn_prediction,
                'churn_probability': round(float(churn_probability), 4),
                'status': 'Will leave' if churn_prediction == 1 else 'Will stay',
                'confidence': round(float(max(1 - churn_probability, churn_probability)) * 100, 2)
            },
            'customer_input': customer_data
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict_churn():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        data = request.get_json()
        customers = data.get('customers', [])
        
        if not customers:
            return jsonify({'error': 'No customers provided'}), 400
        
        predictions = []
        
        for i, customer_data in enumerate(customers):
            X_processed, error = preprocess_customer_data(customer_data)
            
            if error:
                predictions.append({
                    'customer_index': i,
                    'error': error
                })
                continue
            
            churn_probability = model.predict_proba(X_processed)[0][1]
            churn_prediction = int(model.predict(X_processed)[0])
            
            predictions.append({
                'customer_index': i,
                'prediction': {
                    'churn': churn_prediction,
                    'churn_probability': round(float(churn_probability), 4),
                    'status': 'Will leave' if churn_prediction == 1 else 'Will stay',
                    'confidence': round(float(max(1 - churn_probability, churn_probability)) * 100, 2)
                }
            })
        
        return jsonify({
            'total_customers': len(customers),
            'predictions': predictions
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'LightGBM',
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'features': [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography', 'Gender'
        ]
    }), 200


# ============================================================================
# CRUD ENDPOINTS FOR CUSTOMER DATA MANAGEMENT (POSTGRES)
# ============================================================================

@app.route('/data', methods=['GET'])
def get_filtered_customers():
    try:
        limit = request.args.get('limit', default=10, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        allowed_filters = [
            'CustomerId', 'Surname', 'CreditScore', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
            'EstimatedSalary', 'Geography', 'Gender', 'Exited'
        ]
        
        filters = []
        params = []
        
        for field in allowed_filters:
            if field in request.args:
                filters.append(f'"{field}" = %s')
                params.append(request.args.get(field))
        
        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        count_query = f'SELECT COUNT(*) as total FROM customer {where_clause}'
        cur.execute(count_query, params)
        total_count = cur.fetchone()['total']
        
        # Order by CustomerId so pagination is consistent
        data_query = f'SELECT * FROM customer {where_clause} ORDER BY "CustomerId" DESC LIMIT %s OFFSET %s'
        cur.execute(data_query, params + [limit, offset])
        records = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return jsonify({
            'total_filtered_records': total_count,
            'returned_records': len(records),
            'limit': limit,
            'offset': offset,
            'customers': [dict(record) for record in records]
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving customers: {str(e)}'}), 500


@app.route('/data/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute('SELECT * FROM customer WHERE "CustomerId" = %s', (customer_id,))
        record = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not record:
            return jsonify({'error': f'Customer {customer_id} not found'}), 404
            
        return jsonify({
            'customer_id': customer_id,
            'customer': dict(record)
        }), 200
    except Exception as e:
         return jsonify({'error': f'Database error: {str(e)}'}), 500


@app.route('/data', methods=['POST'])
def create_customer():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        customer_data = request.get_json()
        
        required_fields = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender'
        ]
        
        missing_fields = [field for field in required_fields if field not in customer_data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Generate an 8-digit numeric CustomerId
        customer_id = random.randint(10000000, 99999999)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Add Surname if provided, otherwise default to "Unknown"
        surname = customer_data.get('Surname', 'Unknown')
        # Default Exited to 0 (Active) for new customers
        exited = customer_data.get('Exited', 0)
        
        columns = ['CustomerId', 'Surname', 'Exited'] + required_fields
        columns_quoted = [f'"{col}"' for col in columns]
        
        values = [customer_id, surname, exited] + [customer_data[field] for field in required_fields]
        placeholders = ', '.join(['%s'] * len(values))
        
        query = f"INSERT INTO customer ({', '.join(columns_quoted)}) VALUES ({placeholders}) RETURNING *"
        
        cur.execute(query, values)
        new_record = cur.fetchone()
        conn.commit()
        
        cur.close()
        conn.close()
        
        return jsonify({
            'message': 'Customer created successfully',
            'customer_id': customer_id,
            'customer': dict(new_record)
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Error creating customer: {str(e)}'}), 500


@app.route('/data/<customer_id>', methods=['PUT'])
def update_customer(customer_id):
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        update_data = request.get_json()
        
        allowed_fields = [
            'Surname', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender', 'Exited'
        ]
        
        invalid_fields = [field for field in update_data.keys() if field not in allowed_fields]
        if invalid_fields:
            return jsonify({'error': f'Invalid fields: {", ".join(invalid_fields)}'}), 400
            
        if not update_data:
            return jsonify({'error': 'No fields provided to update'}), 400
            
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        set_clauses = []
        params = []
        
        for field, value in update_data.items():
            set_clauses.append(f'"{field}" = %s')
            params.append(value)
            
        params.append(customer_id)
        
        query = f'UPDATE customer SET {", ".join(set_clauses)} WHERE "CustomerId" = %s RETURNING *'
        cur.execute(query, params)
        updated_record = cur.fetchone()
        
        if not updated_record:
            cur.close()
            conn.close()
            return jsonify({'error': f'Customer {customer_id} not found'}), 404
            
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'message': 'Customer updated successfully',
            'customer_id': customer_id,
            'customer': dict(updated_record)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error updating customer: {str(e)}'}), 500


@app.route('/data/<customer_id>', methods=['DELETE'])
def delete_customer(customer_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute('DELETE FROM customer WHERE "CustomerId" = %s RETURNING *', (customer_id,))
        deleted_record = cur.fetchone()
        
        if not deleted_record:
            cur.close()
            conn.close()
            return jsonify({'error': f'Customer {customer_id} not found'}), 404
            
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'message': 'Customer deleted successfully',
            'customer_id': customer_id,
            'deleted_customer': dict(deleted_record)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error deleting customer: {str(e)}'}), 500


@app.route('/data/stats', methods=['GET'])
def get_data_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute('SELECT COUNT(*) as total FROM customer')
        total_customers = cur.fetchone()['total']
        
        if total_customers == 0:
            cur.close()
            conn.close()
            return jsonify({
                'total_customers': 0,
                'message': 'No customers stored yet'
            }), 200
            
        cur.execute('''
            SELECT 
                ROUND(AVG("Age")::numeric, 2) as average_age,
                ROUND(AVG("Tenure")::numeric, 2) as average_tenure,
                ROUND(AVG("Balance")::numeric, 2) as average_balance,
                ROUND(AVG("EstimatedSalary")::numeric, 2) as average_salary,
                MAX("Age") as oldest_customer_age,
                MIN("Age") as youngest_customer_age
            FROM customer
        ''')
        numeric_stats = cur.fetchone()
        
        cur.execute('SELECT "Geography", COUNT(*) as count FROM customer GROUP BY "Geography"')
        geo_dist = {r['Geography']: r['count'] for r in cur.fetchall() if r['Geography']}
        
        cur.execute('SELECT "Gender", COUNT(*) as count FROM customer GROUP BY "Gender"')
        gender_dist = {r['Gender']: r['count'] for r in cur.fetchall() if r['Gender']}
        
        cur.close()
        conn.close()
        
        stats = {
            'total_customers': total_customers,
            'geography_distribution': geo_dist,
            'gender_distribution': gender_dist,
            **dict(numeric_stats)
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': f'Error calculating statistics: {str(e)}'}), 500


if __name__ == '__main__':
    load_or_create_model()
    app.run(debug=True, host='0.0.0.0', port=5000)