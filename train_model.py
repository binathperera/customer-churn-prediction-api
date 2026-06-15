"""
Model Training Script - LightGBM Churn Prediction

This script trains the LightGBM model following the exact pipeline from bank_churn_project.py
and saves the model, scaler, and encoders for use in the Flask API.

Features:
  - Complete preprocessing pipeline (scaling, encoding, outlier removal)
  - Hyperparameter tuning using Optuna for optimal model performance
  - Cross-validation for robust evaluation
  - Model persistence (pkl files)

Usage:
    python train_model.py <path_to_churn_modelling.csv>

Example:
    python train_model.py data/Churn_Modelling.csv
"""

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import optuna
from optuna.trial import TrialState
import warnings

warnings.filterwarnings('ignore')


def load_dataset(file_path):
    """Load the Churn_Modelling.csv dataset."""
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def step1_drop_irrelevant_columns(df):
    """Step 1: Drop irrelevant columns (RowNumber, CustomerId, Surname)."""
    print("\n" + "="*60)
    print("STEP 1: Dropping Irrelevant Columns")
    print("="*60)
    
    df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    print(f"Dropped: RowNumber, CustomerId, Surname")
    print(f"Shape after: {df_clean.shape}")
    return df_clean


def step2_remove_outliers(df):
    """Step 2: Remove outliers using IQR method."""
    print("\n" + "="*60)
    print("STEP 2: Removing Outliers Using IQR Method")
    print("="*60)
    
    numerical_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    df_no_outliers = df.copy()
    
    print(f"Shape before: {df_no_outliers.shape}")
    
    for col in numerical_cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before = len(df_no_outliers)
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) &
            (df_no_outliers[col] <= upper_bound)
        ]
        after = len(df_no_outliers)
        removed = before - after
        
        print(f"{col}: {removed} rows removed")
    
    print(f"Shape after: {df_no_outliers.shape}")
    return df_no_outliers


def step3_encode_categories(df):
    """Step 3: Encode categorical variables (Geography, Gender)."""
    print("\n" + "="*60)
    print("STEP 3: Encoding Categorical Variables")
    print("="*60)
    
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded["Geography"] = le_geography.fit_transform(df_encoded["Geography"])
    df_encoded["Gender"] = le_gender.fit_transform(df_encoded["Gender"])
    
    print("\nGeography Encoding:")
    for label, encoded in zip(le_geography.classes_,
                              le_geography.transform(le_geography.classes_)):
        print(f"  {label} → {encoded}")
    
    print("\nGender Encoding:")
    for label, encoded in zip(le_gender.classes_,
                              le_gender.transform(le_gender.classes_)):
        print(f"  {label} → {encoded}")
    
    return df_encoded, le_geography, le_gender


def step4_scale_features(X):
    """Step 4 & 5: Scale features using StandardScaler."""
    print("\n" + "="*60)
    print("STEP 4 & 5: Feature Scaling")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Scaled feature statistics:")
    print(f"  Mean: {X_scaled.mean().round(4).to_dict()}")
    print(f"  Std:  {X_scaled.std().round(4).to_dict()}")
    
    return X_scaled, scaler


def step6_handle_imbalance(X, y):
    """Step 6: Handle class imbalance using SMOTE."""
    print("\n" + "="*60)
    print("STEP 6: Handling Class Imbalance Using SMOTE")
    print("="*60)
    
    print(f"\nClass distribution before split:")
    print(f"  Stayed (0): {sum(y == 0)} customers")
    print(f"  Left (1):   {sum(y == 1)} customers")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain-test split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nClass distribution after SMOTE (training set only):")
    print(f"  Stayed (0): {sum(y_train_balanced == 0)} samples")
    print(f"  Left (1):   {sum(y_train_balanced == 1)} samples")
    print(f"  Training set size: {X_train.shape[0]} → {X_train_balanced.shape[0]} samples")
    
    return X_train_balanced, X_test, y_train_balanced, y_test


def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model with baseline parameters."""
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Baseline model trained successfully")
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
    
    print("\n" + "-"*60)
    print("BASELINE MODEL EVALUATION RESULTS")
    print("-"*60)
    print(f"  Accuracy  : {accuracy} %")
    print(f"  F1 Score  : {f1} %")
    print(f"  ROC-AUC   : {auc} %")
    print("-"*60)
    
    return accuracy, f1, auc


def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Parameters:
        X_train: Training features (after SMOTE)
        y_train: Training labels (after SMOTE)
        n_trials: Number of optimization trials (default: 100)
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_score: Best F1 score achieved
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: LightGBM with Optuna")
    print("="*60)
    print(f"Optimizing over {n_trials} trials...")
    print("(This may take a few minutes)\n")
    
    # Define the objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10.0, log=True),
        }
        
        # Create model with suggested parameters
        model = LGBMClassifier(
            **params,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # Use stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        
        return scores.mean()
    
    # Create a study object and optimize
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=-1
    )
    
    # Get best parameters and score
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\n✓ Best F1 Score from Tuning: {best_score:.4f}")
    print(f"\nBest Hyperparameters Found:")
    for param, value in sorted(best_params.items()):
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    
    # Display trial statistics
    print(f"\n" + "-"*60)
    print("Optimization Statistics:")
    print("-"*60)
    print(f"  Number of trials: {len(study.trials)}")
    print(f"  Best trial number: {best_trial.number}")
    
    # Count trials by state
    completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == TrialState.FAIL])
    
    print(f"  Completed: {completed}, Pruned: {pruned}, Failed: {failed}")
    
    return best_params, best_score


def train_final_model(X_train, X_test, y_train, y_test, best_params):
    """
    Train the final LightGBM model with best hyperparameters.
    
    Parameters:
        X_train: Training features (after SMOTE)
        X_test: Test features
        y_train: Training labels (after SMOTE)
        y_test: Test labels
        best_params: Best hyperparameters from optimization
    
    Returns:
        model: Trained LightGBM model
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING")
    print("With Best Hyperparameters from Optuna")
    print("="*60)
    
    # Create model with best parameters
    model = LGBMClassifier(
        **best_params,
        random_state=42,
        verbose=-1
    )
    
    # Train on full training set
    model.fit(X_train, y_train)
    print("✓ Final model trained successfully")
    
    # Evaluation on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
    
    print("\n" + "-"*60)
    print("FINAL MODEL EVALUATION RESULTS")
    print("-"*60)
    print(f"  Accuracy  : {accuracy} %")
    print(f"  F1 Score  : {f1} %")
    print(f"  ROC-AUC   : {auc} %")
    print("-"*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TP (caught churners):    {tp}")
    print(f"  TN (kept non-churners):  {tn}")
    print(f"  FP (wrongly flagged):    {fp}")
    print(f"  FN (missed churners):    {fn}")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': auc,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return model, metrics


def save_artifacts(model, scaler, le_geography, le_gender):
    """Save model and preprocessing artifacts."""
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)
    
    # Save model
    with open("lgbm_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved: lgbm_model.pkl")
    
    # Save scaler
    with open("scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved: scaler.pkl")
    
    # Save encoders
    encoders = {
        'geography': le_geography,
        'gender': le_gender
    }
    with open("encoders.pkl", 'wb') as f:
        pickle.dump(encoders, f)
    print("✓ Encoders saved: encoders.pkl")
    
    print("\n✓ All artifacts saved successfully!")
    print("You can now use the Flask API with the trained model.")


def main(csv_file):
    """Main training pipeline with hyperparameter tuning."""
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("With Hyperparameter Optimization")
    print("="*60)
    
    # Load data
    df = load_dataset(csv_file)
    
    # Step 1: Drop irrelevant columns
    df_clean = step1_drop_irrelevant_columns(df)
    
    # Step 2: Remove outliers
    df_no_outliers = step2_remove_outliers(df_clean)
    
    # Step 3: Encode categories
    df_encoded, le_geography, le_gender = step3_encode_categories(df_no_outliers)
    
    # Separate features and target
    X = df_encoded.drop(columns=["Exited"])
    y = df_encoded["Exited"]
    
    # Step 4 & 5: Scale features
    X_scaled, scaler = step4_scale_features(X)
    
    # Step 6: Handle imbalance
    X_train, X_test, y_train, y_test = step6_handle_imbalance(X_scaled, y)
    
    # Train baseline model (for comparison)
    baseline_accuracy, baseline_f1, baseline_auc = train_lightgbm(
        X_train, X_test, y_train, y_test
    )
    
    # Run hyperparameter optimization
    best_params, best_cv_score = optimize_hyperparameters(
        X_train, y_train, n_trials=100
    )
    
    # Train final model with best parameters
    model, metrics = train_final_model(
        X_train, X_test, y_train, y_test, best_params
    )
    
    # Show improvement
    print("\n" + "="*60)
    print("MODEL IMPROVEMENT SUMMARY")
    print("="*60)
    print("\nBaseline Model (Default Parameters):")
    print(f"  Accuracy : {baseline_accuracy} %")
    print(f"  F1 Score : {baseline_f1} %")
    print(f"  ROC-AUC  : {baseline_auc} %")
    
    print("\nOptimized Model (Best Hyperparameters):")
    print(f"  Accuracy : {metrics['accuracy']} %")
    print(f"  F1 Score : {metrics['f1_score']} %")
    print(f"  ROC-AUC  : {metrics['roc_auc']} %")
    
    print("\nImprovement:")
    print(f"  Accuracy : {metrics['accuracy'] - baseline_accuracy:+.2f} %")
    print(f"  F1 Score : {metrics['f1_score'] - baseline_f1:+.2f} %")
    print(f"  ROC-AUC  : {metrics['roc_auc'] - baseline_auc:+.2f} %")
    print("="*60)
    
    # Save artifacts
    save_artifacts(model, scaler, le_geography, le_gender)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <path_to_churn_modelling.csv>")
        print("\nExample:")
        print("  python train_model.py data/Churn_Modelling.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not csv_file.endswith('.csv'):
        print("Error: File must be a CSV file")
        sys.exit(1)
    
    try:
        main(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
