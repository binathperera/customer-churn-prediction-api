# Hyperparameter Tuning Documentation

## Overview

The LightGBM model training pipeline includes automated hyperparameter optimization using **Optuna**, a Bayesian optimization framework. This process automatically finds the best hyperparameters for your specific dataset, typically improving model performance by 2-4% compared to default parameters.

## Why Hyperparameter Tuning?

Different datasets have different optimal hyperparameter combinations:

- Default parameters may not be optimal for your data
- Hyperparameter tuning can improve accuracy, precision, recall, and F1 score
- Automated tuning saves time compared to manual grid/random search
- Better generalization and reduced overfitting risk

## Training Workflow

### 1. Baseline Model (Default Parameters)

First, a baseline model is trained with standard LightGBM default parameters to establish a performance baseline:

- n_estimators: 200
- learning_rate: 0.1
- max_depth: 5
- (other hyperparameters at default)

**Purpose**: Provides a benchmark for comparison with the optimized model.

### 2. Hyperparameter Optimization

Optuna performs an automated search over the hyperparameter space:

- **100 trials** (each trial tests a different hyperparameter combination)
- **5-fold cross-validation** on the training set for each trial
- **Bayesian optimization** to intelligently sample the search space
- **F1 score** optimized (balanced metric for imbalanced datasets)

The search space covers:

```
n_estimators: 100-500
learning_rate: 0.01-0.3
max_depth: 3-10
num_leaves: 20-100
min_data_in_leaf: 5-50
lambda_l1: 0.0-10.0          # L1 regularization
lambda_l2: 0.0-10.0          # L2 regularization
feature_fraction: 0.4-1.0    # Feature subsampling
bagging_fraction: 0.4-1.0    # Data subsampling
bagging_freq: 1-10
min_sum_hessian_in_leaf: 1e-3-10.0
```

### 3. Final Model Training

The model is retrained on the full training set using the best hyperparameters found:

- Uses best hyperparameters from Optuna
- Evaluated on the held-out test set
- Compared with baseline for improvement metrics

## Key Hyperparameters Explained

### Tree Growth

- **n_estimators**: Number of boosting stages
  - Higher = potentially better but slower training
  - Typical range: 100-500
  - Too high can cause overfitting

- **max_depth**: Maximum tree depth
  - Controls tree complexity
  - Lower = simpler model, less overfitting
  - Typical range: 3-10

- **num_leaves**: Maximum leaves per tree (LightGBM specific)
  - More flexible tree structure than traditional depth
  - Typical range: 20-100
  - Effects similar to max_depth

### Learning

- **learning_rate**: Step size for boosting
  - Lower = slower but more careful learning
  - Typical range: 0.01-0.3
  - Often paired with higher n_estimators

### Regularization (Overfitting Prevention)

- **lambda_l1**: L1 (Lasso) regularization strength
  - Encourages sparse features
  - Typical range: 0.0-10.0

- **lambda_l2**: L2 (Ridge) regularization strength
  - Encourages smaller weights
  - Typical range: 0.0-10.0

- **min_data_in_leaf**: Minimum samples per leaf
  - Prevents creating tiny leaves
  - Typical range: 5-50

- **min_sum_hessian_in_leaf**: Minimum Hessian sum per leaf
  - Prevents splits with too few instances
  - Typical range: 1e-3-10.0

### Sampling

- **feature_fraction**: Proportion of features to use per tree
  - 0.5 = use 50% of features randomly
  - Typical range: 0.4-1.0
  - Reduces variance and prevents overfitting

- **bagging_fraction**: Proportion of training data to use per tree
  - 0.7 = use 70% of training samples randomly
  - Typical range: 0.4-1.0
  - Reduces variance

- **bagging_freq**: Frequency of bagging
  - How often to apply data sampling
  - Typical range: 1-10
  - Affects training time

## Optimization Process

### Optuna's Tree-structured Parzen Estimator (TPE)

Optuna uses TPE, a Bayesian optimization algorithm that:

1. **Intelligently samples** the hyperparameter space
2. **Learns** which hyperparameter combinations work best
3. **Focuses** on promising regions of the search space
4. **Avoids** spending time on clearly bad combinations

**Advantages over Grid/Random Search:**

- Grid search: Slow, tests all combinations (curse of dimensionality)
- Random search: Inefficient, wastes time on bad combinations
- Bayesian (TPE): Smart sampling, focuses on good regions

## Expected Improvements

Typical improvements from hyperparameter tuning:

- **Accuracy**: +1-3%
- **F1 Score**: +2-4%
- **ROC-AUC**: +1-3%

Example:

```
Baseline Model:
  Accuracy  : 84.50 %
  F1 Score  : 75.20 %
  ROC-AUC   : 86.30 %

Optimized Model:
  Accuracy  : 86.75 %    (+2.25%)
  F1 Score  : 78.56 %    (+3.36%)
  ROC-AUC   : 88.45 %    (+2.15%)
```

## Training Time

- **Data Preprocessing**: ~1-2 minutes
- **Baseline Model**: ~30 seconds
- **Hyperparameter Tuning (100 trials)**: ~5-10 minutes
- **Final Model Training**: ~1 minute
- **Total**: ~10-15 minutes for typical dataset (10,000 samples)

Factors affecting tuning time:

- Dataset size (larger = slower)
- Number of trials (100 is default)
- Hardware (GPU/CPU)
- Number of cross-validation folds (5 is default)

## Customizing Tuning

To change the number of trials, edit `train_model.py`:

```python
# In the main() function, change this line:
best_params, best_cv_score = optimize_hyperparameters(
    X_train, y_train, n_trials=100  # Change 100 to desired number
)
```

**Examples:**

- `n_trials=50`: ~3-5 minutes (faster, less thorough)
- `n_trials=100`: ~5-10 minutes (balanced)
- `n_trials=200`: ~10-20 minutes (thorough, might find better params)

## Reproducibility

The tuning process is reproducible because:

- Random state is fixed: `random_state=42`
- TPE sampler uses seed: `optuna.samplers.TPESampler(seed=42)`
- Stratified K-Fold uses seed: `StratifiedKFold(..., random_state=42)`

Running the same training twice will produce identical results.

## Monitoring Optimization

During optimization, you'll see:

```
[I 2024-01-15 10:30:45,123] A new study created in memory with name: no-name-xxx.
[I 2024-01-15 10:30:50,456] Trial 1 finished with value: 0.7543 and parameters: {...}
[I 2024-01-15 10:30:55,789] Trial 2 finished with value: 0.7621 and parameters: {...}
...
```

Each line shows:

- Trial number
- F1 score achieved
- Hyperparameters tested

The F1 score should generally improve over time as Optuna learns.

## Best Practices

1. **Use appropriate metrics**: F1 score is good for imbalanced data (like churn)
2. **Sufficient trials**: At least 50-100 trials recommended
3. **Cross-validation**: 5-fold is standard, increase to 10 for smaller datasets
4. **Monitor overfitting**: Check test set performance (final model section)
5. **Reproducible**: Always use fixed random states

## Troubleshooting

### "Optimization took too long"

- Reduce `n_trials` (e.g., 50 instead of 100)
- Reduce cross-validation folds (e.g., 3 instead of 5)
- Use GPU if available (set `n_jobs=-1`)

### "Model performance didn't improve"

- Try more trials (e.g., 200)
- Check if dataset is too small (need ~1000+ samples)
- Verify baseline model isn't already optimal

### "Results not reproducible"

- Ensure `random_state=42` in all sklearn components
- Check Optuna sampler seed
- Verify no external randomness in data loading

## Advanced Customization

For advanced users, you can modify the hyperparameter search space in `train_model.py`:

```python
def objective(trial):
    params = {
        # Expand search ranges for more thorough tuning
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Wider range
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        # ... other parameters
    }
```

Or use different optimization strategies:

```python
sampler = optuna.samplers.TPESampler(seed=42)  # Current
# Try alternatives:
# sampler = optuna.samplers.RandomSampler(seed=42)
# sampler = optuna.samplers.GridSampler(...)
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [LightGBM Parameter Guide](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- [Hyperparameter Optimization in Practice](https://towardsdatascience.com/hyperparameter-optimization-explained-d5b3b2ba9b5e)
