# evaluator.py
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data once to save time
data = fetch_california_housing()
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

def evaluate_xgboost(params):
    """
    Takes a dictionary of hyperparameters, trains XGBoost, 
    and returns the validation RMSE.
    """
    try:
        # We force early stopping to prevent endless training runs
        model = xgb.XGBRegressor(
            **params, 
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse
    except Exception as e:
        # If parameters are completely invalid, return a terrible score
        print(f"Model failed to train with params {params}: {e}")
        return 9999.0