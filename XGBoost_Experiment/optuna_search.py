# optuna_search.py
import optuna
import pandas as pd
from evaluator import evaluate_xgboost

def objective(trial):
    # 1. Define the mathematical search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5)
    }
    
    # 2. Evaluate
    return evaluate_xgboost(params)

if __name__ == "__main__":
    print("Starting Optuna Math Search...")
    study = optuna.create_study(direction="minimize")
    
    # Run exactly 100 iterations (trials)
    study.optimize(objective, n_trials=100)
    
    print(f"\nBest Optuna RMSE: {study.best_value}")
    print(f"Best Optuna Params: {study.best_params}")
    
    # Save results for your portfolio visualization
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)