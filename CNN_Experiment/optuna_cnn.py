# optuna_cnn.py
import optuna
import pandas as pd
from evaluator_cnn import evaluate_model
from baseline_model import MicroCNN

def objective(trial):
    # Optuna can only tweak numerical parameters
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Evaluate the fixed baseline model
    accuracy = evaluate_model(MicroCNN, lr=lr)
    return accuracy

if __name__ == "__main__":
    print("Starting Optuna Math Search...")
    # Maximize accuracy
    study = optuna.create_study(direction="maximize")
    
    # Run 50 iterations (CNNs take longer than XGBoost, so we reduce the trial count)
    study.optimize(objective, n_trials=50)
    
    print(f"\nBest Optuna Accuracy: {study.best_value:.4f}")
    print(f"Best Optuna Params: {study.best_params}")
    
    df = study.trials_dataframe()
    df.to_csv("optuna_cnn_results.csv", index=False)