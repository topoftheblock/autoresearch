# llm_agent.py
import json
import os
import pandas as pd
from openai import OpenAI
from evaluator import evaluate_xgboost

# Initialize your LLM client (Make sure OPENAI_API_KEY is in your environment variables)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Store the history so the LLM can learn from its mistakes
experiment_history = []

def get_llm_proposal(history_text):
    system_prompt = """
    You are an expert Machine Learning Engineer optimizing an XGBRegressor for the California Housing dataset.
    Your goal is to minimize the validation RMSE.
    
    Here is the history of previous hyperparameter attempts and their resulting RMSE scores:
    {history}
    
    Analyze the trends. What parameters are working? What is failing?
    Based on your intuition, propose the next set of hyperparameters to test.
    
    You must output ONLY a valid JSON object with the following keys. Do not include markdown formatting or explanations:
    - learning_rate (float between 0.01 and 0.3)
    - max_depth (int between 3 and 10)
    - subsample (float between 0.5 and 1.0)
    - colsample_bytree (float between 0.5 and 1.0)
    - min_child_weight (int between 1 and 10)
    - gamma (float between 0.0 and 0.5)
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # You can use a smaller/cheaper model for this loop
        messages=[
            {"role": "system", "content": system_prompt.replace("{history}", history_text)}
        ],
        temperature=0.7, # Allow some creativity
        response_format={"type": "json_object"} # Force JSON output
    )
    
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    print("Starting LLM Intuition Search...")
    results_log = []
    
    # Run exactly 100 iterations
    for i in range(100):
        # Format history for the prompt (keep only the last 10 to avoid context limits)
        recent_history = "\n".join([f"Attempt {h['iteration']}: {h['params']} -> RMSE: {h['rmse']:.4f}" 
                                   for h in experiment_history[-10:]])
        if not recent_history:
            recent_history = "No history yet. This is the first run."
            
        try:
            # 1. Get proposal from LLM
            proposed_params = get_llm_proposal(recent_history)
            print(f"Iteration {i+1} - LLM Proposed: {proposed_params}")
            
            # 2. Evaluate the parameters
            rmse = evaluate_xgboost(proposed_params)
            print(f"Iteration {i+1} - Resulting RMSE: {rmse:.4f}\n")
            
            # 3. Log results
            experiment_data = {"iteration": i+1, "params": proposed_params, "rmse": rmse}
            experiment_history.append(experiment_data)
            results_log.append({**proposed_params, "rmse": rmse, "iteration": i+1})
            
        except json.JSONDecodeError:
            print("Error: LLM failed to output valid JSON. Skipping iteration.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save results for comparison
    df_llm = pd.DataFrame(results_log)
    df_llm.to_csv("llm_results.csv", index=False)
    
    best_run = df_llm.loc[df_llm['rmse'].idxmin()]
    print(f"\nBest LLM RMSE: {best_run['rmse']}")