# llm_agent_cnn.py
import os
import re
import pandas as pd
from openai import OpenAI
from evaluator_cnn import evaluate_model

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
experiment_history = []

def get_llm_code_proposal(history_text, current_code):
    system_prompt = f"""
    You are an expert AI Researcher building a PyTorch model for CIFAR-10.
    Your goal is to rewrite the MicroCNN class to maximize validation accuracy.
    You can add Convolutional layers, BatchNorm, Dropout, modify channels, or change activation functions.
    Keep the model relatively small (under 1 million parameters) so it trains fast.

    Here is the history of previous attempts and their validation accuracy:
    {history_text}

    Here is the current code:
    ```python
    {current_code}
    ```

    Output the new, improved Python code. You MUST include imports (like torch.nn as nn) and the MicroCNN class.
    Also, add a comment at the very top specifying the learning rate you want to use, like this: `# LR: 0.001`
    Output ONLY valid Python code inside a markdown code block. No explanations.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # You need a strong coding model for this (GPT-4o or Claude 3.5 Sonnet)
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def extract_code_and_lr(llm_response):
    # Use regex to pull the python code block out of the LLM's text
    code_match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
    code = code_match.group(1) if code_match else llm_response

    # Look for the learning rate comment
    lr_match = re.search(r'# LR:\s*([0-9.]+)', code)
    lr = float(lr_match.group(1)) if lr_match else 0.001
    
    return code, lr

if __name__ == "__main__":
    print("Starting LLM Architect Search...")
    
    # Read the baseline code to start
    with open("baseline_model.py", "r") as f:
        current_code = f.read()

    results_log = []
    
    for i in range(50):
        recent_history = "\n".join([f"Attempt {h['iteration']}: Acc {h['accuracy']:.4f}, LR {h['lr']}" 
                                   for h in experiment_history[-5:]])
        if not recent_history:
            recent_history = "No history yet. This is the baseline."

        print(f"\n--- Iteration {i+1} ---")
        try:
            # 1. Ask LLM for new code
            llm_response = get_llm_code_proposal(recent_history, current_code)
            new_code, lr = extract_code_and_lr(llm_response)
            
            # 2. Save the LLM's code to a temporary file
            with open("generated_model.py", "w") as f:
                f.write(new_code)
            
            # 3. Dynamically import the new model class
            # (We use importlib to reload the module each loop)
            import importlib
            import generated_model
            importlib.reload(generated_model)
            
            # 4. Evaluate
            accuracy = evaluate_model(generated_model.MicroCNN, lr=lr)
            print(f"LLM Proposed LR: {lr} | Accuracy Achieved: {accuracy:.4f}")
            
            # 5. Log and Update
            experiment_history.append({"iteration": i+1, "accuracy": accuracy, "lr": lr})
            results_log.append({"iteration": i+1, "accuracy": accuracy, "lr": lr})
            
            # If the model improved, make it the new current_code for the next iteration
            if accuracy >= max([h['accuracy'] for h in experiment_history] + [0]):
                current_code = new_code

        except Exception as e:
            print(f"LLM failed to generate valid code or model crashed: {e}")
            experiment_history.append({"iteration": i+1, "accuracy": 0.0, "lr": 0.0})

    # Save results
    df_llm = pd.DataFrame(results_log)
    df_llm.to_csv("llm_cnn_results.csv", index=False)