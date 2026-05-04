# Program: AutoResearch (LLM vs. AutoML)

## Objective
Evaluate the capabilities of an autonomous LLM research loop against traditional Bayesian optimization (Optuna) across two distinct machine learning domains. The goal is to determine where LLMs provide an edge (architectural reasoning/code generation) versus where pure mathematical optimizers dominate (continuous numerical search)[cite: 2].

## Experiment 1: Numerical Search Space (XGBoost)
*   **Dataset:** California Housing[cite: 2].
*   **Metric:** Validation RMSE[cite: 2].
*   **Hypothesis:** Optuna will significantly outperform the LLM[cite: 2]. LLMs lack the mathematical efficiency to traverse continuous numerical spaces compared to Bayesian statistics[cite: 2].
*   **Search Space:**
    *   `learning_rate`: 0.01 to 0.3[cite: 2]
    *   `max_depth`: 3 to 10[cite: 2]
    *   `subsample`: 0.5 to 1.0[cite: 2]
    *   `colsample_bytree`: 0.5 to 1.0[cite: 2]
    *   `min_child_weight`: 1 to 10[cite: 2]
    *   `gamma`: 0.0 to 0.5[cite: 2]

### XGBoost Experiment Log
| Run | Optimizer | Best RMSE | Iterations | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Optuna | *Pending* | 100 | Baseline mathematical benchmark. |
| 2 | LLM Agent | *Pending* | 100 | Expecting poorer sample efficiency and difficulty pinpointing exact decimal values. |

---

## Experiment 2: Architectural Search Space (Micro-CNN)
*   **Dataset:** CIFAR-10[cite: 2].
*   **Metric:** Validation Accuracy[cite: 2].
*   **Hypothesis:** The LLM will significantly outperform Optuna[cite: 2]. Traditional AutoML cannot generate novel architectures, whereas an LLM can apply software engineering principles to improve network topology[cite: 2].
*   **Baseline Architecture:** Basic `Conv2d` (3->16 channels) $\rightarrow$ ReLU $\rightarrow$ MaxPool2d $\rightarrow$ Linear[cite: 2].
*   **LLM Capabilities:** Full read/write access to `baseline_model.py` to introduce Batch Normalization, Dropout, new activations, etc., using `importlib` for dynamic reloading[cite: 2].

### CNN Experiment Log
| Run | Optimizer | Best Acc | Iterations | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Optuna | *Pending* | 50 | Restricted to tuning learning rate only[cite: 2]. |
| 2 | LLM Agent | *Pending* | 50 | Tracks whether the agent successfully introduces deeper layers without breaking the code. |

---

## Known Issues & Future Directions
*   **The "Cagey Agent" Problem:** The LLM occasionally gets stuck in local minima, making conservative tweaks rather than proposing radical new topologies[cite: 2]. *Idea: Adjust the temperature in `llm_agent.py` to force larger exploratory leaps.*
*   **Security & Sandboxing:** Currently, `llm_agent_cnn.py` executes LLM-generated Python code directly on the host machine[cite: 2]. **Critical Next Step:** Containerize this loop (e.g., inside an isolated Docker container) to prevent the execution of malicious or destructive system-level code[cite: 2].
*   **Visualization:** Need to parse the output CSVs (`optuna_results.csv`, `llm_results.csv`) into Matplotlib/Seaborn line charts to visualize the learning curves over $N$ iterations[cite: 2].