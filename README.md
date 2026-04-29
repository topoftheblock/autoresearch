# AutoResearch: LLM Intuition vs. Traditional AutoML

A comparative study evaluating autonomous LLM research loops against Bayesian optimization, inspired by Andrej Karpathy's 2026 *AutoResearch* paradigm.

---

## Overview

This repository benchmarks an LLM agent acting as an AI researcher — rewriting code and navigating complex search spaces — against traditional mathematical optimizers (Optuna). Experiments are split across two domains to isolate where LLMs excel (architectural reasoning) versus where math wins (pure numerical optimization).

---

## What is AutoResearch?

AutoResearch is an agentic framework where a Large Language Model runs in a continuous, closed-loop execution cycle to autonomously improve a machine learning model. Instead of human researchers manually tweaking architectures or hyperparameters, the agent handles the scientific method natively.

The core **Karpathy Loop** operates on four primitives:

1. **Observe** — The agent reads the current training script (`train.py`) and logs of past execution metrics.
2. **Hypothesize & Act** — The agent writes new code proposing an architectural change or hyperparameter set, and commits it.
3. **Execute & Evaluate** — The system time-boxes a training run and returns a core metric (e.g., Validation Loss or Accuracy).
4. **Rollback or Advance** — Improvements are kept; regressions are reverted.

This shifts LLMs from static code generation to dynamic agentic engineering, introducing new capabilities (creative architectural leaps) alongside new failure modes (sample inefficiency and local search traps).

---

## Experiment Design

### Experiment 1: Numerical Search Space (Tabular / XGBoost)

**Objective:** Optimize an XGBoost Regressor on the California Housing dataset.

| Track | Script | Approach |
|---|---|---|
| Math | `optuna_search.py` | Bayesian optimization over a large grid of interacting hyperparameters (`learning_rate`, `gamma`, `min_child_weight`, etc.) |
| Agent | `llm_agent.py` | LLM guesses optimal hyperparameter combinations from intuition, reading historical validation logs, and outputting JSON configs |

**Hypothesis:** Optuna will significantly outperform the LLM. LLMs lack the mathematical efficiency to traverse continuous numerical spaces compared to Bayesian statistics.

---

### Experiment 2: Architectural Search Space (Vision / CNN)

**Objective:** Optimize a sub-optimal Micro-CNN on CIFAR-10.

| Track | Script | Approach |
|---|---|---|
| Math | `optuna_cnn.py` | Optuna is locked to learning rate tuning only — no architectural access |
| Agent | `llm_agent_cnn.py` | LLM has full read/write access to `baseline_model.py`, free to introduce Batch Normalization, Dropout, new activations, etc., using `importlib` for dynamic reloading |

**Hypothesis:** The LLM will significantly outperform Optuna. Traditional AutoML cannot generate novel architectures, whereas an LLM can apply software engineering principles to improve network topology.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── XGBoost_Experiment/
│   ├── evaluator.py          # Shared dataset and RMSE scoring
│   ├── optuna_search.py      # Math track: Bayesian optimization
│   └── llm_agent.py          # Agent track: LLM JSON-based researcher
└── CNN_Experiment/
    ├── evaluator_cnn.py      # Shared CIFAR-10 DataLoader and training loop
    ├── baseline_model.py     # Sub-optimal PyTorch CNN (dynamically edited)
    ├── optuna_cnn.py         # Math track: Learning rate optimization
    └── llm_agent_cnn.py      # Agent track: LLM code generation + dynamic execution
```

---

## Setup

### 1. Install Dependencies

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Run the Experiments

Run the Optuna baseline first, then the LLM agent. Both tracks output a `.csv` of validation curves for comparative analysis.

```bash
# Experiment 1 — XGBoost
cd XGBoost_Experiment
python optuna_search.py
python llm_agent.py

# Experiment 2 — CNN
cd CNN_Experiment
python optuna_cnn.py
python llm_agent_cnn.py
```

---

## Results

> **TODO:** Add charts and findings after running the scripts. Use Matplotlib or seaborn to generate line charts comparing RMSE/Accuracy of Optuna vs. the LLM over N iterations.

### Preliminary Observations

- **Sample Efficiency:** Optuna converges faster on purely numerical tasks.
- **Architectural Creativity:** The LLM successfully introduced deep learning best practices (e.g., LayerNorm) without explicit human instruction.
- **The "Cagey Agent" Problem:** The LLM occasionally got stuck in local minima, making conservative tweaks rather than proposing radical new topologies.

---

## Limitations

**Compute Constraints:** Experiments are bounded to 50–100 iterations using fast-training models to accommodate local hardware. Deeper architectural search on larger datasets would require distributed GPU clusters.

**Security & Sandboxing:** `llm_agent_cnn.py` executes LLM-generated Python code directly on the host machine via dynamic module reloading. In any production MLOps environment, this loop **must be containerized** (e.g., inside an isolated Docker container) to prevent execution of malicious or destructive system-level code.

---

## License

MIT