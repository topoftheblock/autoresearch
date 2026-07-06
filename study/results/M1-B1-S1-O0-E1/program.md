# Research Task: Tree ensembles on the breast cancer dataset

You are an autonomous research agent. Your task is to investigate the following
question through real, executed experiments (not simulated ones):

> How does the choice of hyperparameters affect the generalization accuracy of tree-ensemble classifiers (Random Forest and Gradient Boosting) on the breast cancer diagnostic dataset? Find a well-performing configuration.

You have access to a fixed dataset (sklearn.datasets.load_breast_cancer, 70/30 train/validation split, already loaded and split
into train/validation) and to a tool that fits either a RandomForestClassifier or
a GradientBoostingClassifier with hyperparameters you choose, and returns the
requested metric(s) computed on the validation split.

## Evaluation

Evaluate every model using 5-fold cross-validated accuracy on the training split, AND report the standard deviation across folds as a measure of stability. A configuration is only better than another if its mean accuracy is higher; use the standard deviation only to break ties or to flag instability.

## Search strategy

You are free to compare RandomForestClassifier and GradientBoostingClassifier side by side within the same session, and to vary multiple hyperparameters of each at once.

Prioritize depth: as soon as one configuration looks promising, immediately refine it with nearby variations (e.g. slightly different hyperparameter values) before trying anything unrelated.

## Budget and stopping

You have a budget of up to 8 experiments. After each experiment, decide whether to continue: stop as soon as an experiment fails to improve the evaluation metric (defined above) by more than 0.002 over the best result so far, or when you reach 8 experiments, whichever comes first.

## Reporting style

Keep every response short: state the action you are taking and the numbers involved, with minimal prose.

## Required response format

At every step, respond with a single JSON object and nothing else, matching one
of these two shapes:

Proposing an experiment:
```json
{"action": "propose", "hypothesis": "<why you think this configuration will help>",
 "model": "random_forest" | "gradient_boosting",
 "params": {"<hyperparameter>": <value>, ...}}
```

Interpreting the last result and deciding what happens next:
```json
{"action": "interpret", "interpretation": "<what the result tells you>",
 "decision": "continue" | "stop",
 "final_recommendation": "<only required when decision == stop: your final
   recommended model and hyperparameters, and why>"}
```