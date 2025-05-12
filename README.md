# Humor Reward Model Training and Evaluation

This project builds a humor reward model using Reddit Jokes as weak supervision and evaluates it using the Jester dataset.

## Directory Structure

```
Humor_LLM_Tuning/
├── reward_model/
│   ├── train_regression.py
│   ├── map_jester_fit.py
│   ├── evaluate_regression.py
│   ├── utils.py
│   └── __init__.py
├── prepare_jester_data.py
├── data/
│   ├── jester_ratings.csv
│   └── jester_items.csv
├── jester/
│   ├── jester_fit_50.csv
│   ├── jester_eval_50.csv
│   └── linear_map.json
├── models/
│   └── reward_model_regression/
└── requirements.txt
```
## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the reward model on Reddit jokes:

```bash
python -m reward_model.train_regression --epochs 3 ## Can be changed to a larger value
```
The model will be saved to models/reward_model_regression/.

## Prepare Jester Data

Split 100 Jester jokes into 50 for fitting and 50 for evaluation:

```bash
python prepare_jester_data.py
```

## Fit Linear Mapping

Use predictions on 50 jokes to fit a linear mapping from model output to Jester score:

```bash
python -m reward_model.map_jester_fit
```

Saves jester/linear_map.json.

## Evaluate

Evaluate mapped predictions on the remaining 50 jokes:

```bash
python -m reward_model.evaluate_regression
```

Outputs RMSE and MAE.

## Notes

* Reddit scores are log-transformed during training.

* Final predictions on Jester are mapped using a fitted linear function.

## Example Output

```bash
Evaluation Results:
RMSE: 5.6471
MAE : 4.6932
```
