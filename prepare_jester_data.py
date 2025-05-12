import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
OUT_DIR = "jester"

os.makedirs(OUT_DIR, exist_ok=True)

ratings = pd.read_csv(os.path.join(DATA_DIR, "jester_ratings.csv"), header=None, names=["user_id", "item_id", "score"])
items = pd.read_csv(os.path.join(DATA_DIR, "jester_items.csv"), header=None, names=["item_id", "joke"])

# Merge joke text
ratings = ratings.merge(items, on="item_id")

# Drop duplicates and nulls
ratings = ratings.dropna().drop_duplicates()

# Keep only 100 samples with diverse scores
ratings = ratings.sort_values("score")
fit_set, eval_set = train_test_split(ratings.sample(100, random_state=42), test_size=0.5, random_state=42)

# Save
fit_set[["joke", "score"]].to_csv(os.path.join(OUT_DIR, "jester_fit_50.csv"), index=False)
eval_set[["joke", "score"]].to_csv(os.path.join(OUT_DIR, "jester_eval_50.csv"), index=False)

print("Saved jester_fit_50.csv and jester_eval_50.csv to", OUT_DIR)