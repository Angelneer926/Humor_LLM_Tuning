import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

def load_model_and_tokenizer(model_path):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict_scores(model, tokenizer, texts, batch_size=16, max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            preds.extend(outputs.logits.view(-1).cpu().tolist())
    return preds

def apply_mapping(y_pred, a, b):
    return [a * y + b for y in y_pred]

def main():
    MODEL_PATH = "models/reward_model_regression"
    EVAL_CSV = "jester/jester_eval_50.csv"
    MAP_JSON = "jester/linear_map.json"

    df = pd.read_csv(EVAL_CSV)
    texts = df["joke"].tolist()
    true_scores = df["score"].tolist()

    with open(MAP_JSON, "r") as f:
        mapping = json.load(f)
    a, b = mapping["a"], mapping["b"]

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    raw_preds = predict_scores(model, tokenizer, texts)
    mapped_preds = apply_mapping(raw_preds, a, b)

    rmse = mean_squared_error(true_scores, mapped_preds, squared=False)
    mae = mean_absolute_error(true_scores, mapped_preds)

    print(f"Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

if __name__ == "__main__":
    main()