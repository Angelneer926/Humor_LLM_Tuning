import pandas as pd
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from scipy.stats import linregress

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

def fit_linear_mapping(y_pred, y_true):
    slope, intercept, *_ = linregress(y_pred, y_true)
    return slope, intercept

def main():
    MODEL_PATH = "models/reward_model_regression"
    FIT_CSV = "jester/jester_fit_50.csv"
    OUT_PATH = "jester/linear_map.json"

    print("Loading fit dataset...")
    df = pd.read_csv(FIT_CSV)
    texts = df["joke"].tolist()
    true_scores = df["score"].tolist()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    print("Predicting model outputs...")
    preds = predict_scores(model, tokenizer, texts)

    print("Fitting linear map: jester_score = a * model_output + b")
    a, b = fit_linear_mapping(preds, true_scores)
    print(f"Linear fit result: jester_score ≈ {a:.4f} * model_output + {b:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump({"a": a, "b": b}, f)
    print("Mapping saved to", OUT_PATH)

if __name__ == "__main__":
    main()