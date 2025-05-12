import torch
import pandas as pd
from scipy.stats import linregress
from transformers import BertTokenizerFast, BertForSequenceClassification

def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
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

def apply_mapping(y_pred, a, b):
    return [a * y + b for y in y_pred]