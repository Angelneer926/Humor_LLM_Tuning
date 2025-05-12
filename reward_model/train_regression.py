import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from reward_model.weighted_trainer import WeightedMSETrainer


def preprocess(example):
    title = example["title"] or ""
    body = example["selftext"] or ""
    score = example["score"] or 0
    score = min(score, 500)
    label = np.log1p(score)
    text = (title + " " + body).strip()
    return {"text": text, "label": label}


def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=str, default="models/reward_model_regression")
    args = parser.parse_args()

    print("Loading Reddit Jokes dataset...")
    ds = load_dataset("SocialGrep/one-million-reddit-jokes")["train"]

    print("Preprocessing and transforming labels...")
    ds = ds.map(preprocess)
    df = pd.DataFrame(ds).dropna(subset=["text", "label"])

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = Dataset.from_pandas(train_df).map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(lambda x: tokenize_batch(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = WeightedMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()