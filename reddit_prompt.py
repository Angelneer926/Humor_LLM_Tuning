from datasets import load_dataset, Dataset
import os
import html
import re

FILTERED_PATH = "filtered_jokes_titles_text.parquet"

def clean_text(text):
    if text is None:
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Replace common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#x200B;', '')  # Remove zero-width space
    # Replace newlines and extra spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_save():
    ds = load_dataset("SocialGrep/one-million-reddit-jokes")["train"]
    def filter_jokes(example):
        # Clean the text, handle None
        example["selftext"] = clean_text(example.get("selftext"))
        example["title"] = clean_text(example.get("title"))
        
        # Print problematic entries for debugging
        if example["selftext"] in ["[deleted]", "[removed]", ""]:
            print(f"Found problematic entry: Title: {example['title']}, Score: {example['score']}")
        
        return (
            example["selftext"] not in ["[removed]", "[deleted]", ""] and
            example["title"].strip().endswith("?") and
            example["score"] >= 5 and
            len(example["selftext"]) < 500 and
            len(example["selftext"]) > 1  # Ensure there's actual content
        )
    filtered_ds = ds.filter(filter_jokes)
    # Only keep 'title' and 'selftext'
    title_joke_ds = filtered_ds.remove_columns([col for col in filtered_ds.column_names if col not in ['title', 'selftext']])
    print(f"Number of filtered rows: {len(title_joke_ds)}")
    
    # Print some statistics
    print("\nSample of filtered jokes:")
    print(title_joke_ds[:5])
    
    # Check for any remaining problematic entries
    problematic = title_joke_ds.filter(lambda x: x["selftext"] in ["[deleted]", "[removed]", ""])
    if len(problematic) > 0:
        print(f"\nWARNING: Found {len(problematic)} problematic entries after filtering!")
        print(problematic[:5])
    
    title_joke_ds.to_parquet(FILTERED_PATH)
    return title_joke_ds

def load_processed_prompts():
    if not os.path.exists(FILTERED_PATH):
        print(f"{FILTERED_PATH} not found. Running preprocessing...")
        preprocess_and_save()
    return Dataset.from_parquet(FILTERED_PATH)

if __name__ == "__main__":
    load_processed_prompts()