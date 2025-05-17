import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

def load_model():
    print("Loading Llama 2 model...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # You can also use "meta-llama/Llama-2-13b-chat-hf" for better results
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_joke(title, model, tokenizer):
    try:
        prompt = f"""generate a response to the question.

Setup: {title}
response:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the punchline part
        punchline = response.split("response:")[-1].strip()
        return punchline
    except Exception as e:
        print(f"Error generating joke for title '{title}': {str(e)}")
        return None

def main():
    # Load the model and tokenizer
    model, tokenizer = load_model()
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_parquet('filtered_jokes_titles_text.parquet')
    
    # Create a new column for Llama responses
    df['llama_response'] = None
    
    # Process each title
    for idx in tqdm(df.index, desc="Generating jokes"):
        title = df.loc[idx, 'title']
        llama_response = generate_joke(title, model, tokenizer)
        df.loc[idx, 'llama_response'] = llama_response
        
        # Save progress every 10 jokes
        if idx % 10 == 0:
            df.to_parquet('jokes_with_llama.parquet')
        
        # Add a small delay to avoid memory issues
        time.sleep(0.5)
    
    # Save the final result
    df.to_parquet('jokes_with_llama.parquet')
    print(f"Joke generation completed! Generated {len(df)} responses.")

if __name__ == "__main__":
    main() 