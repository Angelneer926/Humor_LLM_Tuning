import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, create_reference_model
#from trl.core import AcceleratePPOTrainer as PPOTrainer
import numpy as np
from train_reward_model import RewardModel
import os
from tqdm import tqdm
import json
from peft import LoraConfig, get_peft_model
import random
from trl import AutoModelForCausalLMWithValueHead
from datasets import Dataset

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["HF_HOME"] = "/net/scratch2/kzhao1/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/net/scratch2/kzhao1/huggingface"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained reward model
reward_model = RewardModel().to(device)
reward_model.load_state_dict(torch.load("/net/scratch2/kzhao1/active_project/reward_model_round_6.pt"))
#reward_model.load_state_dict(torch.load("models/reward_model_round_6.pt", map_location=device))
reward_model.eval()


# Load LLaMA2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"

#"meta-llama/Llama-2-7B-Chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

# Set up generation config
generation_config = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    num_return_sequences=1,  # Use 2 if you want pairs again
    temperature =0.8,
    top_p=0.9,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Load prompts for training
def load_prompts(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    # Randomly sample 500 prompts (reduced from 1000)


all_samples = load_prompts("/net/scratch2/kzhao1/active_project/unlabeled_pool.jsonl")
print(f"Number of samples loaded: {len(all_samples)}")
print(f"First sample: {all_samples[0]}")

random.seed(42)  # for reproducibility
samples = random.sample(all_samples, 500)
print(f"Number of sampled prompts: {len(samples)}")
print(f"First sampled prompt: {samples[0]}")

# Convert samples to Dataset format
train_dataset = Dataset.from_list([{"prompt": sample["prompt"]} for sample in samples])
print(f"Dataset size: {len(train_dataset)}")
print(f"First dataset item: {train_dataset[0]}")

from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["prompt"]
    
wrapped_dataset = PromptDataset(train_dataset) 

# Load model in float16 (or float32 if not supported)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Reduced rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
# Wrap base model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)


#model.print_trainable_parameters()

# Create a reference model for PPO (on CPU to save memory)
ref_model = None #create_reference_model(model)
#ref_model = ref_model.to('cpu')  # Move reference model to CPU

# PPO configuration with memory optimizations
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,  # Reduced batch size
    mini_batch_size=1,
    gradient_accumulation_steps=2,  # Increased gradient accumulation
    #early_stopping=True,
    #target_kl=0.1,
    ppo_epochs=4,
    seed=42,
    init_kl_coef=0.0,
    #output_dir="/net/scratch2/kzhao1/active_project/checkpoints"
    
    #adap_kl_ctrl=True,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    #processing_class=tokenizer,  # We'll handle processing manually
    #reward_model=reward_model,
    dataset=wrapped_dataset  # Using our converted dataset
)

# System prompt for LLaMA2
SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to provide funny responses to questions. \nKeep your responses concise do not include any other information. Focus on being humorous."""

def format_prompt(prompt):
    return f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"



# Training loop
def train_ppo():
    # Load prompts from your dataset

    print(f"Training on {len(samples)} randomly sampled prompts")
    
    for epoch in range(3):  # Number of PPO epochs
        print(f"\nEpoch {epoch + 1}")
        
        for batch_start in tqdm(range(0, len(samples), ppo_config.batch_size)):
            batch_samples = samples[batch_start:batch_start + ppo_config.batch_size]
            batch_prompts = [sample["prompt"] for sample in batch_samples]
            
            # Format prompts with system prompt
            formatted_prompts = [format_prompt(prompt) for prompt in batch_prompts]
            
            # Generate responses
            #query_tensors = [tokenizer.encode(prompt, return_tensors="pt").squeeze() for prompt in formatted_prompts]
            tokenized = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized["input_ids"].to(device)
            query_tensors = [t for t in input_ids] 
            attention_mask = tokenized["attention_mask"].to(device)
            
            # Generate two responses for each prompt using num_return_sequences
            response_tensors = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                num_return_sequences=2,
                do_sample=True,
            )

            # Reshape to group responses per prompt
            response_tensors = response_tensors.view(len(batch_prompts), 2, -1)
            response_pairs = [
                    (tokenizer.decode(r[0], skip_special_tokens=True),
                    tokenizer.decode(r[1], skip_special_tokens=True))
                    for r in response_tensors
                ]
            # Print example pairs for monitoring
            if batch_start % 100 == 0:
                print("\nExample response pairs:")
                for i in range(min(2, len(batch_prompts))):
                    print(f"\nPrompt: {batch_prompts[i]}")
                    print(f"Response 1: {response_pairs[i][0]}")
                    print(f"Response 2: {response_pairs[i][1]}")
                    print("-" * 80)
            
            # Get rewards and determine chosen responses
            chosen_responses = []
            chosen_tensors = []
            
            for sample, (r1, r2), (t1, t2) in zip(batch_samples, response_pairs, response_tensors):
                with torch.no_grad():
                    # Extract only the assistant's answer (after the last "Assistant:")
                    r1_clean = r1.split("Assistant:")[-1].strip()
                    r2_clean = r2.split("Assistant:")[-1].strip()
                    
                    # Get reward scores for both responses
                    s1 = reward_model.score(sample["prompt"], r1_clean).item()
                    s2 = reward_model.score(sample["prompt"], r2_clean).item()
                    
                    # Keep the better response
                    if s1 > s2:
                        chosen_responses.append(r1)
                        chosen_tensors.append(t1)
                    else:
                        chosen_responses.append(r2)
                        chosen_tensors.append(t2)
            
            # Create rewards for each response tensor
            chosen_rewards = []
            for t in chosen_tensors:
                # Create reward tensor of same length as response
                reward = torch.zeros_like(t, dtype=torch.float)
                reward[-1] = 1.0  # assign reward only at the final token
                chosen_rewards.append(reward)
            
            # Run PPO step with chosen responses
            stats = ppo_trainer.step(query_tensors, chosen_tensors, chosen_rewards)
            
            # Print stats
            print(f"Mean reward: {sum(r[-1].item() for r in chosen_rewards) / len(chosen_rewards):.4f}")
            print(f"PPO stats: {stats}")
            
            # Save model periodically
            #if (batch_start + ppo_config.batch_size) % 100 == 0:
             #   model.save_pretrained(f"models/llama2_ppo_lora_epoch{epoch+1}_batch{batch_start}")
              #  print(f"Model saved at batch {batch_start}")

if __name__ == "__main__":
    train_ppo()
    # Save the final model after training
    model.save_pretrained("llama2_ppo_lora_final")
    print("Final model saved at models/llama2_ppo_lora_final") 