import json
import random
import shutil
import os
import torch
from tqdm import tqdm
from train_reward_model import RewardModel, bt_loss
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import pandas as pd
import gc

random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Clear any existing memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("Loading LLaMA-2...")
# Load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load model on CPU first
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="cpu"  # Force CPU loading
)
llama_model.eval()

def compute_prompt_entropy(prompt):
    with torch.no_grad():
        # Move input to CPU
        system_prompt = "You're an AI assistant. Provide a funny response to the following question, be succinct"
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\n Assistant:"
        input_ids = llama_tokenizer(full_prompt, return_tensors="pt").input_ids
        
        outputs = llama_model(input_ids)
        logits = outputs.logits[:, -1, :]
        # More numerically stable softmax
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits, dim=-1)

        # More numerically stable entropy calculation
        mask = probs > 1e-10  # Only consider non-zero probabilities
        entropy = -torch.sum(probs[mask] * torch.log(probs[mask]), dim=-1).item()
        
        return entropy


class JsonlDataset(Dataset):
    def __init__(self, path):
        self.samples = [json.loads(line) for line in open(path)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["prompt"], s["response1"], s["response2"], s["label"]


def select_top_k_uncertain_samples(unlabeled_pool, reward_model, k=50, pool_size=200):
    reward_model.eval()
    scored = []
    
    # Randomly sample a subset of the unlabeled pool
    if len(unlabeled_pool) > pool_size:
        pool_subset = random.sample(unlabeled_pool, pool_size)
    else:
        pool_subset = unlabeled_pool
    
    print(f"Evaluating {len(pool_subset)} samples from unlabeled pool of size {len(unlabeled_pool)}")

    for item in tqdm(pool_subset, desc="Scoring prompts"):
        prompt = item["prompt"]
        r1 = item["response1"]
        r2 = item["response2"]

        entropy = compute_prompt_entropy(f"Q: {prompt}\nA:")
        with torch.no_grad():
            s1 = reward_model.score(prompt, r1).item()
            s2 = reward_model.score(prompt, r2).item()
        score = entropy - abs(s1 - s2)
        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:k]]


def compute_test_loss(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for prompts, r1s, r2s, labels in dataloader:
            s1 = model.score(prompts, r1s)
            s2 = model.score(prompts, r2s)
            total_loss += bt_loss(s1, s2, labels).item()
    return total_loss / len(dataloader)

def main():
    with open("reward_val.jsonl") as f:
        val_data = [json.loads(line) for line in f]
    with open("unlabeled_pool.jsonl") as f:
        unlabeled = [json.loads(line) for line in f]
    with open("reward_train.jsonl") as f:
        labeled = [json.loads(line) for line in f]

    if not os.path.exists("backup_reward_train.jsonl"):
        shutil.copy("reward_train.jsonl", "backup_reward_train.jsonl")

    val_loader = DataLoader(JsonlDataset("reward_val.jsonl"), batch_size=8)
    losses = []

    model = RewardModel().to(device)

    for round_id in range(1, 11):
        print(f"\n=== Round {round_id} ===")

        # Print 5 examples of entropy and BT loss before training
        print("\nExamples before training:")
        print("-" * 80)
        for i in range(5):
            if i < len(labeled):
                item = labeled[i]
                prompt = item["prompt"]
                r1 = item["response1"]
                r2 = item["response2"]
                
                # Compute entropy
                norm_entropy = compute_prompt_entropy(f"Q: {prompt}\nA:")
                
                # Compute BT loss
                with torch.no_grad():
                    s1 = model.score(prompt, r1)
                    s2 = model.score(prompt, r2)
                    loss = bt_loss(s1, s2, torch.tensor([1]).to(device))
                
                print(f"\nExample {i+1}:")
                print(f"Prompt: {prompt}")
                print(f"Normalized Entropy: {norm_entropy:.4f}")
                print(f"Response 1: {r1}")
                print(f"Response 2: {r2}")
                print(f"BT Loss: {loss.item():.4f}")
                print(f"Uncertainty score (norm_entropy - loss): {norm_entropy - loss.item():.4f}")
                print("-" * 80)

        # Use LLaMA entropy for selection
        top50 = select_top_k_uncertain_samples(unlabeled, model, k=50, pool_size=200)
        labeled.extend(top50)
        unlabeled = [ex for ex in unlabeled if ex not in top50]

        # Save updated labeled pool
        with open("current_labeled.jsonl", "w") as f:
            for item in labeled:
                f.write(json.dumps(item) + "\n")

        train_loader = DataLoader(JsonlDataset("current_labeled.jsonl"), batch_size=8, shuffle=True)
        model.train()
        for prompts, r1s, r2s, labels in train_loader:
            s1 = model.score(prompts, r1s)
            s2 = model.score(prompts, r2s)
            loss = bt_loss(s1, s2, labels)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            model.optimizer.step()

        test_loss = compute_test_loss(model, val_loader)
        print(f"[Round {round_id}] Test Loss: {test_loss:.4f}")
        losses.append(test_loss)
        print(f"Current losses list: {losses}")
        
        # Save model after each round
        model_path = f"models/reward_model_round_{round_id}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        with open(f"labeled_round{round_id}.jsonl", "w") as f:
            for item in labeled:
                f.write(json.dumps(item) + "\n")

    # Save final model
    final_model_path = "models/reward_model_active.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot loss curve
    plt.plot(range(1, 11), losses, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.title("Active Learning: Reward Model Loss Over Rounds")
    plt.savefig("active_baseline_loss_curve.png")
    print("Loss curve saved to active_baseline_loss_curve.png")

    df = pd.DataFrame({
        "round": list(range(1, round_id + 1)),
        "test_loss": losses
    })
    df.to_csv("Active_learning_loss_log.csv", index=False)
    print("Loss log saved to Active_learning_loss_log.csv")

if __name__ == "__main__":
    main()

#1 0.506
#2 0.1425
#3 0.0644
#4 0.0359
#5 0.0281
#6 0.0159
#