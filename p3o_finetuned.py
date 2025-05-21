
import os
import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Optional: set huggingface cache paths to avoid quota issues
os.environ["HF_HOME"] = "/net/projects/ycleong/heqianyi926/Humor_LLM_Tuning/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/net/projects/ycleong/heqianyi926/Humor_LLM_Tuning/hf_cache"

model_path = "/net/projects/ycleong/heqianyi926/Humor_LLM_Tuning/llama2"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config)

with open("scored_unlabeled.jsonl") as f:
    data = [json.loads(line) for line in f]

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(1):
    random.shuffle(data)
    total_loss = 0
    for sample in tqdm(data, desc=f"Epoch {epoch}"):
        prompt = sample["prompt"]
        r1, r2 = sample["response1"], sample["response2"]
        s1, s2 = sample["score1"], sample["score2"]
        pref = 1 if s1 > s2 else 0

        chosen = r1 if pref else r2
        rejected = r2 if pref else r1

        chosen_ids = tokenizer(prompt + "\n" + chosen, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        rejected_ids = tokenizer(prompt + "\n" + rejected, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()

        logits_chosen = model(chosen_ids).logits[:, :-1]
        logits_rejected = model(rejected_ids).logits[:, :-1]

        logp_chosen = torch.nn.functional.log_softmax(logits_chosen, dim=-1)
        logp_rejected = torch.nn.functional.log_softmax(logits_rejected, dim=-1)

        ratio = (logp_chosen.sum() - logp_rejected.sum()).exp()
        log_diff = logp_chosen.sum() - logp_rejected.sum()
        loss = -torch.nn.functional.logsigmoid(log_diff)  # 更稳定的loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: avg loss = {total_loss / len(data):.4f}")

# Save the fine-tuned model
output_dir = "/net/projects/ycleong/heqianyi926/Humor_LLM_Tuning/llama2_finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
