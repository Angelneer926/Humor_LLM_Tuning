import torch
from transformers import DistilBertTokenizer, DistilBertModel
import json
from tqdm import tqdm
import argparse

class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.scorer = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def score(self, prompt, response):
        enc = self.tokenizer(prompt, response, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.bert.device)
        outputs = self.bert(**enc).last_hidden_state[:, 0]
        return self.scorer(outputs).squeeze(-1)

    def score_pair(self, prompts, responses1, responses2):
        s1 = self.score(prompts, responses1)
        s2 = self.score(prompts, responses2)
        return s1, s2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="unlabeled_pool.jsonl")
    parser.add_argument("--output", default="scored_unlabeled.jsonl")
    parser.add_argument("--model_path", default="reward_model_distilbert.pt")
    args = parser.parse_args()

    # Load model
    model = RewardModel().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Read and score
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in tqdm(f_in, desc="Scoring unlabeled pool"):
            sample = json.loads(line)
            prompt = sample["prompt"]
            r1 = sample["response1"]
            r2 = sample["response2"]
            with torch.no_grad():
                s1, s2 = model.score_pair([prompt], [r1], [r2])
                sample["score1"] = s1.item()
                sample["score2"] = s2.item()
                sample["margin"] = abs(s1.item() - s2.item())
            f_out.write(json.dumps(sample) + "\n")

    print(f"Done. Scored samples saved to {args.output}")

if __name__ == "__main__":
    main()