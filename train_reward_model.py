import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import json
from tqdm import tqdm

class RewardDataset(Dataset):
    def __init__(self, path):
        self.samples = [json.loads(line) for line in open(path)]
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["prompt"], sample["response1"], sample["response2"], sample["label"]

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )

    def score(self, prompt, response):
        enc = self.tokenizer(prompt, response, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.bert.device)
        outputs = self.bert(**enc).last_hidden_state[:, 0]  # [CLS]
        return self.mlp(outputs).squeeze(-1)

    def score_pair(self, prompts, responses1, responses2):
        s1 = self.score(prompts, responses1)
        s2 = self.score(prompts, responses2)
        return s1, s2

def bt_loss(score1, score2, label):
    margin = score1 - score2
    label = label.to(margin.device)
    adjusted_margin = (1 - 2 * label.float()) * margin
    return -torch.nn.functional.logsigmoid(adjusted_margin).mean()

def compute_pairwise_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for prompts, r1s, r2s, labels in dataloader:
            s1 = model.score(prompts, r1s)
            s2 = model.score(prompts, r2s)
            preds = (s1 < s2).long()
            labels = labels.to(preds.device)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total

def train():
    train_set = RewardDataset("reward_train.jsonl")
    test_set = RewardDataset("reward_val.jsonl")
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)

    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1):
        model.train()
        total_loss = 0
        for prompts, r1s, r2s, labels in tqdm(train_loader, desc="Training"):
            s1 = model.score(prompts, r1s)
            s2 = model.score(prompts, r2s)
            loss = bt_loss(s1, s2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for prompts, r1s, r2s, labels in tqdm(test_loader, desc="Evaluating"):
            s1 = model.score(prompts, r1s)
            s2 = model.score(prompts, r2s)
            test_loss += bt_loss(s1, s2, labels).item()
    test_loss /= len(test_loader)
    test_acc = compute_pairwise_accuracy(model, test_loader)

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "reward_model_distilbert.pt")
    print("Reward model saved.")

if __name__ == "__main__":
    train()
