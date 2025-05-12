import torch
from transformers import Trainer

class WeightedMSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs.logits.view(-1)
        weights = labels + 1e-4  # Avoid 0 weights
        loss = ((preds - labels) ** 2 * weights).mean()
        return (loss, outputs) if return_outputs else loss
