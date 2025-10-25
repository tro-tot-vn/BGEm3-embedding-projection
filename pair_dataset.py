from model import BGEM3WithHead
from train import ContrastiveTrainer
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class PairDataset(Dataset):
    """
    Dataset for contrastive learning with optional hard negatives
    
    Supports two modes:
    1. Query-Positive pairs only (original)
    2. Query-Positive-HardNegatives with weights
    """
    def __init__(self, pairs: List[Dict[str, str]], use_hard_neg: bool = False):
        """
        Args:
            pairs: List of training examples
            use_hard_neg: Whether to include hard negatives from dataset
        """
        self.pairs = pairs
        self.use_hard_neg = use_hard_neg

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, i):
        x = self.pairs[i]
        result = {
            "query": x["query"], 
            "pos": x["pos"]
        }
        
        # Add hard negatives if available and requested
        if self.use_hard_neg and "hard_neg" in x:
            hard_negs = x["hard_neg"]
            
            # Extract texts and weights
            result["hard_neg"] = []
            result["hard_neg_weights"] = []
            
            for hn in hard_negs:
                if isinstance(hn, dict):
                    # New format: {"text": "...", "type": [...], "weight": 2.5}
                    result["hard_neg"].append(hn["text"])
                    result["hard_neg_weights"].append(hn.get("weight", 1.0))
                else:
                    # Old format: just strings
                    result["hard_neg"].append(hn)
                    result["hard_neg_weights"].append(1.0)
        
        return result


def collate(batch):
    """
    Collate function for DataLoader
    
    Handles both query-pos pairs and hard negatives with weights
    """
    result = {
        "query": [b["query"] for b in batch],
        "pos":   [b["pos"] for b in batch],
    }
    
    # Check if any batch has hard negatives
    if "hard_neg" in batch[0]:
        result["hard_neg"] = [b.get("hard_neg", []) for b in batch]
        result["hard_neg_weights"] = [b.get("hard_neg_weights", []) for b in batch]
    
    return result


device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3WithHead(d_out=128, freeze_encoder=True,
                      use_layernorm=False).to(device)
trainer = ContrastiveTrainer(model)

optimizer = torch.optim.AdamW(
    model.head.parameters(), lr=2e-4, weight_decay=0.01)

pairs = [
    {"query": "phòng trọ q10 wc riêng 25m2 5tr5",
     "pos": "Cho thuê phòng Quận 10, 25m², WC khép kín, giá 5.5 triệu/tháng"},
    # ... n mẫu thật từ log click/contact/save ...
]
loader = DataLoader(PairDataset(pairs), batch_size=128,
                    shuffle=True, collate_fn=collate, drop_last=True)

if __name__ == "__main__":
    # Demo training code
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = trainer.training_step(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"epoch {epoch}, avg_loss {avg_loss:.4f}")
