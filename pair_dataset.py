from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str,str]]):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        x = self.pairs[i]
        return {"query": x["query"], "pos": x["pos"]}

def collate(batch):
    return {
        "query": [b["query"] for b in batch],
        "pos":   [b["pos"]   for b in batch],
    }

from model import BGEM3WithHead
from train import ContrastiveTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3WithHead(d_out=128, freeze_encoder=True, use_layernorm=False).to(device)
trainer = ContrastiveTrainer(model)

optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4, weight_decay=0.01)

pairs = [
    {"query":"phòng trọ q10 wc riêng 25m2 5tr5",
     "pos":"Cho thuê phòng Quận 10, 25m², WC khép kín, giá 5.5 triệu/tháng"},
    # ... n mẫu thật từ log click/contact/save ...
]
loader = DataLoader(PairDataset(pairs), batch_size=128, shuffle=True, collate_fn=collate, drop_last=True)

model.train()
for epoch in range(2):
    for batch in loader:
        optimizer.zero_grad()
        loss = trainer.training_step(batch)
        loss.backward()
        optimizer.step()
    print("epoch", epoch, "loss", float(loss))
