from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BGEM3WithHead, TAU

class ContrastiveTrainer(nn.Module):
    def __init__(self, model: BGEM3WithHead, tau: float = TAU):
        super().__init__()
        self.model = model
        self.tau = tau

    def info_nce(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # q, d: [B, D], đã L2-norm => cosine = q @ d.T
        logits = (q @ d.t()) / self.tau                 # [B,B]
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch: Dict[str, List[str]]):
        # batch: {"query": [..], "pos": [..]}
        device = next(self.model.parameters()).device
        q = self.model(batch["query"], device=device)   # [B,128]
        p = self.model(batch["pos"], device=device)     # [B,128]
        loss = self.info_nce(q, p)
        return loss
