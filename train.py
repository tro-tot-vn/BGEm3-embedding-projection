from typing import List, Dict, Optional
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
        """
        Symmetric InfoNCE loss
        
        Args:
            q: Query embeddings [B, D]
            d: Document embeddings [B, D]
            
        Returns:
            Symmetric loss (q2d + d2q) / 2
        """
        # q, d: [B, D], đã L2-norm => cosine = q @ d.T
        logits = (q @ d.t()) / self.tau                 # [B,B]
        labels = torch.arange(q.size(0), device=q.device)
        
        # Symmetric loss
        loss_q2d = F.cross_entropy(logits, labels)       # q matches d
        loss_d2q = F.cross_entropy(logits.t(), labels)   # d matches q
        loss = (loss_q2d + loss_d2q) / 2
        return loss

    def weighted_info_nce(
        self, 
        q: torch.Tensor, 
        p: torch.Tensor,
        hard_neg: List[torch.Tensor],
        hard_neg_weights: List[List[float]]
    ) -> torch.Tensor:
        """
        InfoNCE with weighted hard negatives
        
        Strategy: Per-query loss calculation with weighted negatives
        
        Args:
            q: Query embeddings [B, D]
            p: Positive embeddings [B, D]
            hard_neg: List of hard negative embeddings per query, each [N_i, D]
            hard_neg_weights: List of weight lists per query
            
        Returns:
            Weighted loss scalar
        """
        B = q.size(0)
        device = q.device
        total_loss = 0.0
        
        for i in range(B):
            # Get query, positive, and hard negatives for this example
            q_i = q[i:i+1]  # [1, D]
            p_i = p[i:i+1]  # [1, D]
            
            # Build negatives: in-batch + hard negatives
            # In-batch negatives: all positives except current
            in_batch_neg = torch.cat([p[:i], p[i+1:]], dim=0)  # [B-1, D]
            
            if len(hard_neg[i]) > 0:
                hn_i = hard_neg[i]  # [N_i, D]
                all_neg = torch.cat([in_batch_neg, hn_i], dim=0)  # [B-1+N_i, D]
                
                # Weights: in-batch get 1.0, hard negatives get their weights
                in_batch_weights = torch.ones(B-1, device=device)
                hn_weights = torch.tensor(hard_neg_weights[i], device=device, dtype=torch.float32)
                neg_weights = torch.cat([in_batch_weights, hn_weights], dim=0)  # [B-1+N_i]
            else:
                all_neg = in_batch_neg
                neg_weights = torch.ones(B-1, device=device)
            
            # Compute similarities
            pos_sim = (q_i @ p_i.t()) / self.tau  # [1, 1]
            neg_sim = (q_i @ all_neg.t()) / self.tau  # [1, B-1+N_i]
            
            # Apply weights to negative logits
            weighted_neg_sim = neg_sim * neg_weights.unsqueeze(0)  # [1, B-1+N_i]
            
            # Concatenate positive and weighted negatives
            logits = torch.cat([pos_sim, weighted_neg_sim], dim=1)  # [1, 1+B-1+N_i]
            
            # Label is 0 (first position is positive)
            label = torch.tensor([0], device=device)
            
            # Cross-entropy loss for this query
            loss_i = F.cross_entropy(logits, label)
            total_loss += loss_i
        
        # Average over batch
        return total_loss / B

    def training_step(self, batch: Dict[str, List[str]]):
        """
        Training step with optional weighted hard negatives
        
        Args:
            batch: Dict with keys:
                - "query": List[str]
                - "pos": List[str]
                - "hard_neg": Optional[List[List[str]]]
                - "hard_neg_weights": Optional[List[List[float]]]
                
        Returns:
            Loss scalar
        """
        device = next(self.model.parameters()).device
        
        # Encode query and positive
        q = self.model(batch["query"], device=device)   # [B, D]
        p = self.model(batch["pos"], device=device)     # [B, D]
        
        # Check if hard negatives are provided
        if "hard_neg" in batch and any(len(hn) > 0 for hn in batch["hard_neg"]):
            # Encode hard negatives per query
            hard_neg_embs = []
            for hn_list in batch["hard_neg"]:
                if len(hn_list) > 0:
                    hn_emb = self.model(hn_list, device=device)  # [N_i, D]
                    hard_neg_embs.append(hn_emb)
                else:
                    # Empty tensor for queries without hard negatives
                    hard_neg_embs.append(torch.empty(0, q.size(1), device=device))
            
            # Use weighted loss
            loss = self.weighted_info_nce(q, p, hard_neg_embs, batch["hard_neg_weights"])
        else:
            # Standard symmetric InfoNCE
            loss = self.info_nce(q, p)
        
        return loss
