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
        InfoNCE with weighted hard negatives (Loss Weighting Method)
        
        Strategy: Apply weights to exp(similarity) in the loss denominator,
        ensuring that important hard negatives (high weights) contribute more
        to the loss without distorting the similarity scores themselves.
        
        Formula:
            loss = -log(exp(sim_pos) / (exp(sim_pos) + Σ w_i × exp(sim_neg_i)))
            
        where w_i is the weight for negative i (from dataset: 2.5 for location, 2.0 for price, etc.)
        
        Args:
            q: Query embeddings [B, D]
            p: Positive embeddings [B, D]
            hard_neg: List of hard negative embeddings per query, each [N_i, D]
            hard_neg_weights: List of weight lists per query (e.g., [[2.5, 2.0], [2.5], ...])
            
        Returns:
            Weighted symmetric loss scalar (q2p + p2q) / 2
        """
        B = q.size(0)
        device = q.device
        
        # ===== Query-to-Positive Direction =====
        total_loss_q2p = 0.0
        
        for i in range(B):
            q_i = q[i:i+1]  # [1, D]
            p_i = p[i:i+1]  # [1, D]
            
            # In-batch negatives: all positives except current
            in_batch_neg = torch.cat([p[:i], p[i+1:]], dim=0)  # [B-1, D]
            
            # Compute similarities (scaled by temperature)
            pos_sim = (q_i @ p_i.t()) / self.tau  # [1, 1]
            in_batch_sim = (q_i @ in_batch_neg.t()) / self.tau  # [1, B-1]
            
            # Compute exponentials
            exp_pos = torch.exp(pos_sim)  # [1, 1]
            exp_in_batch = torch.exp(in_batch_sim)  # [1, B-1]
            
            # Weighted denominator: start with positive + in-batch (weight=1.0)
            weighted_denominator = exp_pos + exp_in_batch.sum()
            
            # Add weighted hard negatives
            if len(hard_neg[i]) > 0:
                hn_i = hard_neg[i]  # [N_i, D]
                
                # Hard negative similarities
                hn_sim = (q_i @ hn_i.t()) / self.tau  # [1, N_i]
                exp_hn = torch.exp(hn_sim)  # [1, N_i]
                
                # Get weights from dataset (e.g., [2.5, 2.0, 1.5])
                weights_tensor = torch.tensor(
                    hard_neg_weights[i], 
                    device=device, 
                    dtype=torch.float32
                ).unsqueeze(0)  # [1, N_i]
                
                # ✅ KEY: Apply weights to exp(similarity), NOT similarity!
                # This correctly scales the contribution of each negative in the loss
                weighted_exp_hn = exp_hn * weights_tensor  # [1, N_i]
                weighted_denominator += weighted_exp_hn.sum()
            
            # Compute negative log likelihood
            # P(positive) = exp_pos / weighted_denominator
            # loss = -log(P(positive))
            loss_i = -torch.log(exp_pos / weighted_denominator)
            total_loss_q2p += loss_i
        
        # ===== Positive-to-Query Direction (Symmetric) =====
        total_loss_p2q = 0.0
        
        for i in range(B):
            p_i = p[i:i+1]  # [1, D] - now treated as "query"
            q_i = q[i:i+1]  # [1, D] - now treated as "positive"
            
            # In-batch negatives: all queries except current
            in_batch_neg = torch.cat([q[:i], q[i+1:]], dim=0)  # [B-1, D]
            
            # Compute similarities
            pos_sim = (p_i @ q_i.t()) / self.tau  # [1, 1]
            in_batch_sim = (p_i @ in_batch_neg.t()) / self.tau  # [1, B-1]
            
            # Compute exponentials
            exp_pos = torch.exp(pos_sim)
            exp_in_batch = torch.exp(in_batch_sim)
            
            # Weighted denominator
            weighted_denominator = exp_pos + exp_in_batch.sum()
            
            # Add weighted hard negatives (if available)
            # Note: Hard negatives are relative to queries, so we reuse them here
            if len(hard_neg[i]) > 0:
                hn_i = hard_neg[i]  # [N_i, D]
                hn_sim = (p_i @ hn_i.t()) / self.tau  # [1, N_i]
                exp_hn = torch.exp(hn_sim)
                
                weights_tensor = torch.tensor(
                    hard_neg_weights[i], 
                    device=device, 
                    dtype=torch.float32
                ).unsqueeze(0)
                
                weighted_exp_hn = exp_hn * weights_tensor
                weighted_denominator += weighted_exp_hn.sum()
            
            # Compute loss
            loss_i = -torch.log(exp_pos / weighted_denominator)
            total_loss_p2q += loss_i
        
        # ===== Symmetric Loss =====
        # Average both directions for better gradient flow
        loss_q2p = total_loss_q2p / B
        loss_p2q = total_loss_p2q / B
        symmetric_loss = (loss_q2p + loss_p2q) / 2
        
        return symmetric_loss

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
