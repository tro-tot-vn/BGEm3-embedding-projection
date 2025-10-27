"""
BGE-M3 Projection Model for Hugging Face Transformers

A lightweight projection head trained on top of frozen BGE-M3 encoder
for Vietnamese rental property search.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class BGEM3ProjectionConfig(PretrainedConfig):
    """
    Configuration class for BGEM3ProjectionModel
    
    Args:
        base_model (str): Base model identifier (default: "BAAI/bge-m3")
        d_in (int): Input dimension from base encoder (default: 1024)
        d_out (int): Output dimension after projection (default: 128)
        use_layernorm (bool): Whether to use LayerNorm in projection head
        freeze_encoder (bool): Whether to freeze the base encoder
        max_length (int): Maximum sequence length for tokenization
    """
    
    model_type = "bgem3_projection"
    
    def __init__(
        self,
        base_model: str = "BAAI/bge-m3",
        d_in: int = 1024,
        d_out: int = 128,
        use_layernorm: bool = False,
        freeze_encoder: bool = True,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.d_in = d_in
        self.d_out = d_out
        self.use_layernorm = use_layernorm
        self.freeze_encoder = freeze_encoder
        self.max_length = max_length


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling with attention mask
    
    Args:
        last_hidden_state: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
    
    Returns:
        pooled: [batch_size, hidden_size]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / counts


class ProjectionHead(nn.Module):
    """
    Projection head: Linear + Optional LayerNorm + L2 Normalization
    """
    
    def __init__(self, d_in: int, d_out: int, use_layernorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.LayerNorm(d_out) if use_layernorm else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_in]
        
        Returns:
            [batch_size, d_out] L2-normalized
        """
        x = self.linear(x)
        if self.ln is not None:
            x = self.ln(x)
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        return x


class BGEM3ProjectionModel(PreTrainedModel):
    """
    BGE-M3 with trainable projection head
    
    This model combines:
    1. Frozen BGE-M3 encoder (1024-dim embeddings)
    2. Trainable projection head (1024 -> d_out, default 128)
    
    Usage:
        >>> from transformers import AutoModel, AutoTokenizer
        >>> 
        >>> model = AutoModel.from_pretrained("your-username/bge-m3-vietnamese-rental-projection", trust_remote_code=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        >>> 
        >>> # Encode texts
        >>> texts = ["Phòng trọ Quận 10, 25m2, giá 5tr"]
        >>> inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        >>> embeddings = model(**inputs).last_hidden_state
        >>> 
        >>> # Or use the encode method
        >>> embeddings = model.encode(texts)
    """
    
    config_class = BGEM3ProjectionConfig
    base_model_prefix = "bgem3_projection"
    supports_gradient_checkpointing = False
    
    def __init__(self, config: BGEM3ProjectionConfig):
        super().__init__(config)
        
        self.config = config
        
        # Load base encoder
        self.encoder = AutoModel.from_pretrained(config.base_model)
        
        # Freeze encoder if specified
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection head (trainable)
        self.head = ProjectionHead(
            d_in=config.d_in,
            d_out=config.d_out,
            use_layernorm=config.use_layernorm
        )
        
        # Initialize tokenizer (for convenience)
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                use_fast=True
            )
        return self._tokenizer
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Forward pass through encoder and projection head
        
        Returns:
            BaseModelOutput with last_hidden_state = projected embeddings [batch_size, d_out]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encode with base model
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        # Mean pooling
        pooled = mean_pool(
            encoder_outputs.last_hidden_state,
            attention_mask
        )  # [batch_size, 1024]
        
        # Project to d_out
        projected = self.head(pooled)  # [batch_size, d_out], L2-normalized
        
        if not return_dict:
            return (projected,)
        
        return BaseModelOutput(
            last_hidden_state=projected,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
    
    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: Optional[int] = None,
        show_progress: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode texts to embeddings (convenience method)
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            max_length: Maximum sequence length (default: config.max_length)
            show_progress: Show progress bar
            device: Target device (default: model device)
        
        Returns:
            Tensor of shape [num_texts, d_out], L2-normalized
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if device is None:
            device = next(self.parameters()).device
        
        if max_length is None:
            max_length = self.config.max_length
        
        self.eval()
        all_embeddings = []
        
        # Optional progress bar
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding")
            except ImportError:
                pass
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.forward(**inputs)
            embeddings = outputs.last_hidden_state
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Compute cosine similarity between texts
        
        Args:
            text1: Single text or list of texts
            text2: Single text or list of texts
        
        Returns:
            Similarity scores (cosine similarity)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity (already L2-normalized, so just dot product)
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        similarity = emb1 @ emb2.T
        
        return similarity.squeeze()


# Register model for AutoModel
try:
    from transformers import AutoModel, AutoConfig
    AutoConfig.register("bgem3_projection", BGEM3ProjectionConfig)
    AutoModel.register(BGEM3ProjectionConfig, BGEM3ProjectionModel)
except Exception as e:
    # Registration may fail if models are already registered
    pass

