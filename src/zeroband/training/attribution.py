import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import TYPE_CHECKING

from zeroband.utils.models import ModelType


class AttributionHead(nn.Module):
    """
    Attribution head that learns to weight advantages per token.
    Projects hidden states to scalars, then softmax to get attribution weights.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1, bias=False)
        # Initialize with small weights to start near uniform attribution
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        
    def forward(self, hidden_states: Float[Tensor, "batch seq hidden"]) -> Float[Tensor, "batch seq"]:
        """
        Compute attribution weights from hidden states.
        
        Args:
            hidden_states: Hidden states from the model [batch, seq, hidden]
            
        Returns:
            attribution_weights: Softmax normalized weights [batch, seq] that sum to 1 per batch
        """
        # Project each hidden state to a scalar
        attribution_logits = self.projection(hidden_states).squeeze(-1)  # [batch, seq]
        
        # Apply softmax to get weights that sum to 1
        attribution_weights = F.softmax(attribution_logits, dim=-1)
        
        return attribution_weights


def apply_attribution_to_advantages(
    advantages: Float[Tensor, "batch seq"],
    attribution_weights: Float[Tensor, "batch seq"],
    loss_mask: Float[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    """
    Apply attribution weights to advantages.
    
    Args:
        advantages: Original advantages [batch, seq]
        attribution_weights: Attribution weights [batch, seq] 
        loss_mask: Mask for valid tokens [batch, seq]
        
    Returns:
        attributed_advantages: Advantages weighted by attribution [batch, seq]
    """
    # Only apply attribution to tokens that contribute to loss
    valid_attribution = attribution_weights * loss_mask
    
    # Renormalize attribution weights for valid tokens only
    valid_sum = valid_attribution.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_attribution = valid_attribution / valid_sum
    
    # Apply attribution weights to advantages
    # Scale by sequence length to maintain similar magnitude to original advantages
    seq_len = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    attributed_advantages = advantages * normalized_attribution * seq_len
    
    return attributed_advantages


class AttributionWrapper(nn.Module):
    """
    Wrapper that adds attribution head to an existing model without modifying the model class.
    """
    
    def __init__(self, model: ModelType):
        super().__init__()
        self.model = model
        self.attribution_head = AttributionHead(model.config.hidden_size)
        self._last_hidden_states = None
        
    def forward(self, input_ids, position_ids=None, **kwargs):
        """Forward pass that captures hidden states and computes attribution weights."""
        # Always request hidden states for attribution computation
        kwargs_with_hidden = kwargs.copy()
        kwargs_with_hidden['output_hidden_states'] = True
        
        # Get model outputs with hidden states
        outputs = self.model(
            input_ids=input_ids, 
            position_ids=position_ids, 
            **kwargs_with_hidden
        )
        
        # Store hidden states for attribution computation (detached to save memory)
        self._last_hidden_states = outputs.hidden_states[-1].detach()
        
        return outputs
    
    def get_attribution_weights(self, loss_mask: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
        """Get attribution weights from the last forward pass."""
        if self._last_hidden_states is None:
            raise RuntimeError("Must call forward() before get_attribution_weights()")
        
        # Move hidden states back to compute device if needed    
        hidden_states = self._last_hidden_states.to(loss_mask.device)
        attribution_weights = self.attribution_head(hidden_states)
        
        # Clear stored hidden states to free memory
        self._last_hidden_states = None
        
        return attribution_weights
    
    def parameters(self, recurse: bool = True):
        """Return parameters of both model and attribution head."""
        for param in self.model.parameters(recurse):
            yield param
        for param in self.attribution_head.parameters(recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters of both model and attribution head."""
        for name, param in self.model.named_parameters(prefix=f"{prefix}model." if prefix else "model.", recurse=recurse):
            yield name, param
        for name, param in self.attribution_head.named_parameters(prefix=f"{prefix}attribution_head." if prefix else "attribution_head.", recurse=recurse):
            yield name, param
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict including attribution head."""
        if destination is None:
            destination = {}
        
        # Get model state dict
        model_state = self.model.state_dict(prefix=f"{prefix}model.", keep_vars=keep_vars)
        destination.update(model_state)
        
        # Get attribution head state dict
        attribution_state = self.attribution_head.state_dict(prefix=f"{prefix}attribution_head.", keep_vars=keep_vars)
        destination.update(attribution_state)
        
        return destination
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict including attribution head."""
        model_state = {}
        attribution_state = {}
        
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key[6:]] = value  # Remove "model." prefix
            elif key.startswith("attribution_head."):
                attribution_state[key[17:]] = value  # Remove "attribution_head." prefix
            else:
                # For compatibility with direct model state dicts
                model_state[key] = value
        
        # Load model state
        missing_keys_model, unexpected_keys_model = self.model.load_state_dict(model_state, strict=False)
        
        # Load attribution head state if available
        missing_keys_attr, unexpected_keys_attr = [], []
        if attribution_state:
            missing_keys_attr, unexpected_keys_attr = self.attribution_head.load_state_dict(attribution_state, strict=False)
        
        if strict and (missing_keys_model or unexpected_keys_model or missing_keys_attr or unexpected_keys_attr):
            error_msgs = []
            if missing_keys_model or missing_keys_attr:
                error_msgs.append(f"Missing keys: {missing_keys_model + missing_keys_attr}")
            if unexpected_keys_model or unexpected_keys_attr:
                error_msgs.append(f"Unexpected keys: {unexpected_keys_model + unexpected_keys_attr}")
            raise RuntimeError(f"Error loading state dict: {', '.join(error_msgs)}")
        
        return missing_keys_model + missing_keys_attr, unexpected_keys_model + unexpected_keys_attr
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name) 