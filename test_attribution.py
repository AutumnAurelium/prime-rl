#!/usr/bin/env python3
"""
Simple test script for the attribution mechanism.
This can be run to verify the implementation works before full training.
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

# Import our modules
from src.zeroband.training.attribution import AttributionWrapper, apply_attribution_to_advantages
from src.zeroband.training.loss import grpo_loss_with_attribution

def test_attribution_mechanism():
    """Test the attribution mechanism with a small model."""
    print("Testing attribution mechanism...")
    
    # Create a small test model
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 128
    config.vocab_size = 1000
    config.use_cache = False
    
    model = AutoModelForCausalLM.from_config(config)
    
    # Wrap with attribution wrapper
    model_with_attribution = AttributionWrapper(model)
    
    # Create test inputs
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Create test data
    advantages = torch.randn(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len).int()
    original_logprobs = torch.randn(batch_size, seq_len - 1)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Advantages shape: {advantages.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    outputs = model_with_attribution(input_ids=input_ids, position_ids=position_ids)
    logits = outputs.logits
    print(f"✓ Logits shape: {logits.shape}")
    
    # Test attribution weights
    print("Testing attribution weights...")
    attribution_weights = model_with_attribution.get_attribution_weights(loss_mask)
    print(f"✓ Attribution weights shape: {attribution_weights.shape}")
    print(f"✓ Attribution weights sum per sequence: {attribution_weights.sum(dim=-1)}")
    
    # Verify attribution weights sum to 1
    sums = attribution_weights.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
        print("❌ ERROR: Attribution weights don't sum to 1!")
        return False
    
    # Test apply attribution to advantages
    print("Testing advantage attribution...")
    attributed_advantages = apply_attribution_to_advantages(
        advantages, attribution_weights, loss_mask.float()
    )
    print(f"✓ Attributed advantages shape: {attributed_advantages.shape}")
    
    # Test GRPO loss with attribution
    print("Testing GRPO loss with attribution...")
    try:
        loss, clip_ratio, attribution_entropy = grpo_loss_with_attribution(
            logits=logits,
            input_ids=input_ids,
            advantages=advantages,
            attribution_weights=attribution_weights,
            original_logprobs=original_logprobs,
            loss_mask=loss_mask,
            temperature=0.6,
            epsilon_low=0.2,
            epsilon_high=0.2,
            clamp_log_prob_coef=4.0,
            max_tokens=batch_size * seq_len,
        )
        
        print(f"✓ Loss: {loss.item():.4f}")
        print(f"✓ Clip ratio: {clip_ratio.item():.4f}")
        print(f"✓ Attribution entropy: {attribution_entropy.item():.4f}")
    except Exception as e:
        print(f"❌ ERROR in GRPO loss computation: {e}")
        return False
    
    # Test that gradients flow to attribution head
    print("Testing gradient flow...")
    try:
        loss.backward()
        
        # Check that attribution head has gradients
        has_attr_grad = any(p.grad is not None for p in model_with_attribution.attribution_head.parameters())
        has_model_grad = any(p.grad is not None for p in model_with_attribution.model.parameters())
        
        print(f"✓ Attribution head has gradients: {has_attr_grad}")
        print(f"✓ Model has gradients: {has_model_grad}")
        
        if not has_attr_grad:
            print("❌ ERROR: Attribution head not receiving gradients!")
            return False
    except Exception as e:
        print(f"❌ ERROR in gradient computation: {e}")
        return False
    
    # Test state dict operations
    print("Testing state dict operations...")
    try:
        state_dict = model_with_attribution.state_dict()
        has_attr_keys = any('attribution_head' in k for k in state_dict.keys())
        print(f"✓ State dict has attribution_head keys: {has_attr_keys}")
        
        if not has_attr_keys:
            print("❌ ERROR: State dict missing attribution head keys!")
            return False
        
        # Test loading state dict
        model_with_attribution.load_state_dict(state_dict)
        print("✓ State dict load/save test passed")
    except Exception as e:
        print(f"❌ ERROR in state dict operations: {e}")
        return False
    
    print("\n🎉 All tests passed! Attribution mechanism is working correctly.")
    print("\nReady for GPU testing with real training!")
    
    return True

if __name__ == "__main__":
    success = test_attribution_mechanism()
    exit(0 if success else 1) 