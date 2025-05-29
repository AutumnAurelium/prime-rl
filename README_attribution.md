# Learned Advantage Attribution for GRPO

This implements a novel RL trick for advantage attribution in LLM RL training. Instead of using uniform advantages across all tokens (as in standard GRPO), the model learns to weight advantages per token based on the hidden states.

## How It Works

1. **Attribution Head**: A linear projection from hidden states to scalars, followed by softmax normalization
2. **Advantage Weighting**: Per-token advantages are multiplied by learned attribution weights
3. **Training**: The attribution head learns alongside the main model to optimize advantage assignment

## Usage

### Enable Attribution

Add to your training config:

```python
train = TrainConfig(
    use_attribution=True,  # Enable learned advantage attribution
    # ... other config options
)
```

Or via command line:

```bash
python -m zeroband.train --train.use_attribution=true
```

### Key Components

- `AttributionWrapper`: Wraps the model to add attribution functionality
- `grpo_loss_with_attribution`: Modified loss function that applies attribution weights
- Attribution metrics logged to WandB under `attribution/` prefix

### Monitoring

The following metrics are logged to track attribution behavior:

- `attribution/attribution_entropy`: Entropy of attribution weights (higher = more uniform)
- `attribution/attribution_max`: Maximum attribution weight per sequence
- `attribution/attribution_min`: Minimum attribution weight per sequence  
- `attribution/attribution_std`: Standard deviation of attribution weights

### Memory Considerations

- Attribution requires computing hidden states, adding ~15-20% memory overhead
- Hidden states are detached and cleared after attribution computation to minimize memory usage
- Works with FSDP and distributed training

### Checkpointing

Attribution head weights are automatically saved/loaded with model checkpoints. The wrapper maintains compatibility with existing checkpoint loading.

### Expected Benefits

- **Better Credit Assignment**: Learn which tokens deserve more/less advantage attribution
- **Improved Sample Efficiency**: More targeted learning signal
- **Interpretability**: Understand which parts of responses matter for rewards

### Implementation Details

- Attribution head initialized with small weights (std=0.01) for near-uniform start
- Softmax normalization ensures attribution weights sum to 1 per sequence
- Only valid tokens (according to loss_mask) receive attribution
- Gradients flow through both the main model and attribution head

## Testing

Run the test script to verify implementation:

```bash
python test_attribution.py
```

This will test the attribution mechanism on a small model without requiring GPU resources. 