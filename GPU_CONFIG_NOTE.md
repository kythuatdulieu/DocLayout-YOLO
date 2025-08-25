# GPU Configuration Note

This implementation is optimized for RTX 2050 4GB GPU training. The current test environment uses CPU for validation, but for actual training:

## For RTX 2050 4GB GPU Training:

Update `configs/hardware_configs.yaml`:

```yaml
local_development:
  device: "0"           # Use GPU
  batch_size: 8         # Optimized for 4GB VRAM  
  workers: 4            # For multi-core CPU
  image_size: 512       # Memory efficient
  base_epochs: 50       # Full training
  refinement_epochs: 15 # Sufficient convergence
  mixed_precision: true # Essential for 4GB GPU
```

## Current Test Configuration (CPU):

```yaml
local_development:
  device: "cpu"         # CPU testing
  batch_size: 4         # Reduced for CPU
  image_size: 256       # Smaller for testing
  base_epochs: 3        # Quick validation
  refinement_epochs: 2  # Minimal testing
```

## Quick GPU Setup:

1. Install CUDA-compatible PyTorch
2. Change device from "cpu" to "0" in hardware_configs.yaml
3. Increase batch_size, epochs, and image_size as needed
4. Run: `python train_fast.py --model n --refinement`

The optimizations are designed for 4GB GPU constraints while maintaining scalability.