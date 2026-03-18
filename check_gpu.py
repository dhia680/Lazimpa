#!/usr/bin/env python
"""Check available GPU backends on MacBook."""

import torch
import sys

print("="*70)
print("MacBook GPU Detection")
print("="*70)
print()

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")
print()

# Check CUDA (NVIDIA - unlikely on Mac)
cuda_available = torch.cuda.is_available()
print(f"CUDA available (NVIDIA GPU): {cuda_available}")
if cuda_available:
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Check MPS (Apple Silicon GPU)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f"MPS available (Apple GPU): {mps_available}")
if mps_available:
    print(f"  MPS backend built: {torch.backends.mps.is_built()}")
print()

# Recommendation
print("="*70)
print("RECOMMENDATION")
print("="*70)

if mps_available:
    print("✓ Use MPS (Apple GPU) - MUCH faster than CPU!")
    print()
    print("Speed comparison (approximate):")
    print("  CPU:  1x")
    print("  MPS:  3-5x faster")
    print("  CUDA: 10-20x faster (not available on Mac)")
    print()
    print("To use MPS:")
    print("  python run_experiments_local.py --config experiments_config_mps.json")
    device = "mps"
elif cuda_available:
    print("✓ Use CUDA (NVIDIA GPU)")
    device = "cuda"
else:
    print("⚠ Only CPU available - experiments will be slower")
    print()
    print("To enable MPS:")
    print("  1. Update PyTorch: pip install --upgrade torch")
    print("  2. Requires PyTorch 1.12+ and macOS 12.3+")
    device = "cpu"

print()
print(f"Recommended device: {device}")
print()

# Test computation on available device
print("="*70)
print("TESTING DEVICE")
print("="*70)

if device == "mps":
    try:
        # Test MPS
        x = torch.randn(1000, 1000).to('mps')
        y = torch.randn(1000, 1000).to('mps')
        z = torch.matmul(x, y)
        print("✓ MPS test successful!")
        print(f"  Computed 1000x1000 matrix multiply on Apple GPU")
    except Exception as e:
        print(f"✗ MPS test failed: {e}")
        print("  Falling back to CPU")
        device = "cpu"
elif device == "cuda":
    x = torch.randn(1000, 1000).to('cuda')
    y = torch.randn(1000, 1000).to('cuda')
    z = torch.matmul(x, y)
    print("✓ CUDA test successful!")
else:
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = torch.matmul(x, y)
    print("✓ CPU test successful!")

print()
print(f"Final recommendation: Use device='{device}'")
print()

sys.exit(0 if device != "cpu" else 1)
