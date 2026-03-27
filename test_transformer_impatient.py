#!/usr/bin/env python
"""
Test script for TransformerReceiverImpatient.
Verifies that the implementation produces the correct output shapes and runs without errors.
"""

import torch
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egg.core.reinforce_wrappers import TransformerReceiverImpatient
from egg.zoo.channel.archs import Receiver

def test_transformer_receiver_impatient():
    """Test basic forward pass of TransformerReceiverImpatient."""

    print("="*70)
    print("Testing TransformerReceiverImpatient")
    print("="*70)

    # Test parameters
    n_features = 1000
    vocab_size = 40
    max_len = 30
    embed_dim = 256
    num_heads = 8
    hidden_size = 512
    num_layers = 2
    batch_size = 16

    print(f"\nTest Configuration:")
    print(f"  n_features: {n_features}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_len: {max_len}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  batch_size: {batch_size}")

    # Create receiver agent
    print("\n" + "="*70)
    print("Creating receiver agent...")
    receiver_agent = Receiver(n_features=n_features, n_hidden=embed_dim)

    # Wrap with TransformerReceiverImpatient
    print("Creating TransformerReceiverImpatient wrapper...")
    receiver = TransformerReceiverImpatient(
        agent=receiver_agent,
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_features=n_features,
        causal=True
    )

    # Test forward pass
    print("\n" + "="*70)
    print("Testing forward pass...")

    # Create random message
    message = torch.randint(0, vocab_size, (batch_size, max_len))
    print(f"Input message shape: {message.shape}")

    # Forward pass
    try:
        receiver.eval()  # Set to eval mode
        sequence, logits, entropy = receiver(message)

        print(f"\nOutput shapes:")
        print(f"  sequence: {sequence.shape} (expected: [{batch_size}, {max_len}, {n_features}])")
        print(f"  logits: {logits.shape} (expected: [{batch_size}, {max_len}])")
        print(f"  entropy: {entropy.shape} (expected: [{batch_size}, {max_len}])")

        # Verify shapes
        assert sequence.shape == (batch_size, max_len, n_features), f"Sequence shape mismatch!"
        assert logits.shape == (batch_size, max_len), f"Logits shape mismatch!"
        assert entropy.shape == (batch_size, max_len), f"Entropy shape mismatch!"

        print("\n✓ All shapes correct!")

        # Check for NaN or Inf
        assert not torch.isnan(sequence).any(), "NaN detected in sequence!"
        assert not torch.isinf(sequence).any(), "Inf detected in sequence!"
        assert not torch.isnan(logits).any(), "NaN detected in logits!"
        assert not torch.isinf(logits).any(), "Inf detected in logits!"
        assert not torch.isnan(entropy).any(), "NaN detected in entropy!"
        assert not torch.isinf(entropy).any(), "Inf detected in entropy!"

        print("✓ No NaN or Inf values detected!")

        # Test training mode
        print("\n" + "="*70)
        print("Testing training mode...")
        receiver.train()
        sequence_train, logits_train, entropy_train = receiver(message)

        assert sequence_train.shape == (batch_size, max_len, n_features), "Training mode shape mismatch!"
        print("✓ Training mode works!")

        # Test with explicit lengths
        print("\n" + "="*70)
        print("Testing with explicit lengths...")
        lengths = torch.randint(5, max_len, (batch_size,))
        sequence_len, logits_len, entropy_len = receiver(message, lengths=lengths)

        assert sequence_len.shape == (batch_size, max_len, n_features), "Explicit lengths shape mismatch!"
        print("✓ Explicit lengths work!")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_transformer_receiver_impatient()
    sys.exit(0 if success else 1)
