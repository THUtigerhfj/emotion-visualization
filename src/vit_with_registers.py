"""
ViT Wrapper with Test-Time Register Tokens (Ablation: fc2 version)

Based on "Vision Transformers Don't Need Trained Registers" (Jiang et al., 2025).

Ablation version: Intervention is applied on MLP output (after fc2), not intermediate activation.
This is for comparison with the middle-activation approach.

This module provides a wrapper for Vision Transformer models that adds test-time
register tokens to mitigate high-norm outlier artifacts.

The register token is appended to patch embeddings without positional encoding.
During forward pass, discovered register neurons have their activations shifted
to the register token at the MLP OUTPUT stage (after fc2), preventing
high-norm outlier artifacts from propagating through the network.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


def sign_max(tensor):
    """
    Find the value with the largest absolute value (could be positive or negative).

    This is different from max(abs()) because it preserves the sign of the extreme value.

    Args:
        tensor: Input tensor

    Returns:
        The value (positive max or negative min) with the largest absolute value
    """
    pos_max = torch.max(tensor)
    neg_max = torch.min(tensor)
    return pos_max if abs(pos_max) > abs(neg_max) else neg_max


class ViTWithTestTimeRegisters(nn.Module):
    """
    Wrapper for ViT models with test-time register tokens.

    This wrapper:
    1. Appends zero-initialized register tokens to patch embeddings
    2. Applies neuron intervention via forward hooks on MLP activations (mlp.act)
    3. Excludes register tokens from final classification attention

    Args:
        base_model: Pre-trained ViTForImageClassification model
        num_registers: Number of register tokens to append (default: 1, following paper)
        register_neurons: Dict mapping layer_idx -> list of neuron indices to intervene on
    """

    def __init__(
        self,
        base_model: ViTForImageClassification,
        num_registers: int = 1,
        register_neurons: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_registers = num_registers
        self.register_neurons = register_neurons or {}

        # Store original config values for reference
        self.original_seq_length = base_model.config.image_size // base_model.config.patch_size
        self.original_seq_length = self.original_seq_length ** 2  # 14x14 = 196 for 224px images

        # Store hooks for cleanup
        self.intervention_hooks = []

        # Register intervention hooks on model initialization
        self._register_intervention_hooks()

    @property
    def config(self):
        """Expose base model config for compatibility."""
        return self.base_model.config

    def _register_intervention_hooks(self):
        """
        Register forward hooks on MLP output (after fc2) for register neuron intervention.

        ABLATION VERSION: This version intervenes on the MLP output (after fc2), not the intermediate activation.
        This is different from the paper's approach which uses intermediate activation.

        For HuggingFace ViT, the MLP structure is:
            fc1 -> activation_fn (GELU) -> fc2 -> output
        We hook on fc2 output to intervene at the final hidden_size dimension (768).

        This is an ablation study variant to compare with the middle-activation approach.
        """
        # Clear any existing hooks
        self._remove_hooks()

        if not self.register_neurons:
            return

        # Get reference to layers (handle different ViT implementations)
        if hasattr(self.base_model.vit, 'encoder'):
            layers = self.base_model.vit.encoder.layer
        elif hasattr(self.base_model.vit, 'layers'):
            layers = self.base_model.vit.layers
        else:
            raise ValueError("Cannot find ViT layers")

        for layer_idx, neurons in self.register_neurons.items():
            if layer_idx >= len(layers):
                print(f"Warning: Layer {layer_idx} >= num_layers {len(layers)}, skipping")
                continue

            if not neurons:
                continue

            layer = layers[layer_idx]

            # Hook on fc2 output (MLP output, hidden_size dimension)
            if not hasattr(layer.mlp, 'fc2'):
                print(f"Warning: Layer {layer_idx} MLP has no 'fc2' attribute, skipping")
                continue

            # Create the intervention hook for this layer
            # Hook on fc2 output (MLP output)
            hook = layer.mlp.fc2.register_forward_hook(
                self._make_intervention_hook(layer_idx, neurons)
            )
            self.intervention_hooks.append(hook)
            print(f"Registered intervention hook on layer {layer_idx}.mlp.fc2 (output) for neurons {neurons}")

    def _remove_hooks(self):
        """Remove all intervention hooks."""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks.clear()

    def _make_intervention_hook(self, layer_idx: int, neurons: List[int]):
        """
        Create a forward hook function for register neuron intervention.

        ABLATION VERSION: The hook modifies the MLP output (after fc2):
        1. For each register neuron, find sign_max across ALL tokens
        2. Set register token activations to this sign_max value
        3. Zero out image patch activations

        The intervention happens at the MLP output (hidden_size dimension),
        which is after fc2 in the MLP block.

        Args:
            layer_idx: Layer index for logging/debugging
            neurons: List of neuron indices to intervene on (indices in 768-dim space)

        Returns:
            A hook function that modifies MLP output activations
        """

        def hook(module, input, output):
            """
            Hook function applied to mlp.fc2 output (MLP output).

            Args:
                module: The mlp.fc2 module (Linear layer)
                input: Input tuple (before fc2)
                output: Output tensor (batch, seq_len, hidden_size)

            Returns:
                The modified output tensor
            """
            # output is (batch, seq_len, hidden_size) where hidden_size = 768
            batch_size, seq_len, hidden_size = output.shape

            # Create a modified output tensor
            modified_output = output.clone()

            # For each register neuron, apply intervention
            for neuron_idx in neurons:
                if neuron_idx >= hidden_size:
                    continue

                # Get activations for this neuron across ALL tokens
                all_token_activations = output[0, :, neuron_idx]  # (seq_len,)

                # Find sign_max (value with largest absolute value)
                pos_max = torch.max(all_token_activations)
                neg_max = torch.min(all_token_activations)
                sign_max_value = pos_max if abs(pos_max) > abs(neg_max) else neg_max

                # Set register activations to sign_max (same value for all registers)
                # Register tokens are at the end: [-num_registers:]
                modified_output[0, -self.num_registers:, neuron_idx] = sign_max_value

                # Zero out image patch activations
                # Image patches are indices [1 : -num_registers]
                # Index 0 is CLS, indices 1:-num_registers are patches, -num_registers: are registers
                modified_output[0, 1:-self.num_registers, neuron_idx] = 0

            # Copy modified values back
            with torch.no_grad():
                output.copy_(modified_output)

            return output

        return hook

    def forward(self, pixel_values, output_attentions=False):
        """
        Forward pass with register tokens and neuron intervention.

        Args:
            pixel_values: Input tensor (batch, channels, height, width)
            output_attentions: Whether to return attention weights

        Returns:
            Model outputs with modified attention patterns due to registers
        """
        # Get embeddings (CLS + patches with positional encoding)
        embeddings = self.base_model.vit.embeddings(pixel_values)
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        dtype = pixel_values.dtype
        hidden_size = self.base_model.config.hidden_size

        # Append zero-initialized registers (NO positional encoding)
        registers = torch.zeros(
            batch_size, self.num_registers, hidden_size,
            device=device, dtype=dtype
        )
        embeddings = torch.cat([embeddings, registers], dim=1)

        # Forward through encoder/layers (handle different ViT implementations)
        if hasattr(self.base_model.vit, 'encoder'):
            encoder_outputs = self.base_model.vit.encoder(
                embeddings,
                output_attentions=output_attentions,
            )
        else:
            # Direct layers structure (no separate encoder module)
            # The intervention hooks will be applied automatically during forward pass
            hidden_states = embeddings
            all_attentions = () if output_attentions else None

            for layer_idx, layer in enumerate(self.base_model.vit.layers):
                if output_attentions:
                    # Self Attention part
                    residual = hidden_states
                    hidden_states = layer.layernorm_before(hidden_states)
                    attention_output = layer.attention(hidden_states, None)
                    hidden_states = attention_output[0]
                    attention_weights = attention_output[1]
                    all_attentions = all_attentions + (attention_weights,)
                    hidden_states = layer.dropout(hidden_states)
                    hidden_states = hidden_states + residual

                    # Fully Connected part
                    # NOTE: Intervention hooks on mlp.fc2 will be triggered here
                    residual = hidden_states
                    hidden_states = layer.layernorm_after(hidden_states)
                    hidden_states = layer.mlp(hidden_states)  # Hooks trigger inside
                    hidden_states = layer.dropout(hidden_states)
                    hidden_states = hidden_states + residual
                else:
                    # Standard layer forward
                    # NOTE: Intervention hooks on mlp.fc2 will be triggered here
                    layer_output = layer(hidden_states)
                    hidden_states = layer_output

            # Apply final layernorm
            sequence_output = self.base_model.vit.layernorm(hidden_states)

            # Classification uses only CLS token (index 0)
            logits = self.base_model.classifier(sequence_output[:, 0, :])

            if not output_attentions:
                return type("Outputs", (), {"logits": logits})()

            return type("Outputs", (), {
                "logits": logits,
                "attentions": all_attentions,
                "hidden_states": None,
            })()

        # Layer norm and head
        sequence_output = self.base_model.vit.layernorm(encoder_outputs.last_hidden_state)

        # Classification uses only CLS token (index 0)
        logits = self.base_model.classifier(sequence_output[:, 0, :])

        # Return outputs in expected format
        if not output_attentions:
            return type("Outputs", (), {"logits": logits})()

        return type("Outputs", (), {
            "logits": logits,
            "attentions": encoder_outputs.attentions,
            "hidden_states": encoder_outputs.hidden_states,
        })()

    def __del__(self):
        """Clean up hooks when the object is destroyed."""
        self._remove_hooks()


def load_model_with_registers(
    model_path: str,
    register_neurons_path: str,
    num_registers: int = 1,
    device: str = "cuda",
) -> ViTWithTestTimeRegisters:
    """
    Load a ViT model and wrap it with test-time registers.

    Args:
        model_path: Path to the pre-trained ViT model
        register_neurons_path: Path to JSON file with discovered register neurons
        num_registers: Number of register tokens to append (default: 1, following paper)
        device: Device to load model on

    Returns:
        Wrapped ViTWithTestTimeRegisters model
    """
    import json
    from pathlib import Path

    # Load base model
    print(f"Loading base model from {model_path}...")
    base_model = ViTForImageClassification.from_pretrained(
        model_path,
        attn_implementation="eager",
    )
    base_model.to(device)
    base_model.eval()

    # Load register neurons
    neurons_path = Path(register_neurons_path)
    if not neurons_path.exists():
        raise FileNotFoundError(f"Register neurons file not found: {neurons_path}")

    with open(neurons_path, "r") as f:
        register_neurons_json = json.load(f)

    # Convert string keys to int
    register_neurons = {}
    for layer_str, neurons in register_neurons_json.items():
        register_neurons[int(layer_str)] = neurons

    print(f"Loaded register neurons for {len(register_neurons)} layers")

    # Wrap model
    model = ViTWithTestTimeRegisters(
        base_model=base_model,
        num_registers=num_registers,
        register_neurons=register_neurons,
    )

    return model


if __name__ == "__main__":
    # Test the wrapper
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--register_neurons", type=str, required=True)
    parser.add_argument("--num_registers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_with_registers(
        model_path=args.model_path,
        register_neurons_path=args.register_neurons,
        num_registers=args.num_registers,
        device=device,
    )

    print(f"Loaded model with {args.num_registers} registers")
    print(f"Register neurons: {model.register_neurons}")
