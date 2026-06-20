#!/usr/bin/env python3
"""
Complete 4-Phase Register Neuron Discovery Pipeline

Based on "Vision Transformers Don't Need Trained Registers" (Jiang et al., 2025).

This script implements the complete discovery pipeline:
1. Phase 1: Norm Tracking - Find the "explosion point" layer
2. Phase 2: Extreme Patches - Identify outlier positions at target layer
3. Phase 3: Backward Search - Find register neurons in layers before explosion
4. Phase 4: Verification - Apply test-time intervention (optional)

Key features:
- Reproducible sampling with fixed random seed
- Balanced sampling across all emotion classes
- Total image count (not per-class)
- Interactive layer selection based on norm progression plot
- Comprehensive visualizations for each phase
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def is_image_file(path: Path) -> bool:
    """Check if file is a valid image."""
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return path.is_file() and path.suffix.lower() in valid_suffixes


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iter_images_balanced_total(root_dir: Path, total_images: int, seed: int = None):
    """
    Sample images evenly from each class subdirectory.

    Args:
        root_dir: Directory containing class subdirectories
        total_images: Total number of images to sample (across all classes)
        seed: Random seed for reproducibility

    Yields:
        Paths to images sampled evenly from each class
    """
    if seed is not None:
        np.random.seed(seed)

    # Find all class directories
    class_dirs = {}
    for class_dir in sorted(root_dir.iterdir()):
        if class_dir.is_dir():
            images = [p for p in class_dir.iterdir() if is_image_file(p)]
            if images:
                class_dirs[class_dir.name] = sorted(images)

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {root_dir}")

    num_classes = len(class_dirs)
    images_per_class = max(1, total_images // num_classes)
    remaining = total_images - (images_per_class * num_classes)

    print(f"Sampling {total_images} total images from {num_classes} classes")
    print(f"  Base: {images_per_class} images per class")
    print(f"  Additional: {remaining} classes get +1 image")

    # Shuffle image lists with seed
    rng = np.random.default_rng(seed)
    for class_name in class_dirs:
        rng.shuffle(class_dirs[class_name])

    # Sample evenly from each class
    class_names = sorted(class_dirs.keys())
    for i, class_name in enumerate(class_names):
        images = class_dirs[class_name]
        # First 'remaining' classes get one extra image
        sample_size = images_per_class + (1 if i < remaining else 0)
        sample_size = min(sample_size, len(images))
        sampled = images[:sample_size]

        print(f"  {class_name}: {len(sampled)}/{len(images)} images")
        for img_path in sampled:
            yield img_path


def compute_patch_norms(hidden_states, exclude_cls: bool = True):
    """
    Compute L2 norm of each patch token in hidden states.

    Args:
        hidden_states: (batch, seq_len, hidden_size)
        exclude_cls: If True, exclude CLS token (index 0)

    Returns:
        norms: (batch, num_patches) L2 norms
    """
    norms = torch.norm(hidden_states, p=2, dim=-1)
    if exclude_cls:
        norms = norms[:, 1:]  # Remove CLS token
    return norms


# ==============================================================================
# PHASE 1: Norm Tracking - Find the "Explosion Point"
# ==============================================================================

class NormTracker:
    """Track patch norms across all ViT layers to identify explosion point."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.layer_norms = defaultdict(list)
        self.hooks = []

    def _create_hooks(self):
        """Create forward hooks for all layers."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Compute norms of patch tokens (exclude CLS)
                norms = compute_patch_norms(output, exclude_cls=True)
                self.layer_norms[layer_idx].append(norms.detach().cpu())
            return hook

        # Handle different ViT structures
        if hasattr(self.model.vit, 'encoder'):
            layers = self.model.vit.encoder.layer
        elif hasattr(self.model.vit, 'layers'):
            layers = self.model.vit.layers
        else:
            raise ValueError("Cannot find ViT layers")

        for layer_idx, layer in enumerate(layers):
            hook = layer.mlp.fc2.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)

    def track(self, image_generator, processor):
        """Track norms for all images."""
        self.model.eval()
        with torch.no_grad():
            for img_path in tqdm(image_generator, desc="Phase 1: Tracking norms"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = self.model(**inputs)
                except Exception as e:
                    print(f"Warning: Failed to process {img_path.name}: {e}")
                    continue

    def get_aggregated_stats(self):
        """Aggregate statistics across all images."""
        aggregated = {}
        for layer_idx, norms_list in self.layer_norms.items():
            if not norms_list:
                continue

            # Stack all norms: (total_images, num_patches)
            all_norms = torch.cat(norms_list, dim=0).numpy()

            # Compute statistics
            max_norms_per_image = all_norms.max(axis=1)
            mean_norms_per_image = all_norms.mean(axis=1)
            std_norms_per_image = all_norms.std(axis=1)

            aggregated[layer_idx] = {
                'mean_max': float(np.mean(max_norms_per_image)),
                'max_max': float(np.max(all_norms)),
                'min_max': float(np.min(max_norms_per_image)),
                'std_max': float(np.std(max_norms_per_image)),
                'mean_mean': float(np.mean(mean_norms_per_image)),
                'mean_std': float(np.mean(std_norms_per_image)),
                'num_images': len(norms_list),
            }
        return aggregated

    def plot_norm_progression(self, output_path):
        """Plot norm progression across layers."""
        stats = self.get_aggregated_stats()

        layers = sorted(stats.keys())
        mean_max = [stats[l]['mean_max'] for l in layers]
        max_max = [stats[l]['max_max'] for l in layers]
        min_max = [stats[l]['min_max'] for l in layers]
        mean_mean = [stats[l]['mean_mean'] for l in layers]
        mean_plus_3std = [stats[l]['mean_mean'] + 3*stats[l]['mean_std'] for l in layers]

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot different metrics
        ax.plot(layers, max_max, 'r-', linewidth=2.5, label='Max Norm', marker='o', markersize=6)
        ax.plot(layers, mean_max, 'b-', linewidth=2.5, label='Mean of Max Norms', marker='s', markersize=6)
        ax.fill_between(layers, min_max, max_max, alpha=0.15, color='red', label='Min-Max Range')
        ax.plot(layers, mean_mean, 'g-', linewidth=2, label='Mean Patch Norm', marker='^', markersize=6)
        ax.plot(layers, mean_plus_3std, 'm--', linewidth=2, label='Mean + 3×Std', marker='x', markersize=6)

        ax.set_xlabel('Layer Index', fontsize=13)
        ax.set_ylabel('L2 Norm', fontsize=13)
        ax.set_title('Patch Norm Progression Across Layers\n(Identifying the "Explosion Point")',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticks(layers)

        # Add annotations for key observations
        # Check for significant jumps
        for i in range(1, len(layers)):
            if mean_max[i] - mean_max[i-1] > 5:  # Significant jump
                ax.annotate(f'Jump: L{layers[i]}',
                           xy=(layers[i], mean_max[i]),
                           xytext=(layers[i], mean_max[i] + 5),
                           fontsize=10, color='darkred',
                           arrowprops=dict(arrowstyle='->', color='darkred'))

        # Check for drops in final layer
        if len(layers) >= 2 and max_max[-1] < max_max[-2]:
            ax.annotate(f'Norm Drop',
                       xy=(layers[-1], max_max[-1]),
                       xytext=(layers[-1], max_max[-1] - 8),
                       fontsize=10, color='darkorange',
                       arrowprops=dict(arrowstyle='->', color='darkorange'))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ==============================================================================
# PHASE 2: Extreme Patches - Find Outlier Positions
# ==============================================================================

class ExtremePatchFinder:
    """Find outlier patches at a specific layer."""

    def __init__(self, model, layer_idx, threshold=None, device='cuda'):
        self.model = model
        self.layer_idx = layer_idx
        self.threshold = threshold  # None means use mean + 3*std
        self.device = device
        self.all_norms = []
        self.outlier_positions = defaultdict(int)
        self.patch_grid_size = 14  # For 224x224 images with 16x16 patches
        self.mean_norm = None
        self.std_norm = None

    def find_outliers(self, image_generator, processor):
        """Identify outlier patches at the specified layer."""
        self.model.eval()
        layer_output = None

        def hook(module, input, output):
            nonlocal layer_output
            layer_output = output.detach()

        # Register hook on the target layer
        if hasattr(self.model.vit, 'encoder'):
            layer = self.model.vit.encoder.layer[self.layer_idx]
        else:
            layer = self.model.vit.layers[self.layer_idx]

        handle = layer.mlp.fc2.register_forward_hook(hook)

        with torch.no_grad():
            for img_path in tqdm(image_generator, desc=f"Phase 2: Analyzing layer {self.layer_idx}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    _ = self.model(**inputs)

                    if layer_output is not None:
                        norms = compute_patch_norms(layer_output, exclude_cls=True)
                        self.all_norms.append(norms.cpu())
                except Exception as e:
                    print(f"Warning: Failed to process {img_path.name}: {e}")
                    continue

        handle.remove()

        if not self.all_norms:
            raise ValueError(f"No norms collected for layer {self.layer_idx}")

        # Calculate statistics
        all_norms_flat = torch.cat(self.all_norms).flatten().numpy()
        self.mean_norm = np.mean(all_norms_flat)
        self.std_norm = np.std(all_norms_flat)

        # Use user-specified threshold or calculate mean + 3*std
        if self.threshold is None:
            self.threshold = self.mean_norm + 3 * self.std_norm
            threshold_source = f"mean + 3*std = {self.mean_norm:.2f} + 3*{self.std_norm:.2f}"
        else:
            threshold_source = f"user-specified = {self.threshold}"

        print(f"  Threshold: {self.threshold:.2f} ({threshold_source})")

        # Identify outlier positions
        outlier_count = 0
        total_count = 0

        for norms in self.all_norms:
            outlier_mask = norms > self.threshold
            outlier_count += outlier_mask.sum().item()
            total_count += norms.numel()

            # Track outlier positions
            indices = torch.where(outlier_mask)
            for batch_idx, patch_idx in zip(indices[0], indices[1]):
                row = (patch_idx // self.patch_grid_size).item()
                col = (patch_idx % self.patch_grid_size).item()
                self.outlier_positions[(row, col)] += 1

        self.outlier_count = outlier_count
        self.total_count = total_count

    def plot_norm_histogram(self, output_path):
        """Plot histogram of patch norms with threshold marked."""
        all_norms_flat = torch.cat(self.all_norms).flatten().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Linear scale histogram
        n, bins, patches = ax1.hist(all_norms_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(self.threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Threshold = {self.threshold:.1f}')
        ax1.axvline(self.mean_norm, color='green', linestyle='-', linewidth=2,
                   label=f'Mean = {self.mean_norm:.1f}')
        ax1.set_xlabel('Patch L2 Norm', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Patch Norm Distribution (Layer {self.layer_idx})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # Fine-grained x-axis ticks (every 10 units)
        x_min = np.min(all_norms_flat)
        x_max = np.max(all_norms_flat)
        # Round to nearest 10
        x_min_tick = int(np.floor(x_min / 10) * 10)
        x_max_tick = int(np.ceil(x_max / 10) * 10)
        x_ticks = list(range(x_min_tick, x_max_tick + 1, 10))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks, rotation=45, ha='right', fontsize=9)

        # Add statistics text
        outlier_pct = (self.outlier_count / self.total_count) * 100
        stats_text = f'Total patches: {self.total_count:,}\n'
        stats_text += f'Mean: {self.mean_norm:.2f}, Std: {self.std_norm:.2f}\n'
        stats_text += f'Max: {np.max(all_norms_flat):.2f}\n'
        stats_text += f'Outliers: {self.outlier_count:,} ({outlier_pct:.2f}%)'

        ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Log scale histogram
        ax2.hist(all_norms_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(self.threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Threshold = {self.threshold:.1f}')
        ax2.set_xlabel('Patch L2 Norm', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_title(f'Patch Norm Distribution (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which='both')
        # Fine-grained x-axis ticks for log scale too
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_ticks, rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_spatial_heatmap(self, output_path):
        """Plot spatial distribution of outliers."""
        grid = np.zeros((self.patch_grid_size, self.patch_grid_size))

        for (row, col), count in self.outlier_positions.items():
            if row < self.patch_grid_size and col < self.patch_grid_size:
                grid[row, col] = count

        num_images = len(self.all_norms)
        grid_normalized = grid / num_images

        fig, ax = plt.subplots(figsize=(11, 9))

        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', aspect='equal')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Outlier Count', fontsize=12)

        # Add grid lines
        ax.set_xticks(np.arange(self.patch_grid_size))
        ax.set_yticks(np.arange(self.patch_grid_size))
        ax.set_xticklabels(np.arange(self.patch_grid_size), fontsize=8)
        ax.set_yticklabels(np.arange(self.patch_grid_size), fontsize=8)
        ax.set_xlabel('Patch Column', fontsize=12)
        ax.set_ylabel('Patch Row', fontsize=12)
        ax.set_title(f'Spatial Distribution of Outlier Patches (Layer {self.layer_idx})\n' +
                    f'Threshold = {self.threshold:.2f}, {len(self.outlier_positions)} positions',
                    fontsize=14, fontweight='bold')

        # Add light grid
        ax.set_xticks(np.arange(self.patch_grid_size) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.patch_grid_size) - 0.5, minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Mark top outlier positions
        if len(self.outlier_positions) > 0:
            flat_grid = grid.flatten()
            top_n = min(10, len(self.outlier_positions))
            top_indices = np.argsort(flat_grid)[-top_n:]
            for idx in top_indices:
                row, col = divmod(idx, self.patch_grid_size)
                if grid[row, col] > 0:
                    ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                               fill=False, edgecolor='blue', linewidth=2.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# ==============================================================================
# PHASE 3: Register Neurons - Backward Search
# ==============================================================================

class RegisterNeuronFinder:
    """
    Find register neurons through backward search.

    ABLATION VERSION: We discover neurons at the MLP OUTPUT (after fc2),
    not the intermediate activation. This is for comparison with the middle-activation approach.
    """

    def __init__(self, model, explosion_layer, top_k=8, device='cuda'):
        self.model = model
        self.explosion_layer = explosion_layer
        self.top_k = top_k
        self.device = device
        self.neuron_activations = defaultdict(list)

    def find_neurons(self, image_generator, processor, outlier_positions):
        """
        Search for register neurons in layers before explosion.

        ABLATION VERSION: We hook on fc2 output (MLP output)
        to discover neurons in the hidden_size dimensional space (768).
        """
        self.model.eval()

        # Convert outlier positions to flat indices
        outlier_indices = set()
        for row, col in outlier_positions:
            outlier_indices.add(row * 14 + col)  # 14x14 grid

        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                batch, seq_len, hidden_size = output.shape
                # hidden_size is hidden_size = 768 (MLP output after fc2)
                for b in range(batch):
                    for patch_idx in outlier_indices:
                        if patch_idx < seq_len - 1:  # Ensure valid index
                            # patch_idx maps to actual token index (patch_idx + 1 to skip CLS)
                            neuron_vals = output[b, patch_idx + 1, :].detach().cpu()
                            self.neuron_activations[layer_idx].append(neuron_vals)
            return hook

        # Register hooks on layers before explosion
        if hasattr(self.model.vit, 'encoder'):
            layers = self.model.vit.encoder.layer
        else:
            layers = self.model.vit.layers

        print(f"  Searching layers 0 to {self.explosion_layer - 1} (MLP output after fc2)")
        for layer_idx in range(self.explosion_layer):
            layer = layers[layer_idx]
            # Hook on fc2 output (MLP output)
            # This is the ablation version: discovering register neurons at MLP output
            if not hasattr(layer.mlp, 'fc2'):
                print(f"Warning: Layer {layer_idx} has no fc2, skipping")
                continue
            hook = layer.mlp.fc2.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        with torch.no_grad():
            for img_path in tqdm(image_generator, desc="Phase 3: Finding register neurons"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = self.model(**inputs)
                except Exception as e:
                    print(f"Warning: Failed to process {img_path.name}: {e}")
                    continue

        for hook in hooks:
            hook.remove()

    def aggregate_and_rank(self):
        """
        Rank neurons by mean activation.

        ABLATION VERSION: Returns register neurons with indices in the MLP output space
        (hidden_size = 768 for ViT-B/16), not the intermediate activation.
        """
        register_neurons = {}
        neuron_stats = {}

        for layer_idx, activations_list in self.neuron_activations.items():
            if not activations_list:
                continue

            # Stack all activations: (num_samples, hidden_size)
            # hidden_size = 768 for ViT-B/16 (MLP output after fc2)
            all_activations = torch.stack(activations_list, dim=0)

            print(f"  Layer {layer_idx}: activation shape = {all_activations.shape} (should have hidden_size = 768)")

            # Compute mean activation for each neuron (use absolute values)
            mean_activations = all_activations.abs().mean(dim=0)

            # Get top-k neurons
            top_k = min(self.top_k, len(mean_activations))
            top_neurons = mean_activations.topk(top_k)

            register_neurons[layer_idx] = top_neurons.indices.tolist()
            neuron_stats[layer_idx] = {
                'top_indices': top_neurons.indices.tolist(),
                'top_values': top_neurons.values.tolist(),
                'num_samples': len(activations_list),
                'hidden_size': all_activations.shape[-1],
            }

        return register_neurons, neuron_stats

    def plot_neuron_activation_histogram(self, output_path):
        """Plot histograms of neuron activations."""
        num_layers = len(self.neuron_activations)

        if num_layers == 0:
            print("  Warning: No neuron activations to plot")
            return

        # Aggregate all layer activations
        all_layer_means = []
        layer_labels = []

        for layer_idx in sorted(self.neuron_activations.keys()):
            activations = torch.stack(self.neuron_activations[layer_idx], dim=0)
            mean_acts = activations.abs().mean(dim=0).numpy()
            all_layer_means.append(mean_acts)
            layer_labels.extend([f'L{layer_idx}'] * len(mean_acts))

        all_means = np.concatenate(all_layer_means)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Linear scale
        ax1.hist(all_means, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Mean Neuron Activation at Outlier Positions', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Neuron Activation Distribution (All Searched Layers)', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Add statistics
        ax1.axvline(np.mean(all_means), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(all_means):.2f}')
        ax1.axvline(np.median(all_means), color='green', linestyle='--', linewidth=2,
                   label=f'Median = {np.median(all_means):.2f}')
        ax1.legend(fontsize=11)

        # Log scale
        ax2.hist(all_means, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Mean Neuron Activation at Outlier Positions', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_top_neurons(self, output_path, neuron_stats):
        """Plot bar chart of top neurons and their activation values."""
        if not neuron_stats:
            print("  Warning: No neuron stats to plot")
            return

        # Collect all top neurons
        all_data = []
        for layer_idx in sorted(neuron_stats.keys()):
            stats = neuron_stats[layer_idx]
            for val, idx in zip(stats['top_values'], stats['top_indices']):
                all_data.append({
                    'layer': layer_idx,
                    'neuron': idx,
                    'value': val,
                    'label': f'L{layer_idx}N{idx}'
                })

        # Sort by value
        all_data.sort(key=lambda x: x['value'], reverse=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        labels = [d['label'] for d in all_data]
        values = [d['value'] for d in all_data]

        # Color by layer
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))

        bars = ax.bar(range(len(all_data)), values, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(range(len(all_data)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Activation at Outlier Positions', fontsize=12)
        ax.set_title(f'Top {len(all_data)} Register Neurons by Activation Strength',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete 4-phase register neuron discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 1000 images, interactive layer selection
  python run_complete_discovery.py --image_folder data/fer2013/train --num_images 1000

  # Run with specific layer (skip interactive prompt)
  python run_complete_discovery.py --image_folder data/fer2013/train --num_images 1000 --phase2_layer 11

  # Run with different seed for different sample
  python run_complete_discovery.py --image_folder data/fer2013/train --num_images 500 --seed 123
        """
    )

    # Model and data arguments
    parser.add_argument("--model_path", type=str, default="models/emotion_vit",
                       help="Path to ViT model directory")
    parser.add_argument("--image_folder", type=str, required=True,
                       help="Folder containing training images (with class subdirectories)")

    # Sampling arguments
    parser.add_argument("--num_images", type=int, default=1000,
                       help="Total number of images to sample (balanced across classes)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    # Phase 2 arguments
    parser.add_argument("--phase2_layer", type=int, default=None,
                       help="Layer for Phase 2 analysis (0-11). Where to find outlier positions. Typically last layer (11) or second-to-last (10).")
    parser.add_argument("--explosion_layer", type=int, default=None,
                       help="Explosion layer (0-11). Where norms first spike significantly. Register neurons will be searched in layers 0 to explosion_layer-1.")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Outlier threshold for Phase 2. If not specified, uses mean + 3*std (recommended to inspect histogram first).")

    # Phase 3 arguments
    parser.add_argument("--top_k", type=int, default=8,
                       help="Number of top neurons to select per layer")
    parser.add_argument("--top_k_overall", type=int, default=None,
                       help="After Phase 3, select top-K neurons across ALL layers (not per-layer)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/discovery",
                       help="Output directory for all results")
    parser.add_argument("--output", type=str, default="models/register_neurons.json",
                       help="Final output path for register neurons (default: phase3 dir, or models/register_neurons.json if top_k_overall is set)")

    args = parser.parse_args()

    # Validate basic arguments
    if args.num_images < 7:
        print("Warning: num_images should be at least 7 for balanced sampling across 7 classes")

    # Set random seed
    set_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    phase1_dir = output_dir / "phase1_norm_tracking"
    phase2_dir = output_dir / "phase2_extreme_patches"
    phase3_dir = output_dir / "phase3_register_neurons"

    for d in [phase1_dir, phase2_dir, phase3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGISTER NEURON DISCOVERY PIPELINE")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Images: {args.image_folder}")
    print(f"Total images: {args.num_images}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model on {device}...")

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_path)
        model = ViTForImageClassification.from_pretrained(
            args.model_path,
            attn_implementation="eager",
        )
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model exists at {args.model_path}")
        return 1

    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        print(f"Error: Image folder not found: {image_folder}")
        return 1

    # Get total number of layers
    if hasattr(model.vit, 'encoder'):
        num_layers = len(model.vit.encoder.layer)
    else:
        num_layers = len(model.vit.layers)

    print(f"Model has {num_layers} layers (indices 0-{num_layers-1})")

    # Validate layer arguments (now that we know num_layers)
    if args.phase2_layer is not None and not (0 <= args.phase2_layer < num_layers):
        print(f"Error: phase2_layer must be between 0 and {num_layers-1}")
        return 1

    if args.explosion_layer is not None and not (0 <= args.explosion_layer < num_layers):
        print(f"Error: explosion_layer must be between 0 and {num_layers-1}")
        return 1

    # ==============================================================================
    # PHASE 1: Norm Tracking
    # ==============================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Norm Tracking Across All Layers")
    print("=" * 70)

    tracker = NormTracker(model, device)
    tracker._create_hooks()

    # Get image list once for consistency
    image_list = list(iter_images_balanced_total(image_folder, args.num_images, args.seed))
    print(f"\nProcessing {len(image_list)} images...")

    tracker.track(iter(image_list), processor)
    tracker.remove_hooks()

    # Save Phase 1 results
    stats = tracker.get_aggregated_stats()

    stats_path = phase1_dir / "norm_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    plot_path = phase1_dir / "norm_progression.png"
    tracker.plot_norm_progression(plot_path)

    print(f"\n✓ Phase 1 complete:")
    print(f"  Statistics: {stats_path}")
    print(f"  Plot: {plot_path}")

    # Print summary table (skip metadata entry)
    print(f"\n{'Layer':>6} | {'Mean Max':>10} | {'Max Max':>10} | {'Mean+3Std':>10}")
    print("-" * 46)
    for layer in sorted(stats.keys()):
        s = stats[layer]
        print(f"{layer:>6} | {s['mean_max']:>10.2f} | {s['max_max']:>10.2f} | "
              f"{s['mean_mean']+3*s['mean_std']:>10.2f}")

    # ==============================================================================
    # INTERACTIVE PAUSE FOR LAYER SELECTION
    # ==============================================================================
    if args.phase2_layer is None or args.explosion_layer is None:
        print("\n" + "=" * 70)
        print("LAYER SELECTION")
        print("=" * 70)
        print(f"\nPlease inspect the norm progression plot:")
        print(f"  {plot_path.absolute()}")
        print("\n" + "-" * 70)
        print("PHASE 2 LAYER (where to find outlier positions):")
        print("  - Typically the last layer (11) or second-to-last layer (10)")
        print("  - Use last layer if norms continue rising")
        print("  - Use second-to-last if norms drop in final layer")
        print("-" * 70)
        print("EXPLOSION LAYER (where norms first spike):")
        print("  - Look at the norm progression plot")
        print("  - Find the layer where mean_max shows first significant increase")
        print("  - This is where outliers emerge - search for neurons BEFORE this")
        print("-" * 70)
        print("REGISTER NEURON SEARCH RANGE:")
        print("  - Neurons will be searched in layers 0 to explosion_layer-1")
        print("  - Outlier positions from phase2_layer are used for measurement")
        print("=" * 70)

    # Get Phase 2 layer
    if args.phase2_layer is None:
        while True:
            user_input = input("\nEnter Phase 2 layer (typically 10 or 11, or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                print("Exiting.")
                return 0
            try:
                phase2_layer = int(user_input)
                if 0 <= phase2_layer < num_layers:
                    break
                else:
                    print(f"Layer must be between 0 and {num_layers-1}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    else:
        phase2_layer = args.phase2_layer

    # Get explosion layer
    if args.explosion_layer is None:
        while True:
            user_input = input("\nEnter explosion layer (where norms first spike, or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                print("Exiting.")
                return 0
            try:
                explosion_layer = int(user_input)
                if 0 <= explosion_layer < num_layers:
                    break
                else:
                    print(f"Layer must be between 0 and {num_layers-1}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    else:
        explosion_layer = args.explosion_layer

    print(f"\nConfiguration:")
    print(f"  Phase 2 layer (outlier positions): {phase2_layer}")
    print(f"  Explosion layer: {explosion_layer}")
    print(f"  Register neuron search: layers 0 to {explosion_layer - 1}")

    if explosion_layer > phase2_layer:
        print("\n⚠️  Warning: explosion_layer > phase2_layer")
        print("   This is unusual - typically explosion happens before phase2_layer")

    # ==============================================================================
    # PHASE 2: Extreme Patches
    # ==============================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: Finding Extreme Patches at Layer {phase2_layer}")
    print("=" * 70)

    finder = ExtremePatchFinder(model, phase2_layer, threshold=args.threshold, device=device)
    finder.find_outliers(iter_images_balanced_total(image_folder, args.num_images, args.seed), processor)

    # Generate plots
    hist_path = phase2_dir / "patch_norm_histogram.png"
    finder.plot_norm_histogram(hist_path)

    heatmap_path = phase2_dir / "spatial_outlier_heatmap.png"
    finder.plot_spatial_heatmap(heatmap_path)

    # Save outlier positions
    outlier_positions_list = [(int(r), int(c)) for (r, c) in finder.outlier_positions.keys()]

    # Determine threshold source for documentation
    if args.threshold is None:
        threshold_source = 'mean + 3*std (auto-calculated)'
    else:
        threshold_source = 'user-specified'

    outlier_data = {
        'layer': phase2_layer,
        'threshold': float(finder.threshold),
        'threshold_source': threshold_source,
        'mean_norm': float(finder.mean_norm),
        'std_norm': float(finder.std_norm),
        'outlier_count': finder.outlier_count,
        'total_count': finder.total_count,
        'outlier_percentage': 100 * finder.outlier_count / finder.total_count,
        'num_positions': len(finder.outlier_positions),
        'outlier_positions': [f"{r},{c}" for r, c in outlier_positions_list]
    }
    outlier_path = phase2_dir / "outlier_positions.json"
    with open(outlier_path, 'w') as f:
        json.dump(outlier_data, f, indent=2)

    print(f"\n✓ Phase 2 complete:")
    if args.threshold is None:
        print(f"  Threshold: {finder.threshold:.2f} (mean + 3*std, auto-calculated)")
    else:
        print(f"  Threshold: {finder.threshold:.2f} (user-specified)")
    print(f"  Outliers: {finder.outlier_count:,} / {finder.total_count:,} "
          f"({100 * finder.outlier_count / finder.total_count:.2f}%)")
    print(f"  Unique positions: {len(finder.outlier_positions)}")
    print(f"\n  Files:")
    print(f"    Histogram: {hist_path}")
    print(f"    Heatmap: {heatmap_path}")
    print(f"    Positions: {outlier_path}")

    # ==============================================================================
    # PHASE 3: Register Neurons
    # ==============================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Finding Register Neurons (Backward Search)")
    print("=" * 70)
    print(f"\n🔍 Backward search configuration:")
    print(f"   - Phase 2 analysis layer: {phase2_layer} (where outlier positions were found)")
    print(f"   - Explosion layer: {explosion_layer} (where norms first spiked)")
    print(f"   - Searching layers: 0 to {explosion_layer - 1} (before explosion point)")
    print(f"   - Using outlier positions from layer {phase2_layer}")

    neuron_finder = RegisterNeuronFinder(model, explosion_layer, args.top_k, device)
    neuron_finder.find_neurons(
        iter_images_balanced_total(image_folder, args.num_images, args.seed),
        processor,
        set(outlier_positions_list)
    )

    register_neurons, neuron_stats = neuron_finder.aggregate_and_rank()

    # Generate visualizations
    neuron_hist_path = phase3_dir / "neuron_activation_histogram.png"
    neuron_finder.plot_neuron_activation_histogram(neuron_hist_path)

    top_neurons_path = phase3_dir / "top_neurons.png"
    neuron_finder.plot_top_neurons(top_neurons_path, neuron_stats)

    # Save results
    results = {str(k): v for k, v in register_neurons.items()}
    results_path = phase3_dir / "register_neurons.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save neuron stats
    stats_path = phase3_dir / "neuron_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump({str(k): v for k, v in neuron_stats.items()}, f, indent=2)

    print(f"\n✓ Phase 3 complete:")
    print(f"\nRegister neurons discovered:")
    total_neurons = 0
    for layer in sorted(register_neurons.keys()):
        neurons = register_neurons[layer]
        total_neurons += len(neurons)
        values = neuron_stats[layer]['top_values']
        print(f"  Layer {layer:2d}: {len(neurons)} neurons")
        print(f"    Indices: {neurons}")
        print(f"    Values: {[f'{v:.2f}' for v in values]}")

    print(f"\n  Total neurons: {total_neurons}")
    print(f"\n  Files:")
    print(f"    Register neurons: {results_path}")
    print(f"    Neuron stats: {stats_path}")
    print(f"    Activation histogram: {neuron_hist_path}")
    print(f"    Top neurons plot: {top_neurons_path}")

    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    print("\n" + "=" * 70)
    print("DISCOVERY PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  Phase 1:")
    print(f"    - norm_progression.png")
    print(f"    - norm_statistics.json")
    print(f"  Phase 2:")
    print(f"    - patch_norm_histogram.png")
    print(f"    - spatial_outlier_heatmap.png")
    print(f"    - outlier_positions.json")
    print(f"  Phase 3:")
    print(f"    - register_neurons.json")
    print(f"    - neuron_statistics.json")
    print(f"    - neuron_activation_histogram.png")
    print(f"    - top_neurons.png")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. To use these register neurons for test-time intervention:")
    print(f"   python src/gradio_app.py --register_neurons {results_path}")
    print("\n2. Or update the main register neurons file:")
    print(f"   cp {results_path} models/register_neurons.json")
    print("\n3. Verify reproducibility by running again with same seed:")
    print(f"   python src/run_complete_discovery.py --image_folder {args.image_folder} \\")
    print(f"       --num_images {args.num_images} --seed {args.seed} \\")
    print(f"       --phase2_layer {phase2_layer}")

    print("\n" + "=" * 70)
    print("KEY DISTINCTIONS")
    print("=" * 70)
    print(f"\n📍 Phase 2 Layer ({phase2_layer}):")
    print("   - Where we analyzed patches to find outlier positions")
    print("   - Typically the last layer (11) or second-to-last (10)")
    print(f"   - Based on manual inspection of norm progression plot")
    print(f"\n💥 Explosion Layer ({explosion_layer}):")
    print("   - Where patch norms first showed significant increase")
    print("   - Automatically detected from Phase 1 statistics")
    print("   - Defines the search range for register neurons")
    print(f"\n🔍 Backward Search Range: Layer 0 to {explosion_layer - 1}")
    print("   - Register neurons exist in layers BEFORE the explosion point")
    print("   - We search these layers using outlier positions from Phase 2")
    print(f"\n⚠️  ABLATION VERSION:")
    print("   - Neurons are discovered at MLP OUTPUT (after fc2), 768-dim space")
    print("   - This is different from the paper's middle-activation approach")

    # ==============================================================================
    # PHASE 4: Overall Top-K Selection (Optional)
    # ==============================================================================
    if args.top_k_overall is not None:
        print("\n" + "=" * 70)
        print(f"PHASE 4: Selecting Top-{args.top_k_overall} Neurons Overall")
        print("=" * 70)

        # Collect all neurons with their activation values and layer info
        all_neurons = []
        for layer_key, data in neuron_stats.items():
            # Skip metadata entries (if any)
            if isinstance(layer_key, str) and layer_key.startswith('_'):
                continue
            # Handle both integer and string keys
            layer_idx = int(layer_key)
            if 'top_indices' not in data or 'top_values' not in data:
                continue

            for neuron_idx, activation in zip(data['top_indices'], data['top_values']):
                all_neurons.append({
                    'layer': layer_idx,
                    'neuron': neuron_idx,
                    'activation': activation
                })

        # Sort by activation strength (descending)
        all_neurons.sort(key=lambda x: x['activation'], reverse=True)

        print(f"\n📊 Found {len(all_neurons)} neurons across {len(neuron_stats)} layers")
        print(f"\nTop {args.top_k_overall} neurons by activation strength:")
        print(f"{'Rank':>5} | {'Layer':>5} | {'Neuron':>6} | {'Activation':>12}")
        print("-" * 42)

        # Select top-k neurons
        selected_neurons = {}
        for i, neuron_data in enumerate(all_neurons[:args.top_k_overall]):
            layer_idx = neuron_data['layer']
            neuron_idx = neuron_data['neuron']
            activation = neuron_data['activation']

            print(f"{i+1:>5} | {layer_idx:>5} | {neuron_idx:>6} | {activation:>12.2f}")

            # Group by layer for the output format
            if layer_idx not in selected_neurons:
                selected_neurons[layer_idx] = []
            selected_neurons[layer_idx].append(neuron_idx)

        # Print summary by layer
        print(f"\n{'='*50}")
        print("Selected neurons by layer:")
        print(f"{'='*50}")
        for layer in sorted(selected_neurons.keys()):
            neurons = selected_neurons[layer]
            print(f"  Layer {layer:2d}: {len(neurons)} neurons -> {neurons}")

        print(f"\n{'='*50}")
        print(f"Total neurons selected: {sum(len(v) for v in selected_neurons.values())}")
        print(f"{'='*50}")

        # Determine output path
        if args.output:
            final_output_path = Path(args.output)
        else:
            final_output_path = Path("models/register_neurons.json")

        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in the expected format (layer -> list of neurons)
        result = {str(k): v for k, v in selected_neurons.items()}
        with open(final_output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Saved top-{args.top_k_overall} neurons to: {final_output_path}")

        # Update results_path for next steps
        results_path = final_output_path

    return 0


if __name__ == "__main__":
    sys.exit(main())
