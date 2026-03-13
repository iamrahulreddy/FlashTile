"""
Attention Pattern Visualizer
============================

This module provides visualization utilities for understanding attention
patterns, memory usage, and block access patterns in Flash Attention.

Features
--------
- Attention heatmaps with highlighting
- Memory timeline plots showing HBM vs SRAM usage
- Block access pattern visualization
- Comparison plots for different implementations

Requirements
------------
- matplotlib >= 3.5.0
- numpy >= 1.20.0
- seaborn >= 0.11.0 (optional, for enhanced styling)

Usage
-----
>>> from flashtile.utils import AttentionVisualizer
>>> visualizer = AttentionVisualizer()
>>> fig = visualizer.plot_attention_heatmap(attention_weights)
>>> fig.savefig("attention_pattern.png")
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class AttentionVisualizer:
    """
    Visualization toolkit for Flash Attention analysis.
    This class provides methods for visualizations of attention patterns, memory usage, and
    algorithm behavior.
    """

    def __init__(
        self,
        style: str = "default",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """Initialize the visualizer."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is required for AttentionVisualizer. "
                "Install with: pip install matplotlib"
            )

        self.figsize = figsize
        self.dpi = dpi
        self._setup_style(style)

    def _setup_style(self, style: str) -> None:
        """Configure matplotlib style."""
        if style == "dark":
            plt.style.use("dark_background")
        elif style == "paper":
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "figure.titlesize": 18,
            })
            if HAS_SEABORN:
                sns.set_palette("deep")

    def plot_attention_heatmap(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        title: str = "Attention Pattern",
        show_colorbar: bool = True,
        highlight_causal: bool = False,
        highlight_blocks: Optional[List[Tuple[int, int, int, int]]] = None,
        cmap: str = "viridis",
    ) -> "plt.Figure":
        """
        Plot attention weights as a heatmap.

        Parameters
        ----------
        attention_weights : torch.Tensor or np.ndarray
            2D attention matrix of shape (seq_len, seq_len).

        title : str, optional
            Plot title. Default is "Attention Pattern".

        show_colorbar : bool, optional
            Whether to show colorbar. Default is True.

        highlight_causal : bool, optional
            Highlight the causal mask boundary. Default is False.

        highlight_blocks : list of tuples, optional
            List of (row_start, row_end, col_start, col_end) blocks to highlight.

        cmap : str, optional
            Colormap name. Default is "viridis".

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot heatmap
        im = ax.imshow(attention_weights, cmap=cmap, aspect="auto")

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Attention Weight")

        # Highlight causal boundary
        if highlight_causal:
            seq_len = attention_weights.shape[0]
            ax.plot([0, seq_len], [0, seq_len], "r--", linewidth=2, label="Causal boundary")

        # Highlight blocks
        if highlight_blocks:
            for i, (r_start, r_end, c_start, c_end) in enumerate(highlight_blocks):
                rect = patches.Rectangle(
                    (c_start - 0.5, r_start - 0.5),
                    c_end - c_start,
                    r_end - r_start,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)

        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_block_access_pattern(
        self,
        seq_len: int,
        block_size: int = 64,
        causal: bool = True,
        title: str = "Flash Attention Block Access Pattern",
    ) -> "plt.Figure":
        """
        Visualize which Q-K/V block pairs are computed in Flash Attention.

        This shows the tiling pattern and demonstrates causal block skipping.

        Parameters
        ----------
        seq_len : int
            Sequence length.

        block_size : int, optional
            Block size for tiling. Default is 64.

        causal : bool, optional
            Whether to show causal block skipping. Default is True.

        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        num_blocks = math.ceil(seq_len / block_size)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create pattern matrix
        # 0 = skipped, 1 = computed, 2 = diagonal (partial causal)
        pattern = np.ones((num_blocks, num_blocks), dtype=int)

        if causal:
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if j > i:
                        pattern[i, j] = 0  # Skipped
                    elif j == i:
                        pattern[i, j] = 2  # Diagonal

        # Custom colormap: white=skipped, blue=computed, orange=diagonal
        colors = ["#ffffff", "#1f77b4", "#ff7f0e"]
        cmap = LinearSegmentedColormap.from_list("block_pattern", colors, N=3)

        im = ax.imshow(pattern, cmap=cmap, vmin=0, vmax=2)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, num_blocks, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_blocks, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Labels
        ax.set_xticks(np.arange(0, num_blocks, max(1, num_blocks // 8)))
        ax.set_yticks(np.arange(0, num_blocks, max(1, num_blocks // 8)))
        ax.set_xlabel(f"K/V Block Index (block_size={block_size})")
        ax.set_ylabel(f"Q Block Index (block_size={block_size})")
        ax.set_title(title)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ffffff", edgecolor="black", label="Skipped (future)"),
            Patch(facecolor="#1f77b4", label="Computed"),
            Patch(facecolor="#ff7f0e", label="Diagonal (partial mask)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Compute savings
        if causal:
            total_blocks = num_blocks * num_blocks
            computed_blocks = (num_blocks * (num_blocks + 1)) // 2
            savings = (1 - computed_blocks / total_blocks) * 100
            ax.annotate(
                f"Causal savings: {savings:.1f}%",
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        plt.tight_layout()
        return fig

    def plot_memory_comparison(
        self,
        seq_lengths: List[int],
        title: str = "Memory Usage: Naive vs Flash Attention",
    ) -> "plt.Figure":
        """
        Compare memory usage between naive and Flash Attention.

        Parameters
        ----------
        seq_lengths : list of int
            Sequence lengths to compare.

        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate memory for each approach
        # Assuming batch=1, heads=8, head_dim=64, FP16
        dtype_bytes = 2
        num_heads = 8
        head_dim = 64

        naive_memory = []
        flash_memory = []

        for n in seq_lengths:
            # Naive: stores N×N attention matrix per head
            naive_bytes = num_heads * n * n * dtype_bytes
            naive_memory.append(naive_bytes / 1e9)  # GB

            # Flash: only stores running statistics O(N)
            flash_bytes = num_heads * n * (head_dim + 2) * dtype_bytes  # Output + m + l
            flash_memory.append(flash_bytes / 1e9)  # GB

        x = np.arange(len(seq_lengths))
        width = 0.35

        bars1 = ax.bar(x - width / 2, naive_memory, width, label="Naive Attention", color="#d62728")
        bars2 = ax.bar(x + width / 2, flash_memory, width, label="Flash Attention", color="#2ca02c")

        ax.set_ylabel("Memory Usage (GB)")
        ax.set_xlabel("Sequence Length")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in seq_lengths])
        ax.legend()
        ax.set_yscale("log")

        # Add reduction annotations
        for i, (n_mem, f_mem) in enumerate(zip(naive_memory, flash_memory)):
            reduction = n_mem / f_mem if f_mem > 0 else float("inf")
            ax.annotate(
                f"{reduction:.0f}×",
                xy=(i, max(n_mem, f_mem)),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def plot_online_softmax_state(
        self,
        scores: Union[torch.Tensor, np.ndarray],
        block_size: int = 4,
        title: str = "Online Softmax State Evolution",
    ) -> "plt.Figure":
        """
        Visualize how online softmax statistics evolve during block processing.

        Parameters
        ----------
        scores : torch.Tensor or np.ndarray
            1D attention scores for a single query row.

        block_size : int, optional
            Block size for processing. Default is 4 (small for visualization).

        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        if scores.ndim > 1:
            scores = scores.flatten()

        n = len(scores)
        num_blocks = math.ceil(n / block_size)

        # Simulate online softmax
        m_history = []
        l_history = []
        m = float("-inf")
        l = 0.0

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, n)
            block = scores[start:end]

            block_max = np.max(block)
            m_new = max(m, block_max)

            if m == float("-inf"):
                l_new = np.sum(np.exp(block - m_new))
            else:
                alpha = np.exp(m - m_new)
                l_new = l * alpha + np.sum(np.exp(block - m_new))

            m = m_new
            l = l_new

            m_history.append(m)
            l_history.append(l)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=self.dpi)

        # Plot running maximum
        axes[0].plot(range(1, num_blocks + 1), m_history, "o-", color="#1f77b4", linewidth=2, markersize=8)
        axes[0].axhline(y=np.max(scores), color="red", linestyle="--", label="True global max")
        axes[0].set_xlabel("Block Number")
        axes[0].set_ylabel("Running Maximum (m)")
        axes[0].set_title("Running Maximum Evolution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot running sum
        true_sum = np.sum(np.exp(scores - np.max(scores)))
        axes[1].plot(range(1, num_blocks + 1), l_history, "o-", color="#ff7f0e", linewidth=2, markersize=8)
        axes[1].axhline(y=true_sum, color="red", linestyle="--", label="True sum (after rescaling)")
        axes[1].set_xlabel("Block Number")
        axes[1].set_ylabel("Running Sum (ℓ)")
        axes[1].set_title("Running Sum Evolution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def save_all_visualizations(
        self,
        output_dir: str,
        seq_len: int = 512,
        block_size: int = 64,
    ) -> Dict[str, str]:
        """
        Generate and save all standard visualizations.

        Parameters
        ----------
        output_dir : str
            Directory to save visualizations.

        seq_len : int, optional
            Sequence length for visualizations. Default is 512.

        block_size : int, optional
            Block size for visualizations. Default is 64.

        Returns
        -------
        dict
            Mapping of visualization name to saved file path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Block access pattern
        fig = self.plot_block_access_pattern(seq_len, block_size, causal=True)
        path = os.path.join(output_dir, "block_access_pattern.png")
        fig.savefig(path)
        plt.close(fig)
        saved_files["block_access_pattern"] = path

        # Memory comparison
        fig = self.plot_memory_comparison([512, 1024, 2048, 4096, 8192])
        path = os.path.join(output_dir, "memory_comparison.png")
        fig.savefig(path)
        plt.close(fig)
        saved_files["memory_comparison"] = path

        # Online softmax state
        scores = np.random.randn(64)
        fig = self.plot_online_softmax_state(scores, block_size=8)
        path = os.path.join(output_dir, "online_softmax_state.png")
        fig.savefig(path)
        plt.close(fig)
        saved_files["online_softmax_state"] = path

        return saved_files
