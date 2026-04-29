"""
FlashTile Visualization Module
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "Matplotlib not installed. Visualization features unavailable. "
        "Install with: pip install matplotlib"
    )


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )

class ColorPalette:
    """Professional color palettes for consistent styling."""
    
    PRIMARY = {
        "naive": "#E74C3C",      
        "flash_v1": "#3498DB",   
        "flash_v2": "#2ECC71",   
        "triton": "#9B59B6",    
        "gqa": "#F39C12",        
        "mqa": "#1ABC9C",        
        "pytorch": "#34495E",   
    }
    
    GRADIENT = [
        "#3498DB",
        "#2980B9",  
        "#1ABC9C",  
        "#27AE60",  
        "#F39C12",  
        "#E74C3C",  
        "#8E44AD",  
    ]
    
    # Background colors
    BG_LIGHT = "#FAFAFA"
    BG_DARK = "#1A1A2E"
    GRID_LIGHT = "#E0E0E0"
    GRID_DARK = "#2D2D44"



# Base Plot Class
class BasePlot:
    """
    Base class for all FlashTile visualizations.
    
    Provides common styling, theming, and export functionality.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        theme: str = "light",
        title: Optional[str] = None,
    ):
        _check_matplotlib()
        
        self.figsize = figsize
        self.dpi = dpi
        self.theme = theme
        self.title = title
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._apply_theme()
        
        if title:
            self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    def _apply_theme(self):
        """Apply light or dark theme to the plot."""
        if self.theme == "dark":
            self.fig.patch.set_facecolor(ColorPalette.BG_DARK)
            self.ax.set_facecolor(ColorPalette.BG_DARK)
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.title.set_color('white')
            self.ax.grid(True, alpha=0.2, color=ColorPalette.GRID_DARK)
            for spine in self.ax.spines.values():
                spine.set_color(ColorPalette.GRID_DARK)
        else:
            self.fig.patch.set_facecolor(ColorPalette.BG_LIGHT)
            self.ax.set_facecolor(ColorPalette.BG_LIGHT)
            self.ax.grid(True, alpha=0.3, color=ColorPalette.GRID_LIGHT)
    
    def add_annotation(
        self,
        text: str,
        xy: Tuple[float, float],
        xytext: Optional[Tuple[float, float]] = None,
        arrow: bool = True,
        fontsize: int = 10,
        color: Optional[str] = None,
        box: bool = True,
    ):
        """Add an annotation with optional arrow and background box."""
        color = color or ("white" if self.theme == "dark" else "black")
        
        kwargs = {
            "fontsize": fontsize,
            "color": color,
            "ha": "center",
        }
        
        if box:
            kwargs["bbox"] = dict(
                boxstyle="round,pad=0.5",
                facecolor="yellow" if self.theme == "light" else "#FFD700",
                alpha=0.9,
                edgecolor="none",
            )
        
        if arrow and xytext:
            kwargs["arrowprops"] = dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
            )
        
        self.ax.annotate(text, xy=xy, xytext=xytext or xy, **kwargs)
    
    def save(self, filepath: Union[str, Path], bbox_inches: str = "tight"):
        """Save plot to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches=bbox_inches,
            facecolor=self.fig.get_facecolor(),
        )
        print(f"Saved plot to: {filepath}")
    
    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the figure."""
        plt.close(self.fig)

class MemoryScalingPlot(BasePlot):
    """
    Plot memory usage scaling with sequence length.

    """
    
    def __init__(
        self,
        title: str = "Memory Usage: Naive vs Flash Attention",
        yscale: str = "log",
        **kwargs,
    ):
        super().__init__(title=title, **kwargs)
        self.yscale = yscale
        self.ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        self.ax.set_ylabel("Memory Usage (MB)", fontsize=12)
        if yscale:
            self.ax.set_yscale(yscale)
    
    def add_series(
        self,
        name: str,
        seq_lengths: List[int],
        memory_mb: List[float],
        color: Optional[str] = None,
        linewidth: float = 2.5,
        marker: str = "o",
        markersize: int = 8,
        linestyle: str = "-",
    ):
        """Add a memory scaling series."""
        color = color or ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
        
        # Replace None with NaN so matplotlib skips those points gracefully
        memory_mb = [float('nan') if v is None else v for v in memory_mb]
        
        self.ax.plot(
            seq_lengths,
            memory_mb,
            label=name,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            linestyle=linestyle,
            markeredgecolor='white' if self.theme == "light" else 'black',
            markeredgewidth=1,
        )
    
    def add_oom_region(
        self,
        gpu_memory_gb: float = 24.0,
        label: str = "GPU Memory Limit",
    ):
        """Add shading for out-of-memory region."""
        gpu_memory_mb = gpu_memory_gb * 1024
        
        # Add horizontal line
        self.ax.axhline(
            y=gpu_memory_mb,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=label,
        )
        
        # Shade OOM region
        self.ax.fill_between(
            self.ax.get_xlim(),
            gpu_memory_mb,
            self.ax.get_ylim()[1] if self.yscale == "linear" else gpu_memory_mb * 100,
            alpha=0.1,
            color="red",
            label="OOM Region",
        )
    
    def add_complexity_labels(self):
        """Add O(N²) and O(N) annotation labels."""
        # These are approximate positions - adjust based on your data
        x_pos = self.ax.get_xlim()[1] * 0.7
        
        if self.yscale == "log":
            y_naive = self.ax.get_ylim()[1] ** 0.7
            y_flash = self.ax.get_ylim()[1] ** 0.3
        else:
            y_naive = self.ax.get_ylim()[1] * 0.7
            y_flash = self.ax.get_ylim()[1] * 0.3
        
        self.ax.text(
            x_pos, y_naive,
            r"$O(N^2)$ Memory",
            fontsize=14,
            color=ColorPalette.PRIMARY["naive"],
            fontweight='bold',
            ha='center',
        )
        
        self.ax.text(
            x_pos, y_flash,
            r"$O(N)$ Memory",
            fontsize=14,
            color=ColorPalette.PRIMARY["flash_v2"],
            fontweight='bold',
            ha='center',
        )
    
    def finalize(self, legend_loc: str = "upper left", add_complexity: bool = True):
        """Finalize the plot with legend and styling."""
        if add_complexity:
            self.add_complexity_labels()
        
        self.ax.legend(
            loc=legend_loc,
            fontsize=11,
            framealpha=0.9 if self.theme == "light" else 0.8,
        )
        plt.tight_layout()

class PerformancePlot(BasePlot):
    """
    Plot performance metrics (time, throughput) comparison.
    """
    
    def __init__(
        self,
        title: str = "Performance Comparison",
        metric: str = "time",  # "time", "throughput", "speedup"
        **kwargs,
    ):
        super().__init__(title=title, **kwargs)
        self.metric = metric
        
        metric_labels = {
            "time": ("Sequence Length (tokens)", "Time (ms)"),
            "throughput": ("Sequence Length (tokens)", "Throughput (tokens/sec)"),
            "speedup": ("Sequence Length (tokens)", "Speedup vs Naive (×)"),
        }
        
        xlabel, ylabel = metric_labels.get(metric, metric_labels["time"])
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
    
    def add_series(
        self,
        name: str,
        seq_lengths: List[int],
        values: List[float],
        color: Optional[str] = None,
        linewidth: float = 2.5,
        marker: str = "s",
        markersize: int = 7,
    ):
        """Add a performance series."""
        color = color or ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
        
        # Replace None with NaN so matplotlib skips those points gracefully
        values = [float('nan') if v is None else v for v in values]
        
        self.ax.plot(
            seq_lengths,
            values,
            label=name,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            markeredgecolor='white' if self.theme == "light" else 'black',
            markeredgewidth=1,
        )
    
    def add_bar_comparison(
        self,
        categories: List[str],
        values: Dict[str, List[float]],
        width: float = 0.15,
    ):
        """Add grouped bar chart for discrete comparison."""
        x = np.arange(len(categories))
        
        for i, (name, vals) in enumerate(values.items()):
            color = ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), ColorPalette.GRADIENT[i])
            offset = width * (i - len(values) / 2 + 0.5)
            self.ax.bar(
                x + offset,
                vals,
                width,
                label=name,
                color=color,
                alpha=0.85,
                edgecolor='white' if self.theme == "light" else 'black',
                linewidth=1,
            )
        
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(categories)
    
    def finalize(self, legend_loc: str = "best"):
        """Finalize the plot."""
        self.ax.legend(
            loc=legend_loc,
            fontsize=11,
            framealpha=0.9 if self.theme == "light" else 0.8,
        )
        plt.tight_layout()


# Attention Heatmap
class AttentionHeatmap(BasePlot):
    """
    Visualize attention patterns as a heatmap.
    """
    
    def __init__(
        self,
        title: str = "Attention Pattern",
        cmap: str = "viridis",
        **kwargs,
    ):
        super().__init__(title=title, **kwargs)
        self.cmap = cmap
    
    def plot(
        self,
        attention_matrix: np.ndarray,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        show_cbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Plot attention heatmap."""
        im = self.ax.imshow(
            attention_matrix,
            cmap=self.cmap,
            aspect="auto",
            vmin=vmin or attention_matrix.min(),
            vmax=vmax or attention_matrix.max(),
        )
        
        if show_cbar:
            cbar = plt.colorbar(im, ax=self.ax)
            cbar.set_label("Attention Weight", fontsize=11)
            if self.theme == "dark":
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        if x_labels:
            self.ax.set_xticks(range(len(x_labels)))
            self.ax.set_xticklabels(x_labels, rotation=45, ha="right")
        
        if y_labels:
            self.ax.set_yticks(range(len(y_labels)))
            self.ax.set_yticklabels(y_labels)
        
        self.ax.set_xlabel("Key Position", fontsize=12)
        self.ax.set_ylabel("Query Position", fontsize=12)
    
    def add_causal_mask_overlay(self, seq_len: int):
        """Add diagonal line showing causal mask boundary."""
        self.ax.plot(
            [0, seq_len - 1],
            [0, seq_len - 1],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Causal Boundary",
        )
        self.ax.legend(loc="upper right")

class TrainingConvergencePlot(BasePlot):
    """
    Plot training metrics over time.
    """
    
    def __init__(
        self,
        title: str = "Training Convergence",
        **kwargs,
    ):
        super().__init__(title=title, **kwargs)
        self.ax.set_xlabel("Training Step", fontsize=12)
        self.ax.set_ylabel("Loss", fontsize=12)
    
    def add_loss_series(
        self,
        name: str,
        steps: List[int],
        losses: List[float],
        color: Optional[str] = None,
        smooth: Optional[int] = None,
    ):
        """Add a loss curve."""
        color = color or ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
        
        # Optional smoothing
        if smooth and len(losses) > smooth:
            kernel = np.ones(smooth) / smooth
            losses_smooth = np.convolve(losses, kernel, mode='valid')
            steps_smooth = steps[smooth-1:]
            
            # Plot raw with low alpha
            self.ax.plot(steps, losses, alpha=0.2, color=color)
            # Plot smoothed
            self.ax.plot(
                steps_smooth,
                losses_smooth,
                label=name,
                color=color,
                linewidth=2.5,
            )
        else:
            self.ax.plot(
                steps,
                losses,
                label=name,
                color=color,
                linewidth=2,
            )
    
    def add_convergence_zone(
        self,
        target_loss: float,
        tolerance: float = 0.1,
        label: str = "Convergence Target",
    ):
        """Add shaded convergence target zone."""
        self.ax.axhline(
            y=target_loss,
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=label,
        )
        self.ax.fill_between(
            self.ax.get_xlim(),
            target_loss * (1 - tolerance),
            target_loss * (1 + tolerance),
            alpha=0.1,
            color="green",
        )
    
    def finalize(self, legend_loc: str = "upper right"):
        """Finalize the plot."""
        self.ax.legend(
            loc=legend_loc,
            fontsize=11,
            framealpha=0.9 if self.theme == "light" else 0.8,
        )
        plt.tight_layout()

class BenchmarkDashboard:
    """
    Create a comprehensive multi-panel benchmark dashboard.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (18, 12),
        dpi: int = 150,
        theme: str = "light",
    ):
        _check_matplotlib()
        
        self.figsize = figsize
        self.dpi = dpi
        self.theme = theme
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme to all subplots."""
        bg_color = ColorPalette.BG_DARK if self.theme == "dark" else ColorPalette.BG_LIGHT
        grid_color = ColorPalette.GRID_DARK if self.theme == "dark" else ColorPalette.GRID_LIGHT
        text_color = 'white' if self.theme == "dark" else 'black'
        
        self.fig.patch.set_facecolor(bg_color)
        
        for ax in self.axes.flat:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            ax.grid(True, alpha=0.2 if self.theme == "dark" else 0.3, color=grid_color)
            for spine in ax.spines.values():
                spine.set_color(grid_color)
    
    def plot_memory_scaling(
        self,
        ax_idx: Tuple[int, int] = (0, 0),
        seq_lengths: Optional[List[int]] = None,
        memory_data: Optional[Dict[str, List[float]]] = None,
    ):
        """Plot memory scaling in specified subplot."""
        ax = self.axes[ax_idx]
        ax.set_title("Memory Scaling", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Memory (MB)")
        ax.set_yscale("log")
        
        if seq_lengths and memory_data:
            for name, memory in memory_data.items():
                color = ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
                ax.plot(seq_lengths, memory, label=name, marker='o', color=color, linewidth=2)
            ax.legend()
    
    def plot_performance(
        self,
        ax_idx: Tuple[int, int] = (0, 1),
        seq_lengths: Optional[List[int]] = None,
        time_data: Optional[Dict[str, List[float]]] = None,
    ):
        """Plot performance in specified subplot."""
        ax = self.axes[ax_idx]
        ax.set_title("Execution Time", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        
        if seq_lengths and time_data:
            for name, times in time_data.items():
                color = ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
                ax.plot(seq_lengths, times, label=name, marker='s', color=color, linewidth=2)
            ax.legend()
    
    def plot_speedup(
        self,
        ax_idx: Tuple[int, int] = (1, 0),
        seq_lengths: Optional[List[int]] = None,
        speedup_data: Optional[Dict[str, List[float]]] = None,
    ):
        """Plot speedup in specified subplot."""
        ax = self.axes[ax_idx]
        ax.set_title("Speedup vs Naive", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Speedup (×)")
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        if seq_lengths and speedup_data:
            for name, speedups in speedup_data.items():
                color = ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
                ax.plot(seq_lengths, speedups, label=name, marker='^', color=color, linewidth=2)
            ax.legend()
    
    def plot_reduction_ratio(
        self,
        ax_idx: Tuple[int, int] = (1, 1),
        seq_lengths: Optional[List[int]] = None,
        reduction_data: Optional[Dict[str, List[float]]] = None,
    ):
        """Plot memory reduction ratio."""
        ax = self.axes[ax_idx]
        ax.set_title("Memory Reduction", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Reduction Ratio (×)")
        
        if seq_lengths and reduction_data:
            for name, ratios in reduction_data.items():
                color = ColorPalette.PRIMARY.get(name.lower().replace(" ", "_"), "#333")
                ax.plot(seq_lengths, ratios, label=name, marker='d', color=color, linewidth=2)
            ax.legend()
    
    def save(self, filepath: Union[str, Path]):
        """Save dashboard."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        self.fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
        )
        print(f"Saved dashboard to: {filepath}")
    
    def show(self):
        """Display dashboard."""
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the figure."""
        plt.close(self.fig)


__all__ = [
    "HAS_MATPLOTLIB",
    "ColorPalette",
    "BasePlot",
    "MemoryScalingPlot",
    "PerformancePlot",
    "AttentionHeatmap",
    "TrainingConvergencePlot",
    "BenchmarkDashboard",
]
