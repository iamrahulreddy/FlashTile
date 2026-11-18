"""Plot style for every figure, same dark look as v1's visualization.py."""

from __future__ import annotations

BG = "#1A1A2E"
GRID = "#2D2D44"
TEXT = "white"

# same colors as v1
COLORS = {
    "naive": "#E74C3C",
    "flashtile-triton": "#9B59B6",
    "flashtile-tiled": "#1ABC9C",
    "flashtile-atomic": "#FFD700",
    "sdpa-flash": "#3498DB",
    "sdpa-cudnn": "#2ECC71",
    "sdpa-efficient": "#F39C12",
    "eager": "#E74C3C",
    "sdpa": "#3498DB",
    "flashtile": "#9B59B6",
    "fp32-eager": "#BDC3C7",
}
LABELS = {
    "naive": "Naive (PyTorch math, O(N²))",
    "flashtile-triton": "FlashTile (Triton)",
    "flashtile-tiled": "FlashTile (tiled PyTorch)",
    "flashtile-atomic": "FlashTile (Triton, atomic-dQ backward)",
    "sdpa-flash": "PyTorch SDPA flash",
    "sdpa-cudnn": "PyTorch SDPA cuDNN",
    "sdpa-efficient": "PyTorch SDPA mem-efficient",
    "eager": "HF eager (naive)",
    "sdpa": "HF SDPA (PyTorch)",
    "flashtile": "FlashTile",
}
MARKERS = ["o", "s", "^", "d", "v", "P"]


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def style_axes(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, which="both")
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, alpha=0.2, color=GRID, which="both")
        for spine in ax.spines.values():
            spine.set_color(GRID)


def title(ax, text):
    ax.set_title(text, fontsize=14, fontweight="bold", pad=10, color=TEXT)


def legend(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def suptitle(fig, text):
    fig.suptitle(text, fontsize=16, fontweight="bold", color=TEXT)


def save(fig, path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
