"""Visualization helpers for swipe attention analysis.

The primary entry points are:
- `create_layer_comparison_grid`
- `create_summary_visualization`
- `create_layer_pooled_visualization`
- `create_single_layer_timeline_plot`
- `create_attention_timeline_plot`
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

# Set seaborn style for better aesthetics
sns.set_context("notebook", font_scale=1.0)
sns.set_style("whitegrid")


def _filter_path_points(
    path_coords: np.ndarray, path_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if path_mask is None:
        return path_coords, np.arange(len(path_coords))
    valid_indices = np.where(path_mask == 1)[0]
    return path_coords[valid_indices], valid_indices


def _infer_time_axis(path_coords: np.ndarray) -> tuple[np.ndarray, str]:
    """Infer a sensible x-axis for timeline plots.

    Supported path formats:
    - [n, 3]: (x, y, t) where t may be unix time or relative time.
    - [n, 6]: (x, y, dx, dy, ds, log1p(dt)) => time is cumulative expm1(log1p(dt)).
    - [n, 2] or other: fall back to path position index.
    """
    if path_coords.ndim != 2:
        raise ValueError(f"path_coords must be 2D, got shape {path_coords.shape}")

    n_points, d = path_coords.shape
    if n_points == 0:
        return np.array([]), "Time"

    # Engineered features: (x, y, dx, dy, ds, log1p(dt))
    if d >= 6:
        log1dt = path_coords[:, 5]
        dt = np.expm1(log1dt)
        dt = np.clip(dt, 0.0, None)
        times = np.cumsum(dt)
        return times, "Time (cumulative dt)"

    # Raw (x, y, t)
    if d >= 3:
        t = path_coords[:, 2]
        # If it looks like unix time, convert to relative seconds
        if np.nanmedian(t) > 1e8:
            t0 = t[0]
            return t - t0, "Time (relative)"
        return t, "Time"

    return np.arange(n_points), "Path position"


def _pool_layer_attentions(attention_stack: np.ndarray, pooling_method: str) -> np.ndarray:
    """Pool an attention stack across layers.

    Args:
        attention_stack: [n_layers, n_chars, n_path]
        pooling_method: "max" | "mean" | "sum" | "logsumexp"

    Returns:
        [n_chars, n_path]
    """
    if pooling_method == "max":
        return attention_stack.max(axis=0)
    if pooling_method == "mean":
        return attention_stack.mean(axis=0)
    if pooling_method == "sum":
        return attention_stack.sum(axis=0)
    if pooling_method == "logsumexp":
        # scipy-free stable logsumexp over layers
        m = attention_stack.max(axis=0)
        return m + np.log(np.exp(attention_stack - m).sum(axis=0) + 1e-12)
    raise ValueError(f"Unknown pooling method: {pooling_method}")


def plot_attention_heatmap_on_path(
    ax: plt.Axes,
    path_coords: np.ndarray,
    attention: np.ndarray,
    char_idx: int,
    word: str,
    layer_name: str = "",
    path_mask: np.ndarray | None = None,
    global_vmin: float | None = None,
    global_vmax: float | None = None,
):
    """Overlay attention heatmap on 3D swipe path (x, y, time).

    Args:
        ax: Matplotlib 3D axes to plot on
        path_coords: Path coordinates [n_path_points, *] (supports raw [n,3] or engineered [n,6])
        attention: Attention weights [n_chars, n_path_points]
        char_idx: Index of character to visualize (0 to n_chars-1)
        word: The target word
        layer_name: Optional layer name for title (e.g., "Layer 0")
        path_mask: Optional mask indicating valid path points (1=valid, 0=padding)
        global_vmin: Global minimum for color normalization (optional)
        global_vmax: Global maximum for color normalization (optional)
    """
    path_coords_filtered, valid_indices = _filter_path_points(path_coords, path_mask)
    char_attention_full = attention[char_idx, :]
    char_attention = char_attention_full[valid_indices]

    # Extract x, y coordinates and infer time axis
    xs = path_coords_filtered[:, 0]
    ys = path_coords_filtered[:, 1]
    ts, time_label = _infer_time_axis(path_coords_filtered)

    # Use global normalization if provided, otherwise normalize per-plot
    if global_vmin is not None and global_vmax is not None:
        vmin, vmax = global_vmin, global_vmax
        attn_normalized = (char_attention - vmin) / (vmax - vmin + 1e-8)
        attn_normalized = np.clip(attn_normalized, 0, 1)  # Clip to [0, 1]
    else:
        vmin, vmax = char_attention.min(), char_attention.max()
        attn_normalized = (char_attention - vmin) / (vmax - vmin + 1e-8)

    # Create colormap for attention intensity
    cmap = cm.get_cmap("BuPu")
    norm = Normalize(vmin=0, vmax=1)

    # Plot path as thin line in 3D
    ax.plot(xs, ys, ts, color="gray", linewidth=1, alpha=0.3, zorder=1)

    # Mark start and end of path
    ax.scatter(
        xs[0],
        ys[0],
        ts[0],
        c="green",
        s=80,
        marker="o",
        edgecolors="green",
        linewidths=2,
        zorder=2,
        label="Path start",
        alpha=0.8,
    )
    ax.scatter(
        xs[-1],
        ys[-1],
        ts[-1],
        c="red",
        s=80,
        marker="s",
        edgecolors="red",
        linewidths=2,
        zorder=2,
        label="Path end",
        alpha=0.8,
    )

    # 3D scatter plot with variable size and color based on attention
    sizes = 4 + 120 * (np.exp(attn_normalized) - 1)  # Min 30, max 150

    scatter = ax.scatter(
        xs,
        ys,
        ts,
        c=attn_normalized,
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    # Get character being predicted
    if char_idx < len(word):
        char = word[char_idx]
        char_label = f"'{char}'"
    else:
        char_label = f"pos {char_idx}"

    # Title
    title = f"Character {char_idx}: {char_label}"
    if layer_name:
        title = f"{layer_name} - {title}"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=20)

    # 3D axis labels
    ax.set_xlabel("X", fontsize=9, labelpad=10)
    ax.set_ylabel("Y", fontsize=9, labelpad=10)
    ax.set_zlabel(time_label, fontsize=9, labelpad=10)

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Time axis autoscales based on data

    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)

    # Grid and legend
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.1, shrink=0.8)
    cbar.set_label("Attention", rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def create_layer_comparison_grid(
    layer_attentions: dict[int, np.ndarray],
    path_coords: np.ndarray,
    word: str,
    char_indices: list[int] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
    path_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Create subplot grid comparing attention across layers for multiple characters.

    Layout: rows = characters, columns = layers

    Args:
        layer_attentions: Dict mapping layer_idx -> attention [n_chars, n_path_points]
        path_coords: Path coordinates [n_path_points, *]
        word: Target word
        char_indices: Which characters to visualize (default: first 5 or all if word is short)
        figsize: Figure size (default: auto-calculated)
        save_path: Optional path to save figure
        path_mask: Optional mask indicating valid path points (1=valid, 0=padding)

    Returns:
        Matplotlib figure
    """
    # Determine which characters to visualize
    word_len = len(word)
    if char_indices is None:
        # Default: show first 5 characters or all if word is short
        char_indices = list(range(min(5, word_len)))

    n_chars = len(char_indices)
    n_layers = len(layer_attentions)
    layer_ids = sorted(layer_attentions.keys())

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (5 * n_layers, 4 * n_chars)

    # Compute global min/max across all layers and characters for normalization
    global_vmin = float("inf")
    global_vmax = float("-inf")
    for layer_idx in layer_ids:
        attention = layer_attentions[layer_idx]
        for char_idx in char_indices:
            char_attn = attention[char_idx, :]
            global_vmin = min(global_vmin, char_attn.min())
            global_vmax = max(global_vmax, char_attn.max())

    # Create figure with 3D subplots
    fig = plt.figure(figsize=figsize)

    # Plot each (character, layer) combination
    for row_idx, char_idx in enumerate(char_indices):
        for col_idx, layer_idx in enumerate(layer_ids):
            # Create 3D subplot
            subplot_idx = row_idx * n_layers + col_idx + 1
            ax = fig.add_subplot(n_chars, n_layers, subplot_idx, projection="3d")

            attention = layer_attentions[layer_idx]

            plot_attention_heatmap_on_path(
                ax=ax,
                path_coords=path_coords,
                attention=attention,
                char_idx=char_idx,
                word=word,
                layer_name=f"Layer {layer_idx}",
                path_mask=path_mask,
                global_vmin=global_vmin,
                global_vmax=global_vmax,
            )

    # Overall title
    fig.suptitle(f'Attention Evolution: "{word}"', fontsize=16, fontweight="bold", y=0.995)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_summary_visualization(
    layer_attentions: dict[int, np.ndarray],
    path_coords: np.ndarray,
    word: str,
    save_path: str | None = None,
    path_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Create a compact summary figure (n layers × 3 characters)."""
    layer_ids = sorted(layer_attentions.keys())
    n_layers = max(len(layer_ids), 1)
    fig = plt.figure(figsize=(6 * n_layers, 12))

    # Create grid
    gs = fig.add_gridspec(3, n_layers, hspace=0.3, wspace=0.3)

    # Compute global min/max for normalization
    char_indices_to_plot = [0, min(1, len(word) - 1), min(2, len(word) - 1)]

    global_vmin = float("inf")
    global_vmax = float("-inf")
    for layer_idx in layer_ids:
        attention = layer_attentions[layer_idx]
        for char_idx in char_indices_to_plot:
            char_attn = attention[char_idx, :]
            global_vmin = min(global_vmin, char_attn.min())
            global_vmax = max(global_vmax, char_attn.max())

    # Top row: 3D heatmaps for first 3 characters across layers
    for col_idx, layer_idx in enumerate(layer_ids):
        ax = fig.add_subplot(gs[0, col_idx], projection="3d")
        attention = layer_attentions[layer_idx]
        plot_attention_heatmap_on_path(
            ax,
            path_coords,
            attention,
            char_idx=0,
            word=word,
            layer_name=f"Layer {layer_idx}",
            path_mask=path_mask,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
        )

    # Middle row: 2nd character
    for col_idx, layer_idx in enumerate(layer_ids):
        ax = fig.add_subplot(gs[1, col_idx], projection="3d")
        attention = layer_attentions[layer_idx]
        char_idx = min(1, len(word) - 1)
        plot_attention_heatmap_on_path(
            ax,
            path_coords,
            attention,
            char_idx=char_idx,
            word=word,
            layer_name=f"Layer {layer_idx}",
            path_mask=path_mask,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
        )

    # Bottom row: 3rd character
    for col_idx, layer_idx in enumerate(layer_ids):
        ax = fig.add_subplot(gs[2, col_idx], projection="3d")
        attention = layer_attentions[layer_idx]
        char_idx = min(2, len(word) - 1)
        plot_attention_heatmap_on_path(
            ax,
            path_coords,
            attention,
            char_idx=char_idx,
            word=word,
            layer_name=f"Layer {layer_idx}",
            path_mask=path_mask,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
        )

    fig.suptitle(f'Attention Analysis: "{word}"', fontsize=18, fontweight="bold", y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_layer_pooled_visualization(
    layer_attentions: dict[int, np.ndarray],
    path_coords: np.ndarray,
    word: str,
    pooling_method: str = "max",
    save_path: str | None = None,
    path_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Create visualization showing attention pooled across all layers.

    Args:
        layer_attentions: Dict mapping layer_idx -> attention [n_chars, n_path_points]
        path_coords: Path coordinates [n_path_points, *]
        word: Target word
        pooling_method: How layers were pooled ("max", "mean", or "sum")
        save_path: Optional save path
        path_mask: Optional mask indicating valid path points

    Returns:
        Matplotlib figure
    """
    # Pool attention across all layers
    attention_stack = np.stack([attn for attn in layer_attentions.values()], axis=0)
    # Shape: [n_layers, n_chars, n_path_points]

    pooled_attention = _pool_layer_attentions(attention_stack, pooling_method)

    # Shape after pooling: [n_chars, n_path_points]

    # Show all characters in the word
    word_len = len(word)
    n_chars = word_len

    # Compute global min/max across all characters for normalization
    global_vmin = pooled_attention.min()
    global_vmax = pooled_attention.max()

    # Create figure with 3D subplots
    n_cols = 3
    n_rows = (n_chars + n_cols - 1) // n_cols  # Ceiling division

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Plot each character
    for idx in range(n_chars):
        subplot_idx = idx + 1
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")

        plot_attention_heatmap_on_path(
            ax=ax,
            path_coords=path_coords,
            attention=pooled_attention,
            char_idx=idx,
            word=word,
            layer_name=f"All Layers ({pooling_method.capitalize()})",
            path_mask=path_mask,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
        )

    # Hide unused subplots
    for idx in range(n_chars, n_rows * n_cols):
        subplot_idx = idx + 1
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")
        ax.axis("off")

    fig.suptitle(
        f'Layer-Pooled Attention: "{word}" ({pooling_method.capitalize()} across {len(layer_attentions)} layers)',
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_single_layer_timeline_plot(
    layer_attention: np.ndarray,
    layer_idx: int,
    path_coords: np.ndarray,
    word: str,
    save_path: str | None = None,
    path_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Create 2D line plot showing attention vs time for a single layer.

    Args:
        layer_attention: Attention for single layer [n_chars, n_path_points]
        layer_idx: Index of the layer
        path_coords: Path coordinates [n_path_points, *] (supports raw [n,3] or engineered [n,6])
        word: Target word
        save_path: Optional save path
        path_mask: Optional mask indicating valid path points

    Returns:
        Matplotlib figure
    """
    path_coords_filtered, valid_indices = _filter_path_points(path_coords, path_mask)
    layer_attention_filtered = layer_attention[:, valid_indices]

    times, time_label = _infer_time_axis(path_coords_filtered)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for different characters
    colors = plt.cm.tab10(np.linspace(0, 1, len(word)))

    # Plot each character's attention vs time
    for char_idx in range(len(word)):
        char = word[char_idx]
        char_attention = layer_attention_filtered[char_idx, :]

        ax.plot(
            times,
            char_attention,
            linewidth=2.5,
            label=f"'{char}' (pos {char_idx})",
            color=colors[char_idx],
            alpha=0.8,
            marker="o",
            markersize=4,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    # Compute max and mean across all characters
    max_attention = layer_attention_filtered[: len(word), :].max(axis=0)
    mean_attention = layer_attention_filtered[: len(word), :].mean(axis=0)

    # Plot max across all characters (black dotted)
    ax.plot(
        times,
        max_attention,
        linewidth=3,
        label="Max across all chars",
        color="black",
        alpha=0.9,
        linestyle=":",
        zorder=10,
    )

    # Plot mean across all characters (red solid)
    ax.plot(
        times,
        mean_attention,
        linewidth=3,
        label="Mean across all chars",
        color="red",
        alpha=0.9,
        linestyle="-",
        zorder=9,
    )

    # Formatting
    ax.set_xlabel(time_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f'Attention Timeline: "{word}" (Layer {layer_idx})', fontsize=14, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_attention_timeline_plot(
    layer_attentions: dict[int, np.ndarray],
    path_coords: np.ndarray,
    word: str,
    pooling_method: str = "max",
    save_path: str | None = None,
    path_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Create 2D line plot showing attention vs time for each character.

    Args:
        layer_attentions: Dict mapping layer_idx -> attention [n_chars, n_path_points]
        path_coords: Path coordinates [n_path_points, *] (supports raw [n,3] or engineered [n,6])
        word: Target word
        pooling_method: How layers were pooled ("max", "mean", or "sum")
        save_path: Optional save path
        path_mask: Optional mask indicating valid path points
    Returns:
        Matplotlib figure
    """
    path_coords_filtered, valid_indices = _filter_path_points(path_coords, path_mask)

    times, time_label = _infer_time_axis(path_coords_filtered)

    # Pool attention across all layers
    attention_stack = np.stack([attn for attn in layer_attentions.values()], axis=0)

    pooled_attention = _pool_layer_attentions(attention_stack, pooling_method)

    # Filter attention to valid indices
    if path_mask is not None:
        pooled_attention = pooled_attention[:, valid_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for different characters
    colors = plt.cm.tab10(np.linspace(0, 1, len(word)))

    # Plot each character's attention vs time
    for char_idx in range(len(word)):
        char = word[char_idx]
        char_attention = pooled_attention[char_idx, :]

        ax.plot(
            times,
            char_attention,
            linewidth=2.5,
            label=f"'{char}' (pos {char_idx})",
            color=colors[char_idx],
            alpha=0.8,
            marker="o",
            markersize=4,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    # Compute max and mean across characters
    all_attention_stack = np.stack([pooled_attention[i, :] for i in range(len(word))], axis=0)
    max_attention_all = all_attention_stack.max(axis=0)
    # Plot max across all tokens (black dashed)
    ax.plot(
        times,
        max_attention_all,
        linewidth=3,
        label="Max across all tokens",
        color="black",
        alpha=0.9,
        linestyle=":",
        # marker='s',
        # markersize=0,
        # markeredgewidth=0,
        # markeredgecolor='white',
        zorder=10,  # Draw on top
    )

    # Formatting
    ax.set_xlabel(time_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f'Attention Timeline: "{word}" ({pooling_method.capitalize()} across {len(layer_attentions)} layers)',
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Set y-axis to start at 0 for better comparison
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
