#!/usr/bin/env python3
"""
Create clear, educational animation showing time series windowing concepts.

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
Shows:
1. Sliding window concept
2. Train/test splits
3. Non-overlapping windows
4. Purge gaps for leak prevention
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# Configuration
FPS = 5
DURATION = 12  # seconds
N_FRAMES = FPS * DURATION

# Generate synthetic time series
n_points = 100
time = np.arange(n_points)
# Create realistic time series with trend and seasonality
trend = 0.5 * time
seasonal = 10 * np.sin(2 * np.pi * time / 20)
noise = np.random.normal(0, 2, n_points)
data = trend + seasonal + noise + 50

# Windowing parameters
window_size = 20
test_size = 5
purge_gap = 2

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10), facecolor="white")
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])  # Main time series
ax2 = fig.add_subplot(gs[1, 0])  # Sliding window
ax3 = fig.add_subplot(gs[1, 1])  # Train/test split
ax4 = fig.add_subplot(gs[2, 0])  # Non-overlapping windows
ax5 = fig.add_subplot(gs[2, 1])  # Purged forward CV

# Remove top and right spines (Tufte style)
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def update(frame):
    """Update animation frame."""
    # Clear all axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.clear()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Calculate current position based on frame
    progress = frame / N_FRAMES
    # ========================================================================
    # PLOT 1: Main time series with current position
    # ========================================================================
    ax1.plot(time, data, "black", linewidth=2, alpha=0.7)
    # Show current analysis window
    current_pos = int(progress * (n_points - window_size))
    window_data_x = time[current_pos : current_pos + window_size]
    window_data_y = data[current_pos : current_pos + window_size]
    ax1.fill_between(window_data_x, 0, window_data_y, alpha=0.2, color="blue")
    ax1.plot(window_data_x, window_data_y, "blue", linewidth=3)
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.set_title("Time Series Windowing Animation", fontsize=13, fontweight="normal")
    ax1.set_xlim(0, n_points)
    # ========================================================================
    # PLOT 2: Sliding Window (overlapping)
    # ========================================================================
    ax2.plot(time, data, "gray", linewidth=1, alpha=0.3)
    # Show multiple overlapping windows
    for i in range(0, min(current_pos + 1, 60), 5):
        alpha = 0.1 + 0.3 * (i / max(current_pos, 1))
        if i + window_size <= n_points:
            ax2.fill_between(
                time[i : i + window_size],
                0,
                data[i : i + window_size],
                alpha=alpha,
                color="blue",
            )

    # Highlight current window
    if current_pos + window_size <= n_points:
        ax2.plot(window_data_x, window_data_y, "blue", linewidth=2)

    ax2.set_xlabel("Time", fontsize=10)
    ax2.set_ylabel("Value", fontsize=10)
    ax2.set_title("1. Sliding Window (Overlapping)", fontsize=11, fontweight="normal")
    ax2.text(
        0.05,
        0.95,
        "Step size: 5\nOverlap: YES",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
    )
    ax2.set_xlim(0, n_points)
    # ========================================================================
    # PLOT 3: Train/Test Split
    # ========================================================================
    split_point = int(0.7 * n_points)
    # Train data
    ax3.plot(
        time[:split_point],
        data[:split_point],
        "blue",
        linewidth=2,
        label="Train",
        alpha=0.7,
    )
    ax3.fill_between(time[:split_point], 0, data[:split_point], alpha=0.2, color="blue")
    # Test data (reveal progressively)
    test_reveal = int(progress * (n_points - split_point))
    if test_reveal > 0:
        test_end = min(split_point + test_reveal, n_points)
        ax3.plot(
            time[split_point:test_end],
            data[split_point:test_end],
            "red",
            linewidth=2,
            label="Test",
            alpha=0.7,
        )
        ax3.fill_between(
            time[split_point:test_end],
            0,
            data[split_point:test_end],
            alpha=0.2,
            color="red",
        )

    # Split line
    ax3.axvline(split_point, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax3.text(
        split_point, ax3.get_ylim()[1], " Split", fontsize=9, verticalalignment="top"
    )
    ax3.set_xlabel("Time", fontsize=10)
    ax3.set_ylabel("Value", fontsize=10)
    ax3.set_title("2. Train/Test Split (70/30)", fontsize=11, fontweight="normal")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_xlim(0, n_points)
    # ========================================================================
    # PLOT 4: Non-overlapping Windows
    # ========================================================================
    ax4.plot(time, data, "gray", linewidth=1, alpha=0.3)
    # Show non-overlapping windows
    n_windows = int(progress * 4) + 1
    colors_cycle = ["blue", "green", "orange", "purple"]
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n_points)
        if end <= n_points:
            color = colors_cycle[i % len(colors_cycle)]
            ax4.fill_between(
                time[start:end], 0, data[start:end], alpha=0.3, color=color
            )
            ax4.plot(time[start:end], data[start:end], color, linewidth=2)
            # Add window label
            mid_point = (start + end) // 2
            ax4.text(
                mid_point,
                ax4.get_ylim()[1] * 0.9,
                f"W{i + 1}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax4.set_xlabel("Time", fontsize=10)
    ax4.set_ylabel("Value", fontsize=10)
    ax4.set_title("3. Non-Overlapping Windows", fontsize=11, fontweight="normal")
    ax4.text(
        0.05,
        0.95,
        "No overlap\nNo leakage",
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
    )
    ax4.set_xlim(0, n_points)
    # ========================================================================
    # PLOT 5: Purged Forward Cross-Validation
    # ========================================================================
    ax5.plot(time, data, "gray", linewidth=1, alpha=0.3)
    # Show expanding window with purge gap
    n_cv_folds = int(progress * 3) + 1
    for fold in range(n_cv_folds):
        # Train: expanding from start
        train_end = 30 + fold * 15
        if train_end < n_points - purge_gap - test_size:
            ax5.fill_between(
                time[:train_end], 0, data[:train_end], alpha=0.2, color="blue"
            )
            ax5.plot(time[:train_end], data[:train_end], "blue", linewidth=1, alpha=0.5)
            # Purge gap
            purge_start = train_end
            purge_end = train_end + purge_gap
            if purge_end < n_points:
                ax5.fill_between(
                    time[purge_start:purge_end],
                    ax5.get_ylim()[0],
                    ax5.get_ylim()[1],
                    alpha=0.2,
                    color="yellow",
                    zorder=0,
                )

            # Test: after purge gap
            test_start = train_end + purge_gap
            test_end = min(test_start + test_size, n_points)
            if test_end <= n_points:
                ax5.fill_between(
                    time[test_start:test_end],
                    0,
                    data[test_start:test_end],
                    alpha=0.4,
                    color="red",
                )
                ax5.plot(
                    time[test_start:test_end],
                    data[test_start:test_end],
                    "red",
                    linewidth=2,
                )
                # Add fold label
                ax5.text(
                    test_start + test_size // 2,
                    ax5.get_ylim()[1] * 0.9,
                    f"Fold {fold + 1}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax5.set_xlabel("Time", fontsize=10)
    ax5.set_ylabel("Value", fontsize=10)
    ax5.set_title("4. Purged Forward CV (Leak-Safe)", fontsize=11, fontweight="normal")
    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.3, label="Train"),
        Patch(facecolor="yellow", alpha=0.3, label="Purge Gap"),
        Patch(facecolor="red", alpha=0.3, label="Test"),
    ]
    ax5.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax5.set_xlim(0, n_points)
    return []


def main():
    logger.info("Creating time series windowing animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True
    )
    # Save animation
    output_file = "time_series_windowing.gif"
    logger.info(f"Saving animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=FPS, dpi=100)
    logger.info(f"✓ Animation saved: {output_file}")
    plt.close()
    logger.info("""
    Animation created successfully!
    The animation demonstrates 4 key windowing concepts:
    1. Sliding Window - Shows overlapping windows (common but can have leakage)
    2. Train/Test Split - Simple chronological split (70/30)
    3. Non-Overlapping Windows - Safe windowing without data leakage
    4. Purged Forward CV - Most rigorous approach with purge gaps
    Each method is visualized with:
    - Blue = Training data
    - Red = Test data
    - Yellow = Purge gap (data excluded to prevent leakage)
    - Clear temporal progression showing how windows move
    """)


if __name__ == "__main__":
    main()
