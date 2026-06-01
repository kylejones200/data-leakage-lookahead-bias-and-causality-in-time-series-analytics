"""Generated from Jupyter notebook: TS split illustration animation

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


def main():
    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True
    )
    HTML(anim.to_jshtml())


def update(frame):
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.clear()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    progress = frame / N_FRAMES
    current_pos = int(progress * (n_points - window_size))
    ax1.plot(time, data, "black", linewidth=2, alpha=0.7)
    window_x = time[current_pos : current_pos + window_size]
    window_y = data[current_pos : current_pos + window_size]
    ax1.fill_between(window_x, 0, window_y, alpha=0.2, color="blue")
    ax1.plot(window_x, window_y, "blue", linewidth=3)
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.set_title("Time Series Windowing Methods", fontsize=13, fontweight="normal")
    ax1.set_xlim(0, n_points)
    ax2.plot(time, data, "gray", linewidth=1, alpha=0.3)
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
    if current_pos + window_size <= n_points:
        ax2.plot(window_x, window_y, "blue", linewidth=2)
    ax2.set_title("1. Sliding Window (Overlapping)", fontsize=11, fontweight="normal")
    ax2.text(
        0.05,
        0.95,
        "Overlap: YES\n⚠️ Potential leakage",
        transform=ax2.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax2.set_xlim(0, n_points)
    split = int(0.7 * n_points)
    ax3.plot(time[:split], data[:split], "blue", linewidth=2, alpha=0.7)
    ax3.fill_between(time[:split], 0, data[:split], alpha=0.2, color="blue")
    test_reveal = int(progress * (n_points - split))
    if test_reveal > 0:
        test_end = min(split + test_reveal, n_points)
        ax3.plot(time[split:test_end], data[split:test_end], "red", linewidth=2)
        ax3.fill_between(
            time[split:test_end], 0, data[split:test_end], alpha=0.2, color="red"
        )
    ax3.axvline(split, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax3.set_title("2. Train/Test Split (70/30)", fontsize=11, fontweight="normal")
    ax3.legend(["Train", "Test"], loc="upper left", fontsize=9)
    ax3.set_xlim(0, n_points)
    ax4.plot(time, data, "gray", linewidth=1, alpha=0.3)
    n_windows = int(progress * 4) + 1
    colors = ["blue", "green", "orange", "purple"]
    for i in range(n_windows):
        start, end = (i * window_size, min((i + 1) * window_size, n_points))
        if end <= n_points:
            ax4.fill_between(
                time[start:end], 0, data[start:end], alpha=0.3, color=colors[i % 4]
            )
            ax4.plot(time[start:end], data[start:end], colors[i % 4], linewidth=2)
            ax4.text(
                (start + end) // 2,
                ax4.get_ylim()[1] * 0.9,
                f"W{i + 1}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    ax4.set_title("3. Non-Overlapping Windows", fontsize=11, fontweight="normal")
    ax4.text(
        0.05,
        0.95,
        "No overlap\n✅ Leak-safe",
        transform=ax4.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )
    ax4.set_xlim(0, n_points)
    ax5.plot(time, data, "gray", linewidth=1, alpha=0.3)
    n_folds = int(progress * 3) + 1
    for fold in range(n_folds):
        train_end = 30 + fold * 15
        if train_end < n_points - purge_gap - test_size:
            ax5.fill_between(
                time[:train_end], 0, data[:train_end], alpha=0.2, color="blue"
            )
            ax5.plot(time[:train_end], data[:train_end], "blue", linewidth=1, alpha=0.5)
            purge_start, purge_end = (train_end, train_end + purge_gap)
            if purge_end < n_points:
                ax5.axvspan(purge_start, purge_end, alpha=0.2, color="yellow", zorder=0)
            test_start = train_end + purge_gap
            test_end_idx = min(test_start + test_size, n_points)
            if test_end_idx <= n_points:
                ax5.fill_between(
                    time[test_start:test_end_idx],
                    0,
                    data[test_start:test_end_idx],
                    alpha=0.4,
                    color="red",
                )
                ax5.plot(
                    time[test_start:test_end_idx],
                    data[test_start:test_end_idx],
                    "red",
                    linewidth=2,
                )
                ax5.text(
                    test_start + test_size // 2,
                    ax5.get_ylim()[1] * 0.9,
                    f"F{fold + 1}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )
    ax5.set_title("4. Purged Forward CV", fontsize=11, fontweight="normal")
    legend_elements = [
        Patch(facecolor="blue", alpha=0.3, label="Train"),
        Patch(facecolor="yellow", alpha=0.3, label="Purge"),
        Patch(facecolor="red", alpha=0.3, label="Test"),
    ]
    ax5.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax5.set_xlim(0, n_points)
    return []


def main() -> None:
    "\nImproved Time Series Windowing Animation\n\nThis creates a clear, educational visualization showing 4 key windowing methods:\n1. Sliding Window (overlapping)\n2. Train/Test Split (simple chronological)\n3. Non-Overlapping Windows (leak-safe)\n4. Purged Forward CV (most rigorous)\n\nUses actual time series data and clear color coding.\n"
    np.random.seed(42)
    FPS = 5
    DURATION = 12
    FPS * DURATION
    n_points = 100
    time = np.arange(n_points)
    trend = 0.5 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 20)
    noise = np.random.normal(0, 2, n_points)
    trend + seasonal + noise + 50
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    main()


if __name__ == "__main__":
    main()
