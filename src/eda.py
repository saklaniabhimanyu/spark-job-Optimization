import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import OUTPUT_DIR, TARGET


def plot_eda(df_raw: pd.DataFrame, save_path: str = None, show: bool = True):
    """4-panel EDA figure: execution-time distribution, correlation matrix,
    partitions-vs-time, and skew-vs-time."""
    print("\nGenerating EDA plots …")

    save_path = save_path or os.path.join(OUTPUT_DIR, "eda_plots.png")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Spark Job Execution – Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_raw[TARGET], bins=40, color="#2196F3", edgecolor="white")
    ax1.set_title("Distribution of Execution Time")
    ax1.set_xlabel("Seconds")

    ax2 = fig.add_subplot(gs[0, 1])
    corr = df_raw.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax2, linewidths=0.5)
    ax2.set_title("Feature Correlation Matrix")

    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(df_raw["partitions"], df_raw[TARGET],
                      c=df_raw["num_records"], cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax3, label="Num Records")
    ax3.set_title("Partitions vs Exec Time")
    ax3.set_xlabel("Partitions")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df_raw["skewness"], df_raw[TARGET],
                color="#E91E63", alpha=0.4, s=15)
    ax4.set_title("Data Skew vs Exec Time")
    ax4.set_xlabel("Skew Factor (0–1)")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
