"""
eda.py
------
Full Exploratory Data Analysis: distributions, correlations, churn drivers.
Saves plots to assets/ directory.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

PALETTE = {"churned": "#E63946", "retained": "#2A9D8F"}
BG = "#0F1117"
FG = "#EAEAEA"
ACCENT = "#E63946"
GRID_C = "#2A2D3E"

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID_C,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "text.color": FG,
        "grid.color": GRID_C,
        "grid.linewidth": 0.5,
        "font.family": "DejaVu Sans",
        "axes.titlecolor": FG,
    })


def run_eda(df: pd.DataFrame, out_dir: str):
    """Run full EDA and save all plots."""
    os.makedirs(out_dir, exist_ok=True)
    _style()

    _plot_churn_distribution(df, out_dir)
    _plot_missing_values(df, out_dir)
    _plot_numeric_distributions(df, out_dir)
    _plot_categorical_churn(df, out_dir)
    _plot_correlation_heatmap(df, out_dir)
    _plot_tenure_vs_charges(df, out_dir)
    print(f"✅ EDA plots saved to {out_dir}")


def _plot_churn_distribution(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)

    counts = df["Churn"].value_counts()
    colors = [PALETTE["retained"], PALETTE["churned"]]
    axes[0].pie(counts, labels=["Retained", "Churned"], colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"color": FG, "fontsize": 12},
                wedgeprops={"edgecolor": BG, "linewidth": 2})
    axes[0].set_title("Overall Churn Distribution", fontsize=14, fontweight="bold")

    churn_labels = ["Retained", "Churned"]
    bars = axes[1].bar(churn_labels, counts.values, color=colors,
                       width=0.5, edgecolor=BG, linewidth=1.5)
    for bar, v in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f"{v:,}", ha="center", va="bottom", fontsize=12, color=FG)
    axes[1].set_title("Churn Count", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Customers", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_facecolor(BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_churn_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_missing_values(df, out_dir):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    bars = ax.barh(miss.index, miss.values, color=ACCENT, edgecolor=BG)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", va="center", fontsize=10, color=FG)
    ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Missing Count")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_missing_values.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_numeric_distributions(df, out_dir):
    num_cols = ["Tenure", "Age", "MonthlyCharges", "TotalCharges", "NumComplaints", "SupportCalls"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.set_facecolor(BG)
        for churn_val, label, color in [(0, "Retained", PALETTE["retained"]),
                                         (1, "Churned", PALETTE["churned"])]:
            data = df[df["Churn"] == churn_val][col].dropna()
            ax.hist(data, bins=30, alpha=0.65, color=color, label=label, density=True)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    plt.suptitle("Numeric Feature Distributions by Churn Status", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_numeric_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_categorical_churn(df, out_dir):
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ax.set_facecolor(BG)
        churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100
        colors = [plt.cm.RdYlGn_r(v / 100) for v in churn_rate.values]
        bars = ax.bar(churn_rate.index, churn_rate.values, color=colors, edgecolor=BG)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10, color=FG)
        ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, churn_rate.max() * 1.2)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Churn Rate by Key Categorical Features", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_categorical_churn.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_correlation_heatmap(df, out_dir):
    num_df = df[["Tenure", "Age", "MonthlyCharges", "TotalCharges",
                 "NumComplaints", "SupportCalls", "SeniorCitizen", "Churn"]].dropna()
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", linewidths=0.5, linecolor=BG,
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9, "color": FG})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_tenure_vs_charges(df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for churn_val, label, color, alpha in [(0, "Retained", PALETTE["retained"], 0.4),
                                            (1, "Churned", PALETTE["churned"], 0.7)]:
        subset = df[df["Churn"] == churn_val].dropna(subset=["Tenure", "MonthlyCharges"])
        ax.scatter(subset["Tenure"], subset["MonthlyCharges"],
                   c=color, alpha=alpha, s=15, label=label)
    ax.set_xlabel("Tenure (months)", fontsize=12)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax.set_title("Tenure vs Monthly Charges — Churn Segmentation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_tenure_vs_charges.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/customers.csv"
    df = pd.read_csv(csv_path)
    run_eda(df, "assets")
