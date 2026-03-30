"""
Forest plot visualization for ITS segmented regression results.

One row per incident x emotion/keyword combination.
Dot = β2 (immediate level change), horizontal line = 95% CI.
Color = severity group. Vertical line at 0 = no effect.

Usage:
    python visualize_its.py
    python visualize_its.py --metric emotions
    python visualize_its.py --metric keywords
    python visualize_its.py --exclude Minneapolis_MN_1
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_CSV = Path("its_results.csv")
OUTPUT_DIR  = Path("plots/its")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_COLORS = {
    "death":         "#e63946",
    "injury":        "#f4a261",
    "no_casualties": "#457b9d",
}

SEVERITY_LABELS = {
    "death":         "Fatal",
    "injury":        "Non-Fatal (Injury)",
    "no_casualties": "No Casualties",
}

EMOTION_LABELS = {
    "emotion_anger":   "Anger",
    "emotion_disgust": "Disgust",
    "emotion_fear":    "Fear",
    "emotion_sadness": "Sadness",
    "emotion_joy":    "Joy",
    "emotion_surprise": "Surprise",
    "emotion_neutral": "Neutral",
    "emotion_all_negative": "All",
    "keyword_mention": "Keyword Mention",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_results(metric: str, exclude: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    if exclude:
        df = df[~df["incident_key"].isin(exclude)]
    if metric == "emotions":
        df = df[df["metric_type"] == "emotion"]
    elif metric == "keywords":
        df = df[df["metric_type"] == "keyword"]
    return df


def setup_style():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "figure.dpi":      150,
    })


# ── Forest plot ────────────────────────────────────────────────────────────────

def plot_forest(df: pd.DataFrame, metric_filter: str, title: str, fname: Path):
    """
    One panel per emotion/keyword metric, rows = incidents.
    """
    metrics = df["metric"].unique()
    n_panels = len(metrics)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.5 * n_panels, max(6, len(df["incident_key"].unique()) * 0.45)),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    # Consistent incident ordering: sort by severity then label
    severity_order = {"death": 0, "injury": 1, "no_casualties": 2}
    incident_order = (
        df[["incident_key", "label", "severity"]]
        .drop_duplicates()
        .assign(sev_rank=lambda x: x["severity"].map(severity_order))
        .sort_values(["sev_rank", "label"])
        ["incident_key"]
        .tolist()
    )
    label_map = df.set_index("incident_key")["label"].to_dict()
    y_positions = {ik: i for i, ik in enumerate(reversed(incident_order))}

    for ax, metric in zip(axes, sorted(metrics)):
        mdf = df[df["metric"] == metric].copy()

        for _, row in mdf.iterrows():
            y    = y_positions.get(row["incident_key"])
            if y is None:
                continue
            color = SEVERITY_COLORS[row["severity"]]
            beta  = row["beta_level"]
            ci_lo = row["ci_level_low"]
            ci_hi = row["ci_level_high"]
            sig   = row["level_significant"]

            # CI line
            ax.hlines(y, ci_lo, ci_hi, color=color, linewidth=1.2, alpha=0.7)

            # Point estimate — filled if significant, open if not
            marker = "D" if sig else "o"
            ms     = 5 if sig else 4
            ax.plot(beta, y,
                    marker=marker, markersize=ms,
                    color=color,
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color,
                    linewidth=0, zorder=4)

        # Zero line
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Y axis labels on leftmost panel only
        if ax == axes[0]:
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels(
                [label_map.get(ik, ik) for ik in reversed(incident_order)],
                fontsize=8
            )
            # Severity group dividers
            prev_sev = None
            for ik in reversed(incident_order):
                sev = df[df["incident_key"] == ik]["severity"].values
                if len(sev) == 0:
                    continue
                sev = sev[0]
                if prev_sev and sev != prev_sev:
                    y_div = y_positions[ik] + 0.5
                    ax.axhline(y_div, color="gray", linewidth=0.5,
                               linestyle=":", alpha=0.5, xmin=0, xmax=1)
                prev_sev = sev
        else:
            ax.set_yticks([])

        ax.set_xlabel("β₂ (Immediate Level Change)", fontsize=8)
        ax.set_title(EMOTION_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_ylim(-0.5, len(incident_order) - 0.5)

        # Light horizontal grid
        for y_val in y_positions.values():
            ax.axhline(y_val, color="gray", linewidth=0.3, alpha=0.2, zorder=0)

    # Legend
    legend_handles = [
        mpatches.Patch(color=SEVERITY_COLORS[s], label=SEVERITY_LABELS[s])
        for s in ["death", "injury", "no_casualties"]
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="D", color="gray", markersize=5,
                   markerfacecolor="gray", linewidth=0, label="Significant (p<0.05)"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=4,
                   markerfacecolor="white", markeredgecolor="gray",
                   linewidth=0, label="Non-significant"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.7)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["emotions", "keywords", "both"],
                        default="both")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Incident keys to exclude e.g. --exclude Minneapolis_MN_1")
    args = parser.parse_args()

    setup_style()

    df = load_results(args.metric, args.exclude)
    if df.empty:
        print("No results found — check that its_results.csv exists.")
        return

    excl_label = f" (excl. {', '.join(args.exclude)})" if args.exclude else ""

    if args.metric in ("emotions", "both"):
        edf = df[df["metric_type"] == "emotion"]
        if not edf.empty:
            print("Generating emotion forest plot...")
            plot_forest(
                edf,
                metric_filter="emotions",
                title=f"ITS Level Change (β₂) by Incident — Negative Emotions{excl_label}",
                fname=OUTPUT_DIR / "forest_emotions.png",
            )

    if args.metric in ("keywords", "both"):
        kdf = df[df["metric_type"] == "keyword"]
        if not kdf.empty:
            print("Generating keyword forest plot...")
            plot_forest(
                kdf,
                metric_filter="keywords",
                title=f"ITS Level Change (β₂) by Incident — Keyword Mentions{excl_label}",
                fname=OUTPUT_DIR / "forest_keywords.png",
            )

    df = pd.read_csv("its_results.csv")
    table = df[["label", "severity", "metric", "pre_mean", 
                "beta_level", "ci_level_low", "ci_level_high", 
                "p_level", "cohens_d", "days_to_return"]].copy()
    table["95% CI"] = table.apply(
        lambda r: f"[{r.ci_level_low:.3f}, {r.ci_level_high:.3f}]", axis=1)
    table["sig"] = table["p_level"].apply(lambda p: "✓" if p < 0.05 else "")
    print(table.to_string(index=False))

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()