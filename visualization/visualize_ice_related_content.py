"""
Keyword mention visualization for ICE shooting Reddit data.

Mirrors the structure of visualize_emotions.py. Produces three sets of plots:
  1. Per-incident plots
  2. Per-severity plots (averaged across incidents)
  3. Aggregate plot (averaged across all incidents)

Y-axis: proportion of posts/comments mentioning any keyword in the list
X-axis: days relative to incident (-7 to +21)

Usage:
    python visualize_keywords.py
    python visualize_keywords.py --type posts
    python visualize_keywords.py --type comments
    python visualize_keywords.py --smooth 5
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_PATH   = Path("keyword_results.csv")
OUTPUT_DIR = Path("plots/keywords")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW    = (-7, 21)
DAY_RANGE = list(range(WINDOW[0], WINDOW[1] + 1))
BASELINE_DAYS = list(range(WINDOW[0], 0))  # days -7 to -1

INCIDENTS_META = {
    "Brownsville_TX":     {"label": "Brownsville TX",     "date": "2025-05-13", "dead": 0, "injured": 1},
    "Nogales_AZ":         {"label": "Nogales AZ",         "date": "2025-07-01", "dead": 0, "injured": 1},
    "New_York_City_NY_1": {"label": "New York City NY 1", "date": "2025-07-19", "dead": 0, "injured": 2},
    "San_Bernardino_CA":  {"label": "San Bernardino CA",  "date": "2025-08-16", "dead": 0, "injured": 0},
    "El_Paso_TX":         {"label": "El Paso TX",         "date": "2025-09-09", "dead": 0, "injured": 0},
    "Chicago_IL_1":       {"label": "Chicago IL 1",       "date": "2025-09-12", "dead": 1, "injured": 0},
    "Chicago_IL_2":       {"label": "Chicago IL 2",       "date": "2025-10-04", "dead": 0, "injured": 1},
    "Washington_DC_1":    {"label": "Washington DC 1",    "date": "2025-10-17", "dead": 0, "injured": 0},
    "Los_Angeles_CA_1":   {"label": "Los Angeles CA 1",   "date": "2025-10-21", "dead": 0, "injured": 2},
    "Phoenix_AZ":         {"label": "Phoenix AZ",         "date": "2025-10-29", "dead": 0, "injured": 1},
    "Washington_DC_2":    {"label": "Washington DC 2",    "date": "2025-11-13", "dead": 0, "injured": 0},
    "New_York_City_NY_2": {"label": "New York City NY 2", "date": "2025-12-16", "dead": 0, "injured": 0},
    "Saint_Paul_MN":      {"label": "Saint Paul MN",      "date": "2025-12-21", "dead": 0, "injured": 0},
    "Glen_Burnie_MD":     {"label": "Glen Burnie MD",     "date": "2025-12-24", "dead": 0, "injured": 1},
    "Los_Angeles_CA_2":   {"label": "Los Angeles CA 2",   "date": "2025-12-31", "dead": 1, "injured": 0},
    "Minneapolis_MN_1":   {"label": "Minneapolis MN 1",   "date": "2026-01-07", "dead": 1, "injured": 0},
    "Portland_OR":        {"label": "Portland OR",        "date": "2026-01-08", "dead": 0, "injured": 2},
    "Minneapolis_MN_2":   {"label": "Minneapolis MN 2",   "date": "2026-01-14", "dead": 0, "injured": 1},
    "Minneapolis_MN_3":   {"label": "Minneapolis MN 3",   "date": "2026-01-24", "dead": 1, "injured": 0},
}

def severity(meta: dict) -> str:
    if meta["dead"] > 0:
        return "death"
    elif meta["injured"] > 0:
        return "injury"
    else:
        return "no_casualties"

SEVERITY_LABELS = {
    "death":         "Fatal Shootings",
    "injury":        "Non-Fatal Shootings (Injury)",
    "no_casualties": "Shootings with No Casualties",
}

SEVERITY_COLORS = {
    "death":         "#e63946",
    "injury":        "#f4a261",
    "no_casualties": "#457b9d",
}

LINE_COLOR  = "#2b2d42"
FILL_COLOR  = "#8ecae6"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(text_type: str = "both") -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[(df["days_relative"] >= WINDOW[0]) & (df["days_relative"] <= WINDOW[1])]
    if text_type == "posts":
        df = df[df["type"] == "submission"]
    elif text_type == "comments":
        df = df[df["type"] == "comment"]
    return df


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_daily_keyword_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each day in the window, compute the proportion of texts
    where any_keyword == 1.
    Returns DataFrame indexed by days_relative with columns: proportion, n.
    """
    rows = []
    for day in DAY_RANGE:
        day_df = df[df["days_relative"] == day]
        n = len(day_df)
        if n == 0:
            rows.append({"days_relative": day, "proportion": np.nan, "n": 0})
        else:
            rows.append({
                "days_relative": day,
                "proportion":    day_df["any_keyword"].mean(),
                "n":             n,
            })
    return pd.DataFrame(rows).set_index("days_relative")


def smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()




def compute_baseline_keyword(daily: pd.DataFrame) -> float:
    """
    Mean keyword proportion over available pre-event days (n > 0).
    Returns 0.0 if no pre-event data exists.
    """
    available = [
        d for d in BASELINE_DAYS
        if d in daily.index and daily.loc[d, "n"] > 0
    ]
    if not available:
        return 0.0
    return daily.loc[available, "proportion"].mean()


def normalize_daily_keyword(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract pre-event baseline from proportion column.
    Preserves n column unchanged.
    """
    baseline   = compute_baseline_keyword(daily)
    normalized = daily.copy()
    normalized["proportion"] = daily["proportion"] - baseline
    return normalized


def deduplicate_for_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For aggregate and severity plots, each text should only appear once —
    assigned to the incident whose event date it is closest to (minimum
    absolute days_relative). Prevents overlapping Minneapolis windows from
    being counted multiple times.
    """
    dupes  = df[df.duplicated(subset=["id"], keep=False)].copy()
    unique = df[~df.duplicated(subset=["id"], keep=False)].copy()

    if dupes.empty:
        return df

    dupes["abs_days"] = dupes["days_relative"].abs()
    dupes = dupes.sort_values(["id", "abs_days", "days_relative"])
    dupes = dupes.drop_duplicates(subset=["id"], keep="first")
    dupes = dupes.drop(columns=["abs_days"])

    return pd.concat([unique, dupes], ignore_index=True)

# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.dpi":        150,
    })


def plot_single(ax, daily: pd.DataFrame, color: str, smooth_window: int,
                min_n: int = 5, normalized: bool = False):
    x   = daily.index.values
    raw = daily["proportion"]
    smo = smooth(raw, window=smooth_window)

    # Raw unsmoothed faint
    ax.plot(x, raw, color=color, alpha=0.15, linewidth=1.0)
    # Smoothed line
    ax.plot(x, smo, color=color, alpha=0.85, linewidth=1.5, label="Any keyword")

    # Mark low-confidence days
    low_conf = daily["n"] < min_n
    if low_conf.any():
        ax.scatter(
            x[low_conf.values],
            smo.values[low_conf.values],
            color=color, s=18, alpha=0.4, zorder=4,
            label=f"< {min_n} texts that day"
        )

    ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    if normalized:
        ax.axhline(y=0, color="gray", linewidth=0.9, linestyle="-", alpha=0.5, zorder=1)


def format_ax(ax, title: str, n_label: str = "", normalized: bool = False):
    ax.set_xlabel("Days Relative to Incident", fontsize=10)
    if normalized:
        ax.set_ylabel("Change in Keyword Mention Rate from Baseline", fontsize=10)
    else:
        ax.set_ylabel("Proportion Mentioning Any Keyword", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(WINDOW[0] - 0.5, WINDOW[1] + 0.5)
    ticks = sorted(set(
        list(range(0, WINDOW[0] - 1, -2)) +
        list(range(0, WINDOW[1] + 1,  2))
    ))
    ax.set_xticks(ticks)
    if not normalized:
        ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.text(WINDOW[1] + 0.3, 0, "Event →", fontsize=7, color="gray",
            ha="right", va="bottom", transform=ax.get_xaxis_transform())
    if n_label:
        ax.text(0.98, 0.98, n_label, transform=ax.transAxes,
                fontsize=8, color="gray", ha="right", va="top")


# ── Plot set 1: per-incident ───────────────────────────────────────────────────

def plot_per_incident(df: pd.DataFrame, smooth_window: int, normalize: bool = False):
    out_dir = OUTPUT_DIR / "per_incident"
    out_dir.mkdir(exist_ok=True)

    for incident_key, meta in INCIDENTS_META.items():
        inc_df = df[df["incident_key"] == incident_key]
        if inc_df.empty:
            print(f"  No data for {incident_key}, skipping")
            continue

        daily = compute_daily_keyword_proportion(inc_df)
        if normalize:
            daily = normalize_daily_keyword(daily)
        color = SEVERITY_COLORS[severity(meta)]

        fig, ax = plt.subplots(figsize=(11, 4))
        plot_single(ax, daily, color=color, smooth_window=smooth_window, normalized=normalize)
        format_ax(
            ax,
            title=f"{meta['label']}  |  {meta['date']}  |  {severity(meta).replace('_', ' ').title()}",
            n_label=f"n={len(inc_df):,} texts",
            normalized=normalize,
        )
        ax.legend(fontsize=8, framealpha=0.6)
        fig.tight_layout()

        fname = out_dir / f"{incident_key}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot set 2: per-severity ───────────────────────────────────────────────────

def plot_per_severity(df: pd.DataFrame, smooth_window: int, normalize: bool = False):
    out_dir = OUTPUT_DIR / "per_severity"
    out_dir.mkdir(exist_ok=True)

    df = deduplicate_for_aggregate(df)

    severity_groups: dict[str, list[str]] = {"death": [], "injury": [], "no_casualties": []}
    for incident_key, meta in INCIDENTS_META.items():
        severity_groups[severity(meta)].append(incident_key)

    for sev, incident_keys in severity_groups.items():
        sev_df = df[df["incident_key"].isin(incident_keys)]
        if sev_df.empty:
            print(f"  No data for severity={sev}, skipping")
            continue

        # Average daily proportions across incidents
        all_proportions = []
        valid_keys = []
        for ik in incident_keys:
            inc_df = sev_df[sev_df["incident_key"] == ik]
            if inc_df.empty:
                continue
            daily = compute_daily_keyword_proportion(inc_df)
            if normalize:
                daily = normalize_daily_keyword(daily)
            all_proportions.append(daily["proportion"])
            valid_keys.append(ik)

        if not all_proportions:
            continue

        avg_proportion = pd.concat(all_proportions, axis=1).mean(axis=1)
        n_per_day      = sev_df.groupby("days_relative").size().reindex(DAY_RANGE, fill_value=0)

        avg_daily = pd.DataFrame({
            "proportion": avg_proportion.reindex(DAY_RANGE),
            "n":          n_per_day.values,
        }, index=DAY_RANGE)
        avg_daily.index.name = "days_relative"

        color = SEVERITY_COLORS[sev]
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_single(ax, avg_daily, color=color, smooth_window=smooth_window, normalized=normalize)
        format_ax(
            ax,
            title=f"{SEVERITY_LABELS[sev]}  |  {len(valid_keys)} incidents averaged",
            n_label=f"n={len(sev_df):,} texts across {len(valid_keys)} incidents",
            normalized=normalize,
        )
        ax.legend(fontsize=8, framealpha=0.6)
        fig.tight_layout()

        fname = out_dir / f"severity_{sev}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot set 3: aggregate ──────────────────────────────────────────────────────

def plot_aggregate(df: pd.DataFrame, smooth_window: int, normalize: bool = False):
    df = deduplicate_for_aggregate(df)
    all_proportions = []
    for incident_key in INCIDENTS_META:
        inc_df = df[df["incident_key"] == incident_key]
        if inc_df.empty:
            continue
        daily = compute_daily_keyword_proportion(inc_df)
        if normalize:
            daily = normalize_daily_keyword(daily)
        all_proportions.append(daily["proportion"])

    if not all_proportions:
        print("  No data for aggregate plot")
        return

    avg_proportion = pd.concat(all_proportions, axis=1).mean(axis=1)
    n_per_day      = df.groupby("days_relative").size().reindex(DAY_RANGE, fill_value=0)

    avg_daily = pd.DataFrame({
        "proportion": avg_proportion.reindex(DAY_RANGE),
        "n":          n_per_day.values,
    }, index=DAY_RANGE)
    avg_daily.index.name = "days_relative"

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_single(ax, avg_daily, color=LINE_COLOR, smooth_window=smooth_window, normalized=normalize)
    format_ax(
        ax,
        title=f"Aggregate Keyword Mentions  |  All {len(INCIDENTS_META)} Incidents",
        n_label=f"n={len(df):,} texts across {len(INCIDENTS_META)} incidents",
        normalized=normalize,
    )
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()

    fname = OUTPUT_DIR / "aggregate.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",   choices=["both", "posts", "comments"], default="both")
    parser.add_argument("--smooth", type=int, default=3)
    parser.add_argument("--normalize", action="store_true",
                        help="Subtract pre-event baseline from all plots")
    parser.add_argument("--no-per-incident", action="store_true")
    parser.add_argument("--no-per-severity", action="store_true")
    parser.add_argument("--no-aggregate",    action="store_true")
    args = parser.parse_args()

    setup_style()
    print(f"Loading {CSV_PATH}...")
    df = load_data(text_type=args.type)
    print(f"  {len(df):,} rows loaded (type={args.type})")

    if not args.no_per_incident:
        print("\nGenerating per-incident plots...")
        plot_per_incident(df, smooth_window=args.smooth, normalize=args.normalize)

    if not args.no_per_severity:
        print("\nGenerating per-severity plots...")
        plot_per_severity(df, smooth_window=args.smooth, normalize=args.normalize)

    if not args.no_aggregate:
        print("\nGenerating aggregate plot...")
        plot_aggregate(df, smooth_window=args.smooth, normalize=args.normalize)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()