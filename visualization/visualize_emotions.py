"""
Emotion visualization for ICE shooting Reddit data.

Produces three sets of plots:
  1. Per-incident plots — absolute values, one plot per incident
  2. Per-severity plots — averaged across incidents grouped by outcome
  3. Aggregate plot — averaged across all incidents

For severity and aggregate plots, baseline normalization is available via
--normalize. This subtracts each incident's pre-event baseline (mean of
available days -7 to -1) before averaging, so the y-axis shows *change*
from that incident's own baseline rather than absolute levels. This corrects
for subreddits having different baseline emotional tones.

Y-axis: mean emotion score (or change from baseline if --normalize)
X-axis: days relative to incident (-7 to +21)

Usage:
    python visualize_emotions.py
    python visualize_emotions.py --normalize
    python visualize_emotions.py --metric dominant
    python visualize_emotions.py --type posts
    python visualize_emotions.py --smooth 5
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_PATH   = Path("emotion_results.csv")
OUTPUT_DIR = Path("plots/emotions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS          = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]

EMOTION_COLORS = {
    "anger":    "#e63946",
    "disgust":  "#6a994e",
    "fear":     "#9b5de5",
    "joy":      "#f9c74f",
    "neutral":  "#8ecae6",
    "sadness":  "#457b9d",
    "surprise": "#f4a261",
}

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

WINDOW    = (-7, 21)
DAY_RANGE = list(range(WINDOW[0], WINDOW[1] + 1))
BASELINE_DAYS = list(range(WINDOW[0], 0))  # days -7 to -1

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(text_type: str = "both") -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[(df["days_relative"] >= WINDOW[0]) & (df["days_relative"] <= WINDOW[1])]
    if text_type == "posts":
        df = df[df["type"] == "submission"]
    elif text_type == "comments":
        df = df[df["type"] == "comment"]
    return df



def deduplicate_for_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For aggregate and severity plots, each text should only appear once —
    assigned to the incident whose event date it is closest to (minimum
    absolute days_relative). This prevents texts in overlapping windows
    (e.g. Minneapolis MN_1/2/3) from being counted multiple times.

    Ties are broken by keeping the earlier incident (lower days_relative
    value favoured, i.e. the text is more "post-event" for the earlier
    incident than "pre-event" for the later one).

    Per-incident plots are NOT affected — they use the full df directly.
    """
    # Only rows with duplicate ids need resolving
    dupes = df[df.duplicated(subset=["id"], keep=False)].copy()
    unique = df[~df.duplicated(subset=["id"], keep=False)].copy()

    if dupes.empty:
        return df

    # For each id keep the row with smallest abs(days_relative),
    # breaking ties by keeping the smallest days_relative (earlier event)
    dupes["abs_days"] = dupes["days_relative"].abs()
    dupes = dupes.sort_values(["id", "abs_days", "days_relative"])
    dupes = dupes.drop_duplicates(subset=["id"], keep="first")
    dupes = dupes.drop(columns=["abs_days"])

    result = pd.concat([unique, dupes], ignore_index=True)
    return result

# ── Metric computation ─────────────────────────────────────────────────────────

def compute_daily_proportions(df: pd.DataFrame, metric: str = "mean") -> pd.DataFrame:
    """
    For each day in the window, compute either:
      - mean: mean score for each emotion (default)
      - dominant: fraction of texts where each emotion is dominant (score >= 0.5)

    Returns DataFrame indexed by days_relative with emotion columns + n.
    """
    rows = []
    for day in DAY_RANGE:
        day_df = df[df["days_relative"] == day]
        n = len(day_df)
        row = {"days_relative": day, "n": n}
        if n == 0:
            for e in NEGATIVE_EMOTIONS:
                row[e] = np.nan
        elif metric == "dominant":
            for e in NEGATIVE_EMOTIONS:
                row[e] = (day_df["dominant_emotions"].str.contains(e, na=False)).sum() / n
        else:
            for e in NEGATIVE_EMOTIONS:
                row[e] = day_df[e].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("days_relative")


def compute_baseline(daily: pd.DataFrame) -> pd.Series:
    """
    Mean emotion scores over available pre-event days (days < 0, n > 0).
    Returns zeros if no pre-event data exists — those incidents contribute
    no baseline shift but still appear in the post-event average.
    """
    available = [
        d for d in BASELINE_DAYS
        if d in daily.index and daily.loc[d, "n"] > 0
    ]
    if not available:
        return pd.Series({e: 0.0 for e in NEGATIVE_EMOTIONS})
    return daily.loc[available, NEGATIVE_EMOTIONS].mean()


def normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract each incident's pre-event baseline from all daily values.
    Preserves the n column. Post-event values near 0 mean no change from
    baseline; positive = elevated; negative = suppressed.
    """
    baseline  = compute_baseline(daily)
    normalized = daily.copy()
    for e in NEGATIVE_EMOTIONS:
        normalized[e] = daily[e] - baseline[e]
    return normalized


def smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ── Plotting helpers ───────────────────────────────────────────────────────────

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


def add_event_line(ax):
    ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)


def add_zero_line(ax):
    ax.axhline(y=0, color="gray", linewidth=0.9, linestyle="-", alpha=0.5, zorder=1)


def plot_emotion_lines(ax, daily: pd.DataFrame, smooth_window: int = 3,
                       min_n: int = 5, alpha: float = 0.85):
    x = daily.index.values
    for emotion in NEGATIVE_EMOTIONS:
        y     = smooth(daily[emotion], window=smooth_window)
        color = EMOTION_COLORS[emotion]
        ax.plot(x, daily[emotion], color=color, alpha=0.15, linewidth=1.0)
        ax.plot(x, y, color=color, alpha=alpha, linewidth=1.5, label=emotion)
        low_conf = daily["n"] < min_n
        if low_conf.any():
            ax.scatter(
                x[low_conf.values],
                y.values[low_conf.values],
                color=color, s=12, alpha=0.4, zorder=3,
            )


def get_xticks():
    return sorted(set(
        list(range(0, WINDOW[0] - 1, -2)) +
        list(range(0, WINDOW[1] + 1,  2))
    ))


def format_ax(ax, title: str, metric: str, normalized: bool = False, n_label: str = ""):
    add_event_line(ax)
    if normalized:
        add_zero_line(ax)

    ax.set_xlabel("Days Relative to Incident", fontsize=10)

    if normalized:
        ylabel = "Change in Emotion Score from Pre-Event Baseline"
    elif metric == "dominant":
        ylabel = "Proportion with Dominant Emotion"
    else:
        ylabel = "Mean Emotion Score"

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(WINDOW[0] - 0.5, WINDOW[1] + 0.5)
    ax.set_xticks(get_xticks())
    
    if metric == "dominant" and not normalized:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    if n_label:
        ax.text(0.98, 0.02, n_label, transform=ax.transAxes,
                fontsize=8, color="gray", ha="right", va="bottom")


def build_legend(ax):
    handles = [
        Line2D([0], [0], color=EMOTION_COLORS[e], linewidth=2, label=e.capitalize())
        for e in NEGATIVE_EMOTIONS
    ]
    handles.append(
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Event day")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.7, ncol=2)


# ── Plot set 1: per-incident ─────────────────────────────────────────────────────

def plot_per_incident(df: pd.DataFrame, metric: str, smooth_window: int, normalize: bool = False):
    out_dir = OUTPUT_DIR / "per_incident"
    out_dir.mkdir(exist_ok=True)

    for incident_key, meta in INCIDENTS_META.items():
        inc_df = df[df["incident_key"] == incident_key]
        if inc_df.empty:
            print(f"  No data for {incident_key}, skipping")
            continue

        daily = compute_daily_proportions(inc_df, metric=metric)
        if normalize:
            daily = normalize_daily(daily)

        fig, ax = plt.subplots(figsize=(11, 5))
        plot_emotion_lines(ax, daily, smooth_window=smooth_window)
        format_ax(
            ax,
            title=f"{meta['label']}  |  {meta['date']}  |  Severity: {severity(meta).replace('_', ' ')}",
            metric=metric,
            normalized=normalize,
            n_label=f"n={len(inc_df):,} texts",
        )

        build_legend(ax)
        fig.tight_layout()

        fname = out_dir / f"{incident_key}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot set 2: per-severity ───────────────────────────────────────────────────

def plot_per_severity(df: pd.DataFrame, metric: str, smooth_window: int,
                      normalize: bool = False):
    out_dir = OUTPUT_DIR / "per_severity"
    out_dir.mkdir(exist_ok=True)

    df = deduplicate_for_aggregate(df)

    severity_groups: dict[str, list[str]] = {
        "death": [], "injury": [], "no_casualties": []
    }
    for incident_key, meta in INCIDENTS_META.items():
        severity_groups[severity(meta)].append(incident_key)

    for sev, incident_keys in severity_groups.items():
        sev_df = df[df["incident_key"].isin(incident_keys)]
        if sev_df.empty:
            continue

        all_dailies = []
        for ik in incident_keys:
            inc_df = sev_df[sev_df["incident_key"] == ik]
            if inc_df.empty:
                continue
            daily = compute_daily_proportions(inc_df, metric=metric)
            if normalize:
                daily = normalize_daily(daily)
            all_dailies.append(daily[NEGATIVE_EMOTIONS])

        if not all_dailies:
            continue

        avg_daily = pd.concat(all_dailies).groupby(level=0).mean()
        avg_daily = avg_daily.reindex(DAY_RANGE)
        n_per_day = sev_df.groupby("days_relative").size().reindex(DAY_RANGE, fill_value=0)
        avg_daily["n"] = n_per_day.values

        n_incidents = len(all_dailies)
        norm_tag    = "_normalized" if normalize else ""
        norm_title  = "  |  Baseline-Normalized" if normalize else ""

        fig, ax = plt.subplots(figsize=(11, 5))
        plot_emotion_lines(ax, avg_daily, smooth_window=smooth_window)
        format_ax(
            ax,
            title=f"{SEVERITY_LABELS[sev]}  |  {n_incidents} incidents{norm_title}",
            metric=metric,
            normalized=normalize,
            n_label=f"n={len(sev_df):,} texts across {n_incidents} incidents",
        )
        build_legend(ax)
        fig.tight_layout()

        fname = out_dir / f"severity_{sev}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot set 3: aggregate ──────────────────────────────────────────────────────

def plot_aggregate(df: pd.DataFrame, metric: str, smooth_window: int,
                   normalize: bool = False):
    df = deduplicate_for_aggregate(df)
    all_dailies = []
    for incident_key in INCIDENTS_META:
        inc_df = df[df["incident_key"] == incident_key]
        if inc_df.empty:
            continue
        daily = compute_daily_proportions(inc_df, metric=metric)
        if normalize:
            daily = normalize_daily(daily)
        all_dailies.append(daily[NEGATIVE_EMOTIONS])

    if not all_dailies:
        print("  No data for aggregate plot")
        return

    avg_daily = pd.concat(all_dailies).groupby(level=0).mean()
    avg_daily = avg_daily.reindex(DAY_RANGE)
    n_per_day = df.groupby("days_relative").size().reindex(DAY_RANGE, fill_value=0)
    avg_daily["n"] = n_per_day.values

    norm_tag   = "_normalized" if normalize else ""
    norm_title = "  |  Baseline-Normalized" if normalize else ""

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_emotion_lines(ax, avg_daily, smooth_window=smooth_window)
    format_ax(
        ax,
        title=f"Aggregate Emotional Response  |  All {len(INCIDENTS_META)} Incidents{norm_title}",
        metric=metric,
        normalized=normalize,
        n_label=f"n={len(df):,} texts across {len(INCIDENTS_META)} incidents",
    )
    build_legend(ax)
    fig.tight_layout()

    fname = OUTPUT_DIR / "aggregate.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric",   choices=["dominant", "mean"], default="dominant")
    parser.add_argument("--type",     choices=["both", "posts", "comments"], default="both")
    parser.add_argument("--smooth",   type=int, default=3)
    parser.add_argument("--normalize", action="store_true",
                        help="Subtract pre-event baseline from severity and aggregate plots")
    parser.add_argument("--no-per-incident", action="store_true")
    parser.add_argument("--no-per-severity", action="store_true")
    parser.add_argument("--no-aggregate",    action="store_true")
    args = parser.parse_args()

    setup_style()
    print(f"Loading {CSV_PATH}...")
    df = load_data(text_type=args.type)
    print(f"  {len(df):,} rows loaded (type={args.type}, metric={args.metric}, normalize={args.normalize})")

    if not args.no_per_incident:
        print("\nGenerating per-incident plots...")
        plot_per_incident(df, metric=args.metric, smooth_window=args.smooth,
                          normalize=args.normalize)

    if not args.no_per_severity:
        print("\nGenerating per-severity plots...")
        plot_per_severity(df, metric=args.metric, smooth_window=args.smooth,
                          normalize=args.normalize)

    if not args.no_aggregate:
        print("\nGenerating aggregate plot...")
        plot_aggregate(df, metric=args.metric, smooth_window=args.smooth,
                       normalize=args.normalize)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()