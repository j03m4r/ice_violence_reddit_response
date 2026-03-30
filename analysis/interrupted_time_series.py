"""
Interrupted Time Series (ITS) Segmented Regression Analysis
for ICE shooting Reddit data.

For each incident x metric combination, fits a segmented regression:

    Y = β0 + β1·time + β2·post + β3·time_after + ε

Where:
    time       = days_relative (continuous, -7 to +21)
    post       = 0 before day 0, 1 on and after day 0
    time_after = 0 before day 0, days_relative on and after day 0
                 (i.e. 0,1,2,3... counting from event day)

Coefficients:
    β0 = pre-event intercept (baseline level at day -7)
    β1 = pre-event slope (trend per day before event)
    β2 = immediate level change at event (the jump)
    β3 = slope change after event (positive = sustained rise,
         negative = decay back toward baseline)

Return-to-baseline: the day post-event when the trajectory
crosses back to the pre-event counterfactual, computed as:
    days_to_return = -β2 / β3  (if β3 opposes β2)

Outputs:
    its_results.csv  — one row per incident x metric
    its_summary.csv  — aggregated by severity group

Usage:
    python its_analysis.py
    python its_analysis.py --metric emotions
    python its_analysis.py --metric keywords
    python its_analysis.py --min-n 20
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

EMOTION_CSV  = Path("emotion_results.csv")
KEYWORD_CSV  = Path("keyword_results.csv")
RESULTS_CSV  = Path("its_results.csv")
SUMMARY_CSV  = Path("its_summary.csv")

WINDOW    = (-7, 21)
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
ALPHA     = 0.05  # significance threshold

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


# Minneapolis combined meta — used when --combine-minneapolis is set.
# Treats all three events as one sustained incident with MN_1 as breakpoint.
MINNEAPOLIS_COMBINED_KEY  = "Minneapolis_MN_combined"
MINNEAPOLIS_COMBINED_META = {
    "label":   "Minneapolis MN (Combined)",
    "date":    "2026-01-07",   # MN_1 event date as breakpoint
    "dead":    3,              # total across all three events
    "injured": 1,
}
MINNEAPOLIS_KEYS = {"Minneapolis_MN_1", "Minneapolis_MN_2", "Minneapolis_MN_3"}

def severity(meta: dict) -> str:
    if meta["dead"] > 0:
        return "death"
    elif meta["injured"] > 0:
        return "injury"
    else:
        return "no_casualties"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_emotion_data() -> pd.DataFrame:
    df = pd.read_csv(EMOTION_CSV)
    return df[(df["days_relative"] >= WINDOW[0]) & (df["days_relative"] <= WINDOW[1])]


def load_keyword_data() -> pd.DataFrame:
    df = pd.read_csv(KEYWORD_CSV)
    return df[(df["days_relative"] >= WINDOW[0]) & (df["days_relative"] <= WINDOW[1])]


def deduplicate(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Assign each text to its nearest incident to avoid double-counting Minneapolis."""
    dupes  = df[df.duplicated(subset=[id_col], keep=False)].copy()
    unique = df[~df.duplicated(subset=[id_col], keep=False)].copy()
    if dupes.empty:
        return df
    dupes["abs_days"] = dupes["days_relative"].abs()
    dupes = dupes.sort_values([id_col, "abs_days", "days_relative"])
    dupes = dupes.drop_duplicates(subset=[id_col], keep="first")
    dupes = dupes.drop(columns=["abs_days"])
    return pd.concat([unique, dupes], ignore_index=True)



def combine_minneapolis(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Replace MN_1/2/3 rows with a single combined Minneapolis entry.
    - Keeps only unique texts (deduplicated by id_col)
    - Recomputes days_relative relative to MN_1 event date (2026-01-07)
    - Assigns incident_key = Minneapolis_MN_combined
    """
    from datetime import datetime, timezone
    mn1_dt = datetime(2026, 1, 7, tzinfo=timezone.utc)

    mpls = df[df["incident_key"].isin(MINNEAPOLIS_KEYS)].copy()
    other = df[~df["incident_key"].isin(MINNEAPOLIS_KEYS)].copy()

    if mpls.empty:
        return df

    # Deduplicate — keep one row per unique text
    mpls = mpls.sort_values([id_col, "created_utc"])
    mpls = mpls.drop_duplicates(subset=[id_col], keep="first")

    # Recompute days_relative from MN_1 breakpoint
    mpls["days_relative"] = mpls["created_utc"].apply(
        lambda utc: (datetime.fromtimestamp(utc, tz=timezone.utc) - mn1_dt).days
    )
    mpls["incident_key"]  = MINNEAPOLIS_COMBINED_KEY
    mpls["incident_date"] = "2026-01-07"

    combined = pd.concat([other, mpls], ignore_index=True)
    return combined

# ── Daily aggregation ──────────────────────────────────────────────────────────

def aggregate_daily_emotion(inc_df: pd.DataFrame, emotion: str,
                               use_dominant: bool = False) -> pd.DataFrame:
    """
    Mean emotion score per day (default), or proportion of texts
    where the emotion is dominant (score >= 0.5) if use_dominant=True.
    """
    if emotion == "all_negative":
        if use_dominant:
            inc_df["y"] = inc_df["dominant_emotions"].str.contains(
                "|".join(NEGATIVE_EMOTIONS), na=False
            ).astype(float)
        else:
            inc_df["y"] = inc_df[NEGATIVE_EMOTIONS].mean(axis=1)
    else:
        if use_dominant:
            inc_df["y"] = inc_df["dominant_emotions"].str.contains(emotion, na=False).astype(float)
        else:
            inc_df = inc_df.rename(columns={emotion: "y"})

    daily = (
        inc_df.groupby("days_relative")["y"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "y", "count": "n"})
        .reindex(range(WINDOW[0], WINDOW[1] + 1))
    )
    return daily.dropna(subset=["y"])


def aggregate_daily_keyword(inc_df: pd.DataFrame) -> pd.DataFrame:
    """Proportion of texts mentioning any keyword per day."""
    daily = (
        inc_df.groupby("days_relative")["any_keyword"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "y", "count": "n"})
        .reindex(range(WINDOW[0], WINDOW[1] + 1))
    )
    return daily.dropna(subset=["y"])


# ── ITS regression ─────────────────────────────────────────────────────────────

def build_its_matrix(days: np.ndarray) -> pd.DataFrame:
    """
    Build the design matrix for segmented regression.
    Columns: intercept, time, post, time_after
    """
    post       = (days >= 0).astype(int)
    time_after = np.where(days >= 0, days, 0)
    X = pd.DataFrame({
        "intercept":  1,
        "time":       days,
        "post":       post,
        "time_after": time_after,
    })
    return X


def fit_its(daily: pd.DataFrame, min_n: int = 5) -> dict | None:
    """
    Fit segmented regression to daily aggregated data.
    Returns dict of results or None if insufficient data.
    """
    # Require data on both sides of the breakpoint
    pre  = daily[daily.index < 0]
    post = daily[daily.index >= 0]

    if len(pre) < 2 or len(post) < 3:
        return None

    # Weight by sqrt(n) to give more influence to days with more data
    # (avoids low-volume days with noisy estimates dominating)
    days    = daily.index.values.astype(float)
    y       = daily["y"].values
    weights = np.sqrt(daily["n"].fillna(1).values)

    X = build_its_matrix(days)

    try:
        model  = sm.WLS(y, X, weights=weights).fit()
    except Exception:
        return None

    b0, b1, b2, b3 = model.params
    p0, p1, p2, p3 = model.pvalues
    ci = model.conf_int()

    # Return-to-baseline: day post-event when post trajectory
    # crosses back to pre-event counterfactual
    # Pre-event counterfactual at day t: b0 + b1*t
    # Post-event trajectory at day t:    b0 + b1*t + b2 + b3*t
    # They are equal when b2 + b3*t = 0 → t = -b2/b3
    if abs(b3) > 1e-10 and np.sign(b2) != np.sign(b3):
        days_to_return = round(-b2 / b3, 1)
    else:
        days_to_return = np.nan  # effect is permanent or no effect

    # Pre-event mean (actual)
    pre_mean = pre["y"].mean()

    # Effect size: Cohen's d approximation using residual std
    residual_std = np.std(model.resid)
    cohens_d = b2 / residual_std if residual_std > 0 else np.nan

    return {
        "pre_mean":         round(pre_mean, 4),
        "beta_intercept":   round(b0, 4),
        "beta_time":        round(b1, 6),        # pre-event slope
        "beta_level":       round(b2, 4),        # immediate level change
        "beta_slope":       round(b3, 6),        # post-event slope change
        "p_time":           round(p1, 4),
        "p_level":          round(p2, 4),        # key: is the jump significant?
        "p_slope":          round(p3, 4),
        "ci_level_low":     round(ci.loc["post", 0], 4),
        "ci_level_high":    round(ci.loc["post", 1], 4),
        "level_significant": p2 < ALPHA,
        "slope_significant": p3 < ALPHA,
        "days_to_return":   days_to_return,
        "cohens_d":         round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
        "r_squared":        round(model.rsquared, 4),
        "n_days":           len(daily),
        "n_pre_days":       len(pre),
        "n_post_days":      len(post),
    }


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_analysis(metric: str, min_n: int, use_dominant: bool = False,
                 combine_mpls: bool = False) -> pd.DataFrame:
    rows = []

    if metric in ("emotions", "both"):
        print("Loading emotion data...")
        edf = load_emotion_data()
        edf = deduplicate(edf, "id")
        if combine_mpls:
            edf = combine_minneapolis(edf, "id")

        for incident_key, meta in INCIDENTS_META.items():
            inc_df = edf[edf["incident_key"] == incident_key]
            if inc_df.empty:
                continue

            # Filter days with sufficient data
            day_counts = inc_df.groupby("days_relative").size()

            for emotion in NEGATIVE_EMOTIONS:
                daily = aggregate_daily_emotion(inc_df, emotion, use_dominant=use_dominant)
                # Drop days below min_n threshold
                daily = daily[daily["n"] >= min_n]

                result = fit_its(daily, min_n=min_n)
                if result is None:
                    print(f"  SKIP {incident_key} / {emotion} — insufficient data")
                    continue

                rows.append({
                    "incident_key":  incident_key,
                    "label":         meta["label"],
                    "severity":      severity(meta),
                    "dead":          meta["dead"],
                    "injured":       meta["injured"],
                    "metric":        f"emotion_{emotion}",
                    "metric_type":   "emotion",
                    "aggregation":   "dominant" if use_dominant else "mean",
                    **result,
                })
            
            emotion = "all_negative"
            daily = aggregate_daily_emotion(inc_df, emotion, use_dominant=use_dominant)
            daily = daily[daily["n"] >= min_n]

            result = fit_its(daily, min_n=min_n)
            if result is None:
                print(f"  SKIP {incident_key} / {emotion} — insufficient data")
                continue

            rows.append({
                "incident_key":  incident_key,
                "label":         meta["label"],
                "severity":      severity(meta),
                "dead":          meta["dead"],
                "injured":       meta["injured"],
                "metric":        f"emotion_{emotion}",
                "metric_type":   "emotion",
                "aggregation":   "dominant" if use_dominant else "mean",
                **result,
            })

        # Handle combined Minneapolis if requested
        if combine_mpls and MINNEAPOLIS_COMBINED_KEY in edf["incident_key"].values:
            inc_df = edf[edf["incident_key"] == MINNEAPOLIS_COMBINED_KEY]
            meta   = MINNEAPOLIS_COMBINED_META
            for emotion in NEGATIVE_EMOTIONS:
                daily = aggregate_daily_emotion(inc_df, emotion, use_dominant=use_dominant)
                daily = daily[daily["n"] >= min_n]
                result = fit_its(daily, min_n=min_n)
                if result is None:
                    print(f"  SKIP Minneapolis_combined / {emotion} — insufficient data")
                    continue
                rows.append({
                    "incident_key":  MINNEAPOLIS_COMBINED_KEY,
                    "label":         meta["label"],
                    "severity":      "death",
                    "dead":          meta["dead"],
                    "injured":       meta["injured"],
                    "metric":        f"emotion_{emotion}",
                    "metric_type":   "emotion",
                    "aggregation":   "dominant" if use_dominant else "mean",
                    **result,
                }) 

            emotion = "all_negative"
            daily = aggregate_daily_emotion(inc_df, emotion, use_dominant=use_dominant)
            daily = daily[daily["n"] >= min_n]

            result = fit_its(daily, min_n=min_n)
            if result:
                rows.append({
                    "incident_key":  MINNEAPOLIS_COMBINED_KEY,
                    "label":         meta["label"],
                    "severity":      severity(meta),
                    "dead":          meta["dead"],
                    "injured":       meta["injured"],
                    "metric":        f"emotion_{emotion}",
                    "metric_type":   "emotion",
                    "aggregation":   "dominant" if use_dominant else "mean",
                    **result,
                })
            else:
                print(f"  SKIP {MINNEAPOLIS_COMBINED_KEY} / {emotion} — insufficient data")

        print(f"  Emotion rows: {len(rows)}")

    if metric in ("keywords", "both"):
        print("Loading keyword data...")
        kdf = load_keyword_data()
        kdf = deduplicate(kdf, "id")
        if combine_mpls:
            kdf = combine_minneapolis(kdf, "id")

        keyword_start = len(rows)
        for incident_key, meta in INCIDENTS_META.items():
            inc_df = kdf[kdf["incident_key"] == incident_key]
            if inc_df.empty:
                continue

            daily = aggregate_daily_keyword(inc_df)
            daily = daily[daily["n"] >= min_n]

            result = fit_its(daily, min_n=min_n)
            if result is None:
                print(f"  SKIP {incident_key} / keywords — insufficient data")
                continue

            rows.append({
                "incident_key":  incident_key,
                "label":         meta["label"],
                "severity":      severity(meta),
                "dead":          meta["dead"],
                "injured":       meta["injured"],
                "metric":        "keyword_mention",
                "metric_type":   "keyword",
                "aggregation":   "proportion",
                **result,
            })


        # Handle combined Minneapolis keywords if requested
        if combine_mpls and MINNEAPOLIS_COMBINED_KEY in kdf["incident_key"].values:
            inc_df = kdf[kdf["incident_key"] == MINNEAPOLIS_COMBINED_KEY]
            meta   = MINNEAPOLIS_COMBINED_META
            daily  = aggregate_daily_keyword(inc_df)
            daily  = daily[daily["n"] >= min_n]
            result = fit_its(daily, min_n=min_n)
            if result is None:
                print(f"  SKIP Minneapolis_combined / keywords — insufficient data")
            else:
                rows.append({
                    "incident_key":  MINNEAPOLIS_COMBINED_KEY,
                    "label":         meta["label"],
                    "severity":      "death",
                    "dead":          meta["dead"],
                    "injured":       meta["injured"],
                    "metric":        "keyword_mention",
                    "metric_type":   "keyword",
                    "aggregation":   "proportion",
                    **result,
                })

        print(f"  Keyword rows: {len(rows) - keyword_start}")

    return pd.DataFrame(rows)


def summarize_by_severity(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ITS coefficients by severity group and metric.
    Reports mean β2 (level change), proportion significant, mean days_to_return.
    """
    summary_rows = []

    for (sev, metric), grp in results.groupby(["severity", "metric"]):
        summary_rows.append({
            "severity":              sev,
            "metric":                metric,
            "n_incidents":           len(grp),
            "mean_pre_mean":         round(grp["pre_mean"].mean(), 4),
            "mean_beta_level":       round(grp["beta_level"].mean(), 4),
            "mean_beta_slope":       round(grp["beta_slope"].mean(), 6),
            "pct_level_significant": round(grp["level_significant"].mean() * 100, 1),
            "pct_slope_significant": round(grp["slope_significant"].mean() * 100, 1),
            "mean_cohens_d":         round(grp["cohens_d"].mean(), 4),
            "mean_days_to_return":   round(grp["days_to_return"].mean(), 1),
            "median_days_to_return": round(grp["days_to_return"].median(), 1),
        })

    return pd.DataFrame(summary_rows).sort_values(["metric", "severity"])


def print_highlights(results: pd.DataFrame):
    """Print the most notable findings to console."""
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    sig = results[results["level_significant"]]
    print(f"\nSignificant immediate level changes (p < {ALPHA}): {len(sig)} / {len(results)}")

    if not sig.empty:
        print("\nTop level changes by magnitude (Cohen's d):")
        top = sig.nlargest(10, "cohens_d")[
            ["label", "metric", "beta_level", "p_level", "cohens_d", "days_to_return"]
        ]
        print(top.to_string(index=False))

    print("\nBy severity group (mean immediate level change):")
    sev_summary = results.groupby(["severity", "metric"])["beta_level"].mean().round(4)
    print(sev_summary.to_string())

    print("\nReturn to baseline (mean days post-event):")
    ret = results[results["days_to_return"].notna()].groupby(
        ["severity", "metric"]
    )["days_to_return"].mean().round(1)
    print(ret.to_string())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["emotions", "keywords", "both"],
                        default="both")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum texts per day to include in regression (default 10)")
    parser.add_argument("--dominant", action="store_true",
                        help="Use proportion of dominant emotions instead of mean score")
    parser.add_argument("--combine-minneapolis", action="store_true",
                        help="Treat Minneapolis MN_1/2/3 as one combined incident with MN_1 as breakpoint")
    args = parser.parse_args()

    agg_label = "dominant proportion" if args.dominant else "mean score"
    print(f"Running ITS analysis (metric={args.metric}, min_n={args.min_n}, aggregation={agg_label}, combine_minneapolis={args.combine_minneapolis})")

    results = run_analysis(metric=args.metric, min_n=args.min_n, use_dominant=args.dominant,
                           combine_mpls=args.combine_minneapolis)

    if results.empty:
        print("No results — check that CSV files exist and have data.")
        return

    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nFull results saved to {RESULTS_CSV} ({len(results)} rows)")

    summary = summarize_by_severity(results)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary saved to {SUMMARY_CSV}")

    print_highlights(results)


if __name__ == "__main__":
    main()