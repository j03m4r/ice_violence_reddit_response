import re
import matplotlib.pyplot as plt
import pandas as pd

CRITICAL_KEYWORD = "immigr"
IMMIGRATION_KEYWORDS = ["trump", "dhs", "ice", "enforcement", "citizen", "illegal", "federal", "murder", "border", "deport"]

def select_submissions(submissions_df, n):
    joined_keywords = "|".join(IMMIGRATION_KEYWORDS)

    submissions_series = submissions_df["text"]
    matched_submissions = submissions_df[submissions_series.str.contains(f"(?=.*{CRITICAL_KEYWORD})(?=.*({joined_keywords}))", regex=True, flags=re.IGNORECASE)].copy()
    matched_submissions["keywords"] = matched_submissions["text"].apply(
        lambda text: [kw for kw in [*IMMIGRATION_KEYWORDS, CRITICAL_KEYWORD] if re.search(kw, text, re.IGNORECASE)]
    )
    matched_submissions["keyword_count"] = matched_submissions["keywords"].str.len()
    matched_submissions["text_word_count"] = matched_submissions["text"].str.split().str.len()
    return matched_submissions.filter(items=["id", "text", "text_word_count", "keyword_count", "keywords", "subreddit", "incident_key", "incident_date", "days_relative", "score", "created_utc"])


def plot_word_count_vs_keyword_count(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["text_word_count"],
        df["keyword_count"],
        alpha=0.5,
        c=df["keyword_count"],
        cmap="viridis",
        edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="keyword count")
    ax.set_xlabel("word count")
    ax.set_ylabel("keyword count")
    ax.set_title("word count vs keyword count")
    plt.tight_layout()
    plt.savefig("word_vs_keyword.png", dpi=150)
    plt.show()

    
def plot_keyword_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["keyword_count"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="steelblue", edgecolor="none")
    ax.set_xlabel("keyword count")
    ax.set_ylabel("number of submissions")
    ax.set_title("distribution of submissions by keyword count")
    ax.set_xticks(counts.index)
    plt.tight_layout()
    plt.savefig("keyword_distribution.png", dpi=150)
    plt.show()


def main():
    df = pd.read_csv("./data/keyword_results.csv", index_col="id")
    submissions_df = df[df["type"] == "submission"]
    selected_submissions_df = select_submissions(submissions_df, 100)
    # print(selected_submissions_df)
    # plot_word_count_vs_keyword_count(selected_submissions_df)
    # plot_keyword_distribution(selected_submissions_df)
    six_or_more = selected_submissions_df[selected_submissions_df["keyword_count"] >= 4]
    print(len(six_or_more))

    fatal_incident_submissions = six_or_more[six_or_more["incident_key"].isin(["Chicago_IL_1", "Minneapolis_MN_1", "Minneapolis_MN_3", "Los_Angeles_CA_2"])]
    print(fatal_incident_submissions)
    avg_wrd_count = fatal_incident_submissions["text_word_count"].mean()
    print(avg_wrd_count)
    agg_df = fatal_incident_submissions.groupby(["incident_key"]).count()
    print(agg_df)

if __name__ == "__main__":
    main()