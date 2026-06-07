"""Generate exploratory data analysis artifacts for the UCI Parkinson dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import PROJECT_ROOT, get_feature_columns, load_dataset_config, load_raw_dataset


REPORTS_DIR = PROJECT_ROOT / "reports"


def _write_markdown_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_counts: pd.Series,
    missing_counts: pd.Series,
    high_correlations: pd.DataFrame,
) -> Path:
    config = load_dataset_config()
    report_path = REPORTS_DIR / "phase1_eda_summary.md"
    positive = int(class_counts.get(config["positive_label"], 0))
    negative = int(class_counts.get(config["negative_label"], 0))
    total = int(class_counts.sum())

    lines = [
        "# Phase 1 EDA Summary",
        "",
        "## Dataset",
        "",
        f"- Name: {config['name']}",
        f"- Source: {config['source_url']}",
        f"- Raw file: `{config['raw_path']}`",
        f"- Shape: {frame.shape[0]} rows x {frame.shape[1]} columns",
        f"- Identifier column: `{config['id_column']}`",
        f"- Target column: `{config['target_column']}`",
        f"- Numeric feature columns: {len(feature_columns)}",
        "",
        "## Target Distribution",
        "",
        "| Label | Meaning | Count | Percentage |",
        "| --- | --- | ---: | ---: |",
        f"| 0 | Healthy | {negative} | {negative / total:.1%} |",
        f"| 1 | Parkinson | {positive} | {positive / total:.1%} |",
        "",
        "## Data Quality",
        "",
        f"- Missing values: {int(missing_counts.sum())}",
        f"- Duplicate rows: {int(frame.duplicated().sum())}",
        f"- Unique recording IDs: {frame[config['id_column']].nunique()}",
        "",
        "## Feature Overview",
        "",
        "- The UCI file already contains acoustic measurements, so Phase 2 can begin with cleaning, scaling, and feature selection rather than raw-audio extraction.",
        "- Subject-level splitting is required later. In this dataset, each `name` value represents one recording ID, not a repeated subject identifier.",
        "- Because explicit subject IDs are unavailable, the safest fallback for Phase 3 is to document this limitation and avoid any split that duplicates the same `name` across train/test.",
        "",
        "## Strong Feature Correlations",
        "",
        "| Feature A | Feature B | Absolute Correlation |",
        "| --- | --- | ---: |",
    ]

    if high_correlations.empty:
        lines.append("| - | - | - |")
    else:
        for _, row in high_correlations.head(15).iterrows():
            lines.append(f"| `{row['feature_a']}` | `{row['feature_b']}` | {row['abs_corr']:.3f} |")

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `reports/class_distribution.png`",
            "- `reports/correlation_heatmap.png`",
            "- `reports/feature_distributions_by_label.png`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _plot_class_distribution(frame: pd.DataFrame) -> None:
    config = load_dataset_config()
    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x=config["target_column"], hue=config["target_column"], palette="Set2", legend=False)
    plt.xticks([0, 1], ["Healthy (0)", "Parkinson (1)"])
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Recordings")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=160)
    plt.close()


def _plot_correlation_heatmap(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    corr = frame[feature_columns].corr(numeric_only=True)
    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    pairs: list[dict[str, object]] = []
    for index, feature_a in enumerate(corr.columns):
        for feature_b in corr.columns[index + 1 :]:
            value = corr.loc[feature_a, feature_b]
            pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "abs_corr": abs(float(value)),
                }
            )
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)


def _plot_feature_distributions(frame: pd.DataFrame) -> None:
    config = load_dataset_config()
    selected = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]
    available = [column for column in selected if column in frame.columns]
    melted = frame.melt(
        id_vars=[config["target_column"]],
        value_vars=available,
        var_name="feature",
        value_name="value",
    )

    grid = sns.displot(
        data=melted,
        x="value",
        hue=config["target_column"],
        col="feature",
        col_wrap=3,
        kind="kde",
        fill=True,
        facet_kws={"sharex": False, "sharey": False},
        height=3.2,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Value", "Density")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_distributions_by_label.png", dpi=160)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_raw_dataset()
    feature_columns = get_feature_columns(frame)
    config = load_dataset_config()

    class_counts = frame[config["target_column"]].value_counts().sort_index()
    missing_counts = frame.isna().sum()

    _plot_class_distribution(frame)
    high_correlations = _plot_correlation_heatmap(frame, feature_columns)
    _plot_feature_distributions(frame)
    report_path = _write_markdown_report(
        frame=frame,
        feature_columns=feature_columns,
        class_counts=class_counts,
        missing_counts=missing_counts,
        high_correlations=high_correlations,
    )

    print(f"EDA report written to: {report_path}")
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Missing values: {int(missing_counts.sum())}")


if __name__ == "__main__":
    main()
