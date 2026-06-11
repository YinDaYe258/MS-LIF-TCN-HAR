from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


V2_DIR = Path("results/final_paper_v2")
RAW_PATH = V2_DIR / "main_results_raw.csv"
SUMMARY_PATH = V2_DIR / "main_results_summary.csv"
TABLE_MAIN_PATH = V2_DIR / "table_main_results_summary.tex"
LOCKED_README_PATH = V2_DIR / "LOCKED_README.md"
MANIFEST_PATH = V2_DIR / "MANIFEST.yaml"
PAIRWISE_PATH = V2_DIR / "pairwise_statistics.csv"
PAIRWISE_TABLE_PATH = V2_DIR / "table_pairwise_statistics.tex"
CLAIM_PATH = V2_DIR / "claim_support_matrix.csv"
CLAIM_TABLE_PATH = V2_DIR / "table_claim_support_matrix.tex"
SEED_LEVEL_PATH = V2_DIR / "seed_level_main_results.csv"
RATIONALE_PATH = V2_DIR / "model_selection_rationale.md"
READINESS_PATH = V2_DIR / "final_readiness_report.md"

MODEL_ORDER = [
    "cnn1d",
    "gru",
    "window_gru",
    "ms_lif_snn",
    "ms_lif_snn_wide",
    "ms_ann_tcn",
    "ms_lif_tcn",
    "ms_lif_tcn_plus",
]
NON_SPIKING_MODELS = {"cnn1d", "gru", "window_gru", "ms_ann_tcn"}
PAIRWISE_COMPARISONS = [
    ("ms_lif_tcn", "ms_lif_snn"),
    ("ms_lif_tcn", "ms_lif_snn_wide"),
    ("ms_lif_tcn", "ms_ann_tcn"),
    ("ms_lif_tcn", "window_gru"),
    ("ms_lif_tcn_plus", "ms_lif_tcn"),
]


def main() -> None:
    V2_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    summary = summarize(raw)
    pairwise = pairwise_statistics(raw)
    claims = build_claim_matrix(summary, pairwise)
    seed_level = build_seed_level_table(raw)

    summary.to_csv(SUMMARY_PATH, index=False)
    write_main_latex(summary)
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    write_pairwise_latex(pairwise)
    claims.to_csv(CLAIM_PATH, index=False)
    write_claim_latex(claims)
    seed_level.to_csv(SEED_LEVEL_PATH, index=False)
    write_locked_readme(summary)
    write_manifest()
    write_model_selection_rationale(summary, pairwise)
    write_readiness_report(summary, claims)

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {TABLE_MAIN_PATH}")
    print(f"Wrote {PAIRWISE_PATH}")
    print(f"Wrote {CLAIM_PATH}")
    print(f"Wrote {SEED_LEVEL_PATH}")


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, model), group in raw.groupby(["dataset_key", "model"], sort=False):
        group = group.sort_values("seed")
        spike_values = group["spike_rate"].dropna()
        rows.append(
            {
                "dataset": group["dataset"].iloc[0],
                "dataset_key": dataset_key,
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "seeds": " ".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": std(group["accuracy"]),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": std(group["macro_f1"]),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": std(group["weighted_f1"]),
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std": std(group["balanced_accuracy"]),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": spike_values.mean() if not spike_values.empty else np.nan,
                "spike_rate_std": std(spike_values) if len(spike_values) else np.nan,
                "best_epoch_mean": group["best_epoch"].mean(),
            }
        )
    frame = pd.DataFrame(rows)
    frame["dataset_order"] = frame["dataset_key"].map({"ucihar": 0, "hapt6": 1})
    frame["model_order"] = frame["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    return frame.sort_values(["dataset_order", "model_order", "model"]).drop(columns=["dataset_order", "model_order"])


def pairwise_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key in ["ucihar", "hapt6"]:
        dataset = raw[raw["dataset_key"].eq(dataset_key)]
        dataset_name = dataset["dataset"].iloc[0]
        for model_a, model_b in PAIRWISE_COMPARISONS:
            a = dataset[dataset["model"].eq(model_a)][["seed", "macro_f1"]].rename(columns={"macro_f1": "a"})
            b = dataset[dataset["model"].eq(model_b)][["seed", "macro_f1"]].rename(columns={"macro_f1": "b"})
            merged = a.merge(b, on="seed", how="inner").sort_values("seed")
            if merged.empty:
                continue
            diffs = (merged["a"] - merged["b"]).to_numpy(dtype=float)
            mean_delta = float(diffs.mean())
            std_delta = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
            ci_low, ci_high = confidence_interval(diffs)
            t_p = paired_t_pvalue(merged["a"].to_numpy(dtype=float), merged["b"].to_numpy(dtype=float))
            w_p = wilcoxon_pvalue(diffs)
            wins = int((diffs > 1e-12).sum())
            ties = int((np.abs(diffs) <= 1e-12).sum())
            losses = int((diffs < -1e-12).sum())
            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_key": dataset_key,
                    "comparison": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "num_pairs": int(len(diffs)),
                    "seeds": " ".join(str(int(seed)) for seed in merged["seed"]),
                    "mean_delta_macro_f1": mean_delta,
                    "std_delta_macro_f1": std_delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "win_count": wins,
                    "tie_count": ties,
                    "loss_count": losses,
                    "per_seed_delta": "; ".join(
                        f"{int(seed)}:{delta:.4f}" for seed, delta in zip(merged["seed"], diffs)
                    ),
                    "interpretation": interpret_pairwise(mean_delta, ci_low, ci_high, t_p, w_p, wins, losses),
                }
            )
    return pd.DataFrame(rows)


def build_seed_level_table(raw: pd.DataFrame) -> pd.DataFrame:
    focus_models = ["ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn", "ms_lif_tcn_plus", "window_gru"]
    subset = raw[raw["model"].isin(focus_models)].copy()
    pivot = subset.pivot_table(index=["dataset", "dataset_key", "seed"], columns="model", values="macro_f1", aggfunc="last")
    pivot = pivot.reset_index()
    ordered_columns = ["dataset", "dataset_key", "seed"] + [model for model in focus_models if model in pivot.columns]
    return pivot[ordered_columns].sort_values(["dataset_key", "seed"])


def build_claim_matrix(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    def pair(dataset_key: str, comparison: str) -> pd.Series:
        match = pairwise[pairwise["dataset_key"].eq(dataset_key) & pairwise["comparison"].eq(comparison)]
        if match.empty:
            raise KeyError(f"Missing pairwise result: {dataset_key} {comparison}")
        return match.iloc[0]

    uci_tcn_vs_snn = pair("ucihar", "ms_lif_tcn - ms_lif_snn")
    hapt_tcn_vs_snn = pair("hapt6", "ms_lif_tcn - ms_lif_snn")
    uci_tcn_vs_wide = pair("ucihar", "ms_lif_tcn - ms_lif_snn_wide")
    hapt_tcn_vs_wide = pair("hapt6", "ms_lif_tcn - ms_lif_snn_wide")
    uci_tcn_vs_ann = pair("ucihar", "ms_lif_tcn - ms_ann_tcn")
    hapt_tcn_vs_ann = pair("hapt6", "ms_lif_tcn - ms_ann_tcn")
    uci_tcn_vs_wgru = pair("ucihar", "ms_lif_tcn - window_gru")
    hapt_tcn_vs_wgru = pair("hapt6", "ms_lif_tcn - window_gru")
    uci_plus_vs_tcn = pair("ucihar", "ms_lif_tcn_plus - ms_lif_tcn")
    hapt_plus_vs_tcn = pair("hapt6", "ms_lif_tcn_plus - ms_lif_tcn")

    return pd.DataFrame(
        [
            claim(
                "MS-LIF-TCN improves over compact MS-LIF-SNN.",
                support_from_pair(uci_tcn_vs_snn, hapt_tcn_vs_snn),
                evidence_two("UCI-HAR", uci_tcn_vs_snn, "HAPT-6", hapt_tcn_vs_snn),
            ),
            claim(
                "MS-LIF-TCN improves over parameter-matched MS-LIF-SNN-wide.",
                support_from_pair(uci_tcn_vs_wide, hapt_tcn_vs_wide),
                evidence_two("UCI-HAR", uci_tcn_vs_wide, "HAPT-6", hapt_tcn_vs_wide),
            ),
            claim(
                "MS-LIF-TCN has a higher five-seed mean than MS-ANN-TCN on UCI-HAR.",
                support_from_single_pair(uci_tcn_vs_ann),
                evidence_one(uci_tcn_vs_ann),
            ),
            claim(
                "MS-LIF-TCN has a higher five-seed mean than MS-ANN-TCN on HAPT-6.",
                support_from_single_pair(hapt_tcn_vs_ann),
                evidence_one(hapt_tcn_vs_ann),
            ),
            claim(
                "MS-LIF-TCN has a higher five-seed mean than Window-GRU on UCI-HAR.",
                support_from_single_pair(uci_tcn_vs_wgru),
                evidence_one(uci_tcn_vs_wgru),
            ),
            claim(
                "MS-LIF-TCN has a higher five-seed mean than Window-GRU on HAPT-6.",
                support_from_single_pair(hapt_tcn_vs_wgru),
                evidence_one(hapt_tcn_vs_wgru),
            ),
            claim(
                "MS-LIF-TCN+ is globally stable best model.",
                "not_supported",
                f"UCI-HAR plus-vs-main {evidence_one(uci_plus_vs_tcn)}; HAPT-6 plus-vs-main {evidence_one(hapt_plus_vs_tcn)}. Treat plus as enhanced diagnostic.",
            ),
            claim(
                "MS-LIF-TCN+ is strongest enhanced diagnostic on HAPT-6.",
                support_from_single_pair(hapt_plus_vs_tcn),
                evidence_one(hapt_plus_vs_tcn),
            ),
            claim(
                "SNN has measured neuromorphic low power.",
                "not_supported",
                "Only RTX 4060 PyTorch/NVML software-stack GPU measurements and spike-rate proxies are available.",
            ),
            claim(
                "RTX 4060 benchmark supports software-stack efficiency analysis.",
                "supported",
                "Verified GPU benchmark is explicitly scoped to PyTorch/CUDA/NVML, not neuromorphic hardware.",
            ),
        ]
    )


def claim(text: str, support: str, evidence: str) -> dict[str, str]:
    return {"claim": text, "support": support, "evidence": evidence}


def support_from_pair(a: pd.Series, b: pd.Series) -> str:
    statuses = [support_from_single_pair(a), support_from_single_pair(b)]
    if all(status == "statistically_supported" for status in statuses):
        return "statistically_supported"
    if all(status in {"statistically_supported", "higher_mean_only"} for status in statuses):
        return "higher_mean_supported"
    if any(status == "not_supported" for status in statuses):
        return "mixed_or_not_supported"
    return "weakly_supported"


def support_from_single_pair(row: pd.Series) -> str:
    mean_delta = float(row["mean_delta_macro_f1"])
    ci_low = float(row["ci95_low"])
    t_p = float(row["paired_t_p"])
    w_p = float(row["wilcoxon_p"])
    wins = int(row["win_count"])
    losses = int(row["loss_count"])
    if mean_delta <= 0:
        return "not_supported"
    if ci_low > 0 and t_p < 0.05 and w_p < 0.05:
        return "statistically_supported"
    if wins > losses:
        return "higher_mean_only"
    return "weakly_supported"


def evidence_two(label_a: str, row_a: pd.Series, label_b: str, row_b: pd.Series) -> str:
    return f"{label_a}: {evidence_one(row_a)}; {label_b}: {evidence_one(row_b)}"


def evidence_one(row: pd.Series) -> str:
    return (
        f"mean delta={float(row['mean_delta_macro_f1']):.4f}, "
        f"95% CI [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}], "
        f"t p={float(row['paired_t_p']):.4g}, Wilcoxon p={float(row['wilcoxon_p']):.4g}, "
        f"W/T/L={int(row['win_count'])}/{int(row['tie_count'])}/{int(row['loss_count'])}"
    )


def interpret_pairwise(
    mean_delta: float,
    ci_low: float,
    ci_high: float,
    t_p: float,
    w_p: float,
    wins: int,
    losses: int,
) -> str:
    if mean_delta <= 0:
        return "not_supported"
    if ci_low > 0 and t_p < 0.05 and w_p < 0.05:
        return "statistically_supported"
    if wins > losses:
        return "higher_mean_only"
    if ci_low <= 0 <= ci_high:
        return "competitive_or_inconclusive"
    return "weak_support"


def write_main_latex(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Dataset & Model & Seeds & Acc. & Macro-F1 & Weighted-F1 & Params & Spike Rate \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        spike = "N/A" if pd.isna(row["spike_rate_mean"]) else mean_std(row["spike_rate_mean"], row["spike_rate_std"], 4)
        lines.append(
            f"{escape_latex(row['dataset'])} & {escape_latex(row['model'])} & {int(row['num_seeds'])} & "
            f"{mean_std(row['accuracy_mean'], row['accuracy_std'])} & "
            f"{mean_std(row['macro_f1_mean'], row['macro_f1_std'])} & "
            f"{mean_std(row['weighted_f1_mean'], row['weighted_f1_std'])} & "
            f"{int(row['params']):,} & {spike} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    TABLE_MAIN_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_pairwise_latex(pairwise: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Comparison & $\\Delta$ Macro-F1 & 95\\% CI & t-test p & Wilcoxon p & W/T/L \\\\",
        "\\midrule",
    ]
    for _, row in pairwise.iterrows():
        lines.append(
            f"{escape_latex(row['dataset'])} & {escape_latex(row['comparison'])} & "
            f"{float(row['mean_delta_macro_f1']):.4f} & "
            f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] & "
            f"{format_p(row['paired_t_p'])} & "
            f"{format_p(row['wilcoxon_p'])} & "
            f"{int(row['win_count'])}/{int(row['tie_count'])}/{int(row['loss_count'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    PAIRWISE_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_claim_latex(claims: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Claim & Support & Evidence \\\\",
        "\\midrule",
    ]
    for _, row in claims.iterrows():
        lines.append(
            f"{escape_latex(row['claim'])} & {escape_latex(row['support'])} & "
            f"{escape_latex(row['evidence'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    CLAIM_TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_locked_readme(summary: pd.DataFrame) -> None:
    lines = [
        "# Final Paper V2 Locked Results",
        "",
        "This directory is the v2 result package. It extends, but does not overwrite, `results/final_paper/`.",
        "",
        "## Protocol",
        "",
        "- Datasets: UCI-HAR and HAPT-6.",
        "- Context length: `K=8`.",
        "- Target mode: `last`.",
        "- Seeds: `42, 43, 44, 45, 46`.",
        "- Main model: `ms_lif_tcn`.",
        "- Enhanced diagnostic: `ms_lif_tcn_plus`.",
        "- Structural non-spiking control: `ms_ann_tcn` / `ms_cnn_tcn`.",
        "- Parameter-matched spiking control: `ms_lif_snn_wide`.",
        "",
        "## Main Files",
        "",
        "- `main_results_raw.csv`: seed-level raw rows.",
        "- `main_results_summary.csv`: 5-seed mean/std table.",
        "- `table_main_results_summary.tex`: LaTeX main table.",
        "- `pairwise_statistics.csv`: paired seed-level comparisons.",
        "- `claim_support_matrix.csv`: supported and unsupported claims.",
        "- `seed_level_main_results.csv`: seed-level Macro-F1 pivot for traceability.",
        "",
        "## Use In Paper",
        "",
        "Use v2 for the main paper table if the manuscript consistently reports five-seed results.",
        "Keep v1 as the earlier locked package and do not mix v1 three-seed means with v2 five-seed means.",
        "",
        "## Caveats",
        "",
        "- `MS-LIF-TCN+` is an enhanced diagnostic variant, not the primary model.",
        "- `MS-ANN-TCN` is a strong continuous non-spiking baseline and should be reported.",
        "- GPU measurements remain RTX 4060 PyTorch/NVML software-stack measurements, not neuromorphic hardware energy.",
        "",
    ]
    LOCKED_README_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    commit = git_head()
    lines = [
        f"repo_commit_at_generation: {commit}",
        "source_result_commit: d28953f",
        "result_package: results/final_paper_v2",
        "datasets:",
        "  ucihar:",
        "    root: data/UCI HAR Dataset",
        "    split: official subject train/test split",
        "    context_len: 8",
        "    target_mode: last",
        "    channels: 9",
        "  hapt6:",
        "    root: data/HAPT Dataset",
        "    task: activity_id 1-6",
        "    split: official subject train/test split",
        "    sequence_within_segment: true",
        "    context_len: 8",
        "    target_mode: last",
        "    channels: 6",
        "seeds: [42, 43, 44, 45, 46]",
        "primary_model: ms_lif_tcn",
        "enhanced_diagnostic: ms_lif_tcn_plus",
        "main_models:",
        "  - cnn1d",
        "  - gru",
        "  - window_gru",
        "  - ms_lif_snn",
        "  - ms_lif_snn_wide",
        "  - ms_ann_tcn",
        "  - ms_lif_tcn",
        "  - ms_lif_tcn_plus",
        "main_tables:",
        "  - results/final_paper_v2/main_results_summary.csv",
        "  - results/final_paper_v2/table_main_results_summary.tex",
        "  - results/final_paper_v2/pairwise_statistics.csv",
        "excluded_from_main_claims:",
        "  - CMG variants",
        "  - HAPT-12 K2",
        "  - GPU neuromorphic energy claims",
        "  - state-of-the-art claims",
        "notes:",
        "  - Non-spiking models have spike_rate reported as N/A in paper tables.",
        "  - MS-ANN-TCN uses the same multi-scale encoder and causal window-level TCN as MS-LIF-TCN, without LIF spike dynamics.",
        "",
    ]
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_model_selection_rationale(summary: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    lines = [
        "# Model Selection Rationale",
        "",
        "## Why MS-LIF-TCN Is The Primary Model",
        "",
        "1. It improves over compact `ms_lif_snn` on both UCI-HAR and HAPT-6.",
        "2. It improves over parameter-matched `ms_lif_snn_wide`, reducing the risk that the gain is only parameter count.",
        "3. It is competitive with the non-spiking `ms_ann_tcn` structural counterpart while retaining spike-based representation.",
        "4. It is simpler and more stable as a central contribution than `ms_lif_tcn_plus`.",
        "",
        "## Why MS-LIF-TCN+ Is Diagnostic",
        "",
        "1. It has the strongest current HAPT-6 mean Macro-F1.",
        "2. It has the highest current UCI-HAR mean Macro-F1, but with higher variance.",
        "3. It adds attention and supervised contrastive loss, which broadens the contribution beyond the core causal window TCN.",
        "",
        "## Why MS-ANN-TCN Must Be Reported",
        "",
        "`ms_ann_tcn` uses the same multi-scale encoder and causal window-level TCN as `ms_lif_tcn`, but replaces LIF spike dynamics with continuous average-pooled features. It is a strong structural baseline and prevents over-attributing gains to spiking dynamics alone.",
        "",
    ]
    RATIONALE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_readiness_report(summary: pd.DataFrame, claims: pd.DataFrame) -> None:
    lines = [
        "# Final Readiness Report After V2",
        "",
        "## New Evidence In V2",
        "",
        "- Five-seed evaluation on UCI-HAR and HAPT-6.",
        "- Parameter-matched `ms_lif_snn_wide` control.",
        "- Non-spiking `ms_ann_tcn` structural control.",
        "- Paired seed-level statistics for core comparisons.",
        "",
        "## Recommended Main Claim",
        "",
        "MS-LIF-TCN provides competitive spiking temporal modeling for wearable HAR by combining a multi-scale spiking window encoder with causal window-level TCN context modeling.",
        "",
        "## Safe Claims",
        "",
    ]
    safe = claims[claims["support"].isin(["supported", "statistically_supported", "higher_mean_supported"])]
    for _, row in safe.iterrows():
        lines.append(f"- {row['claim']} ({row['support']})")
    lines.extend(
        [
            "",
            "## Mean-Only Observations",
            "",
            "These comparisons have higher five-seed means but are not statistically supported at the current seed count.",
        ]
    )
    mean_only = claims[claims["support"].eq("higher_mean_only")]
    for _, row in mean_only.iterrows():
        lines.append(f"- {row['claim']} ({row['support']})")
    lines.extend(
        [
            "",
            "## Unsafe Claims",
            "",
        ]
    )
    unsafe = claims[claims["support"].isin(["not_supported", "mixed_or_not_supported"])]
    for _, row in unsafe.iterrows():
        lines.append(f"- {row['claim']} ({row['support']})")
    lines.extend(
        [
            "",
            "## Remaining Limitations",
            "",
            "- Only two main datasets are used.",
            "- GPU benchmark is RTX 4060 software-stack measurement, not neuromorphic hardware energy.",
            "- `MS-LIF-TCN+` has higher variance on UCI-HAR and should remain an enhanced diagnostic variant.",
            "- `MS-ANN-TCN` is a strong non-spiking baseline, so claims should emphasize competitive spiking temporal modeling rather than universal superiority over ANN models.",
            "",
            "## Recommendation",
            "",
            "Use v2 as the main paper result package, with v1 retained as a historical locked package. Do not add more architectures unless a reviewer explicitly requests them.",
            "",
        ]
    )
    READINESS_PATH.write_text("\n".join(lines), encoding="utf-8")


def std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def confidence_interval(diffs: np.ndarray) -> tuple[float, float]:
    if len(diffs) < 2:
        mean = float(diffs.mean())
        return mean, mean
    mean = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    t_crit = float(stats.t.ppf(0.975, df=len(diffs) - 1))
    return mean - t_crit * sem, mean + t_crit * sem


def paired_t_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    result = stats.ttest_rel(a, b)
    return float(result.pvalue)


def wilcoxon_pvalue(diffs: np.ndarray) -> float:
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        return float("nan")


def mean_std(mean: float, sigma: float, digits: int = 4) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(sigma):.{digits}f}"


def format_p(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.4f}"


def escape_latex(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    main()
