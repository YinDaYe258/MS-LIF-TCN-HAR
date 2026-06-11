from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

V3_DIR = Path("results/final_paper_v3")
RAW_PATH = V3_DIR / "main_results_raw.csv"
SUMMARY_PATH = V3_DIR / "main_results_summary.csv"
PAIRWISE_PATH = V3_DIR / "pairwise_statistics.csv"
ENERGY_PROXY_PATH = V3_DIR / "energy_proxy.csv"
CLAIM_PATH = V3_DIR / "claim_support_matrix.csv"
MANIFEST_PATH = V3_DIR / "MANIFEST.yaml"
LOCKED_README_PATH = V3_DIR / "LOCKED_README.md"
READINESS_PATH = V3_DIR / "final_readiness_report.md"

DATASET_ORDER = ["ucihar", "hapt6", "pamap2", "mhealth"]
MODEL_ORDER = ["cnn1d", "window_gru", "ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
NON_SPIKING = {"cnn1d", "window_gru", "ms_ann_tcn"}
PAIRWISE = [
    ("ms_lif_tcn", "ms_lif_snn"),
    ("ms_lif_tcn", "ms_lif_snn_wide"),
    ("ms_lif_tcn", "ms_ann_tcn"),
    ("ms_lif_tcn", "window_gru"),
]


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()
    if raw.empty:
        write_manifest()
        write_locked_readme(pd.DataFrame(), pd.DataFrame())
        write_readiness_report(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        print(f"No raw v3 results found at {RAW_PATH}; wrote protocol skeleton.")
        return
    summary = summarize(raw)
    pairwise = pairwise_statistics(raw)
    energy = energy_proxy(raw)
    claims = claim_matrix(pairwise)
    summary.to_csv(SUMMARY_PATH, index=False)
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    energy.to_csv(ENERGY_PROXY_PATH, index=False)
    claims.to_csv(CLAIM_PATH, index=False)
    summary.to_latex(V3_DIR / "table_main_results_summary.tex", index=False, escape=False, float_format=lambda value: f"{value:.4f}")
    pairwise.to_latex(V3_DIR / "table_pairwise_statistics.tex", index=False, escape=False, float_format=lambda value: f"{value:.4f}")
    energy.to_latex(V3_DIR / "table_energy_proxy.tex", index=False, escape=False, float_format=lambda value: f"{value:.4f}")
    claims.to_latex(V3_DIR / "table_claim_support_matrix.tex", index=False, escape=False)
    write_manifest()
    write_locked_readme(summary, claims)
    write_readiness_report(summary, pairwise, claims)
    print(f"Wrote v3 summary package under {V3_DIR}")


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, model), group in raw.groupby(["dataset_key", "model"], sort=False):
        group = group.sort_values("seed")
        spike = pd.to_numeric(group.get("spike_rate", pd.Series(dtype=float)), errors="coerce").dropna()
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
                "spike_rate_mean": spike.mean() if not spike.empty else np.nan,
                "spike_rate_std": std(spike) if not spike.empty else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["dataset_order"] = frame["dataset_key"].map({name: idx for idx, name in enumerate(DATASET_ORDER)})
    frame["model_order"] = frame["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    return frame.sort_values(["dataset_order", "model_order", "model"]).drop(columns=["dataset_order", "model_order"])


def pairwise_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, dataset in raw.groupby("dataset_key", sort=False):
        dataset_name = dataset["dataset"].iloc[0]
        for model_a, model_b in PAIRWISE:
            a = dataset[dataset["model"].eq(model_a)][["seed", "macro_f1"]].rename(columns={"macro_f1": "a"})
            b = dataset[dataset["model"].eq(model_b)][["seed", "macro_f1"]].rename(columns={"macro_f1": "b"})
            merged = a.merge(b, on="seed", how="inner").sort_values("seed")
            if merged.empty:
                continue
            diffs = (merged["a"] - merged["b"]).to_numpy(dtype=float)
            ci_low, ci_high = confidence_interval(diffs)
            t_p = paired_t_pvalue(merged["a"].to_numpy(dtype=float), merged["b"].to_numpy(dtype=float))
            w_p = wilcoxon_pvalue(diffs)
            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_key": dataset_key,
                    "comparison": f"{model_a} - {model_b}",
                    "num_pairs": int(len(diffs)),
                    "seeds": " ".join(str(int(seed)) for seed in merged["seed"]),
                    "mean_delta_macro_f1": float(diffs.mean()),
                    "std_delta_macro_f1": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "win_count": int((diffs > 1e-12).sum()),
                    "tie_count": int((np.abs(diffs) <= 1e-12).sum()),
                    "loss_count": int((diffs < -1e-12).sum()),
                    "interpretation": interpret(diffs, ci_low, ci_high, t_p, w_p),
                }
            )
    return pd.DataFrame(rows)


def energy_proxy(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        model = str(row["model"])
        spike_rate = np.nan if model in NON_SPIKING else float(row.get("spike_rate", np.nan))
        hidden_dim = int(row.get("hidden_dim", 128))
        context_len = int(row.get("context_len", 1))
        window_size = int(row.get("window_size", 128))
        spike_count = spike_rate * context_len * window_size * hidden_dim if not np.isnan(spike_rate) else np.nan
        synops_proxy = spike_count * hidden_dim if not np.isnan(spike_count) else np.nan
        rows.append(
            {
                "dataset": row["dataset"],
                "dataset_key": row["dataset_key"],
                "model": model,
                "seed": int(row["seed"]),
                "macro_f1": float(row["macro_f1"]),
                "params": int(row["params"]),
                "model_size_mb": int(row["params"]) * 4.0 / (1024.0 * 1024.0),
                "spike_rate": spike_rate,
                "spike_count_per_sample_proxy": spike_count,
                "synops_proxy": synops_proxy,
                "note": "proxy_only_not_measured_power" if model not in NON_SPIKING else "non_spiking_model",
            }
        )
    return pd.DataFrame(rows)


def claim_matrix(pairwise: pd.DataFrame) -> pd.DataFrame:
    claims = []
    for comparison in ["ms_lif_tcn - ms_lif_snn", "ms_lif_tcn - ms_lif_snn_wide"]:
        subset = (
            pairwise[pairwise["comparison"].eq(comparison)]
            if "comparison" in pairwise.columns
            else pd.DataFrame()
        )
        if subset.empty:
            support = "not_evaluated"
            evidence = "No paired rows available."
        else:
            positive = int((subset["mean_delta_macro_f1"] > 0).sum())
            supported = int((subset["ci95_low"] > 0).sum())
            support = "supported" if supported >= max(1, len(subset) // 2) else "higher_mean_only" if positive == len(subset) else "mixed"
            evidence = "; ".join(
                f"{row.dataset}: Δ={row.mean_delta_macro_f1:.4f}, wins={int(row.win_count)}/{int(row.num_pairs)}"
                for row in subset.itertuples()
            )
        claims.append({"claim": comparison.replace(" - ", " improves over "), "support": support, "evidence": evidence})
    claims.append(
        {
            "claim": "SNN models have measured neuromorphic low power.",
            "support": "not_supported",
            "evidence": "v3 reports spike/SynOps proxies and optional conventional hardware measurements only.",
        }
    )
    return pd.DataFrame(claims)


def write_manifest() -> None:
    MANIFEST_PATH.write_text(
        "\n".join(
            [
                "result_package: results/final_paper_v3",
                "primary_model: ms_lif_tcn",
                "seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]",
                "datasets:",
                "  ucihar:",
                "    root: data/UCI HAR Dataset",
                "    context_len: 8",
                "    target_mode: last",
                "    channels: 9",
                "    window_size: 128",
                "  hapt6:",
                "    root: data/HAPT Dataset",
                "    task: activities 1-6 from raw HAPT segments",
                "    context_len: 8",
                "    target_mode: last",
                "    channels: 6",
                "    window_size: 128",
                "  pamap2:",
                "    root: data/PAMAP2_Dataset",
                "    task: Protocol 12 activities",
                "    test_subjects: [105, 106]",
                "    preprocessing: train-only mean imputation and train z-score",
                "    context_len: 8",
                "    target_mode: last",
                "    channels: 18",
                "    window_size: 256",
                "  mhealth:",
                "    root: data/MHEALTHDATASET",
                "    test_subjects: [9, 10]",
                "    preprocessing: train-only mean imputation and train z-score",
                "    context_len: 8",
                "    target_mode: last",
                "    channels: 15",
                "    window_size: 128",
                "main_models:",
                "  - cnn1d",
                "  - window_gru",
                "  - ms_lif_snn",
                "  - ms_lif_snn_wide",
                "  - ms_ann_tcn",
                "  - ms_lif_tcn",
                "excluded_claims:",
                "  - state_of_the_art",
                "  - measured_neuromorphic_energy",
                "  - universal_snn_superiority_over_ann",
                "excluded_files:",
                "  - smoke_results.csv",
                "  - artifacts generated by smoke tests",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_locked_readme(summary: pd.DataFrame, claims: pd.DataFrame) -> None:
    lines = [
        "# Final Paper v3 Package",
        "",
        "This package is reserved for the controlled v3 expansion: UCI-HAR, HAPT-6, PAMAP2, and MHEALTH with ten seeds.",
        "",
        "No new architecture search should be mixed into this package. The primary model is `ms_lif_tcn`.",
        "",
        "PAMAP2 uses the standard `Protocol/` files as a 12-activity task. Optional PAMAP2 activity classes are not part of the main v3 protocol.",
        "",
        "MHEALTH uses subjects 1-8 for training/validation and subjects 9-10 for testing by default.",
        "",
        "PAMAP2 and MHEALTH rows must be generated only from real downloaded datasets. Synthetic or smoke-test rows are not paper results.",
        "",
        "External raw datasets use train-only missing-value imputation and train-only z-score standardization. The canonical stats live under `preprocessing_stats/`.",
    ]
    if not summary.empty:
        lines.extend(["", "## Available Summary", "", "```text", summary.to_string(index=False), "```"])
    if not claims.empty:
        lines.extend(["", "## Claim Matrix", "", "```text", claims.to_string(index=False), "```"])
    LOCKED_README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness_report(summary: pd.DataFrame, pairwise: pd.DataFrame, claims: pd.DataFrame) -> None:
    if summary.empty:
        text = "\n".join(
            [
                "# v3 Readiness Report",
                "",
                "Status: protocol skeleton only.",
                "",
                "PAMAP2 and MHEALTH loaders, configs, download scripts, and dataset inspection outputs are available.",
                "The paper-ready v3 main result table is not ready because `main_results_raw.csv` has not been generated.",
                "",
                "Required next steps:",
                "",
                "1. Run `python scripts/run_final_paper_v3.py --datasets ucihar hapt6 pamap2 mhealth --models main --seeds 42 43 44 45 46 47 48 49 50 51`.",
                "2. Run `python -m src.analysis.final_paper_v3` after all real-data rows are complete.",
                "3. Treat `smoke_results.csv` and `artifacts/*_smoke_*` as code-path validation only, not paper results.",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "# v3 Readiness Report",
                "",
                f"Raw models summarized: {len(summary)} dataset/model rows.",
                "",
                "Use this package only after all four datasets and ten seeds are complete.",
                "",
                "Energy numbers are proxies unless explicitly generated by hardware benchmark scripts.",
            ]
        )
    READINESS_PATH.write_text(text, encoding="utf-8")


def std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def confidence_interval(diffs: np.ndarray) -> tuple[float, float]:
    if len(diffs) < 2:
        return float(diffs.mean()), float(diffs.mean())
    sem = stats.sem(diffs)
    margin = stats.t.ppf(0.975, len(diffs) - 1) * sem
    return float(diffs.mean() - margin), float(diffs.mean() + margin)


def paired_t_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def wilcoxon_pvalue(diffs: np.ndarray) -> float:
    if len(diffs) < 2 or np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        return 1.0


def interpret(diffs: np.ndarray, ci_low: float, ci_high: float, t_p: float, w_p: float) -> str:
    if ci_low > 0 and t_p < 0.05 and w_p < 0.05:
        return "statistically_supported"
    if diffs.mean() > 0 and (diffs > 0).sum() >= max(1, len(diffs) - 1):
        return "higher_mean_consistent_wins"
    if diffs.mean() > 0:
        return "higher_mean_only"
    return "not_supported"


if __name__ == "__main__":
    main()
