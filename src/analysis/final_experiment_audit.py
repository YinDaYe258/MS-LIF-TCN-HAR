from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def row_value(rows: pd.DataFrame, model: str, column: str, default: str = "N/A") -> str:
    if rows.empty or "model" not in rows.columns or column not in rows.columns:
        return default
    match = rows[rows["model"].astype(str) == model]
    if match.empty:
        return default
    value = match.iloc[0][column]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def claim_checklist(
    uci: pd.DataFrame,
    hapt6: pd.DataFrame,
    param_pairs: pd.DataFrame,
    hapt12: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        {
            "claim": "CMG-LIF-Lite improves average Macro-F1 over default LIF-SNN on UCI-HAR.",
            "status": "supported_with_caveat",
            "evidence": "UCI-HAR paired mean diff cmg_lif_lite - lif_snn is positive; win count is 2/3.",
        },
        {
            "claim": "CMG-LIF-Lite improvement is independent of model capacity.",
            "status": "not_supported",
            "evidence": "Parameter-matched lif_snn_h192 beats cmg_lif_lite on mean Macro-F1 for UCI-HAR and HAPT-6.",
        },
        {
            "claim": "MS-CMG-LIF beats all ANN baselines.",
            "status": "not_supported",
            "evidence": "Window-GRU is stronger on UCI-HAR and HAPT-6; CNN1D is also stronger than MS-CMG-LIF on UCI-HAR.",
        },
        {
            "claim": "HAPT-6 is usable as a second main dataset.",
            "status": "supported",
            "evidence": "Raw signals, official subject split, no train/test subject overlap, three seeds.",
        },
        {
            "claim": "HAPT-12 K=2 is a full 12-class main benchmark.",
            "status": "not_supported",
            "evidence": "K=2 sequence filtering leaves 11 effective test classes and very small transition supports.",
        },
        {
            "claim": "Efficiency results measure real power consumption.",
            "status": "not_supported",
            "evidence": "Only spike rate and operation proxies are computed; no hardware measurement is present.",
        },
    ]
    return pd.DataFrame(records)


def concise_summary_table(uci: pd.DataFrame, hapt6: pd.DataFrame, param: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, source in (("UCI-HAR", uci), ("HAPT-6", hapt6)):
        for model in ["window_gru", "cnn1d", "ms_cmg_lif", "ms_lif_snn", "cmg_lif_lite", "lif_snn"]:
            if source.empty or model not in set(source["model"].astype(str)):
                continue
            match = source[source["model"].astype(str) == model].iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "macro_f1_mean": match.get("macro_f1_mean", ""),
                    "macro_f1_std": match.get("macro_f1_std", ""),
                    "params": match.get("params", ""),
                    "spike_rate_mean": match.get("spike_rate_mean", ""),
                }
            )
    if not param.empty:
        for _, match in param[param["model"].astype(str) == "lif_snn_h192"].iterrows():
            rows.append(
                {
                    "dataset": "UCI-HAR" if str(match["task"]) == "ucihar" else "HAPT-6",
                    "model": "lif_snn_h192",
                    "macro_f1_mean": match.get("macro_f1_mean", ""),
                    "macro_f1_std": match.get("macro_f1_std", ""),
                    "params": match.get("params", ""),
                    "spike_rate_mean": match.get("spike_rate_mean", ""),
                }
            )
    return pd.DataFrame(rows)


def write_report() -> None:
    results_dir = Path("results")
    uci = read_csv(results_dir / "ucihar_formal_multiseed_summary.csv")
    hapt6 = read_csv(results_dir / "hapt6_multiseed_summary.csv")
    hapt12 = read_csv(results_dir / "hapt12_k2_multiseed_summary.csv")
    param = read_csv(results_dir / "param_matched_summary.csv")
    param_pairs = read_csv(results_dir / "param_matched_pairwise_summary.csv")
    robustness = read_csv(results_dir / "ucihar_robustness_summary.csv")
    checklist = claim_checklist(uci, hapt6, param_pairs, hapt12)
    checklist.to_csv(results_dir / "final_claim_checklist.csv", index=False)
    concise = concise_summary_table(uci, hapt6, param)
    concise.to_csv(results_dir / "final_core_result_snapshot.csv", index=False)

    lines = [
        "# Final Experiment Audit",
        "",
        "This report is a conservative self-audit for paper writing. It is not a submission guarantee.",
        "",
        "## Core Result Snapshot",
        "",
        dataframe_to_markdown(concise),
        "",
        "## Claim Checklist",
        "",
        dataframe_to_markdown(checklist),
        "",
        "## Parameter-Matched Finding",
        "",
        dataframe_to_markdown(param_pairs),
        "",
        "## Robustness Caveat",
        "",
    ]
    if robustness.empty:
        lines.append("- UCI-HAR robustness summary was not found.")
    else:
        lines.extend(
            [
                "- UCI-HAR robustness is useful as a supplementary analysis, not the central claim.",
                "- CMG-LIF-Lite has higher absolute Macro-F1 than LIF-SNN in several perturbations, but robustness drop is not uniformly smaller.",
                "- MS-CMG-LIF does not show a reliable robustness advantage over MS-LIF-SNN.",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended Paper Direction",
            "",
            "- Main title direction: Multi-Scale Spiking Neural Network with Lightweight Context Memory for Wearable Human Activity Recognition.",
            "- Main datasets: UCI-HAR and HAPT-6.",
            "- HAPT-12 K=2 should be a transition-aware diagnostic only.",
            "- Main performance model: MS-CMG-LIF or MS-LIF-SNN as competitive SNN variants, with Window-GRU acknowledged as strongest ANN baseline.",
            "- Lightweight context claim: CMG-LIF-Lite improves average Macro-F1 over default LIF-SNN, but parameter-matched LIF weakens the structural-only claim.",
            "- Efficiency claim: report parameters, spike rate, and operation proxy only.",
            "",
            "## Unsupported Wording To Avoid",
            "",
            "- Do not say CMG-LIF-Lite consistently beats LIF-SNN across every seed.",
            "- Do not say CMG-LIF-Lite beats a parameter-matched LIF-SNN.",
            "- Do not say MS-CMG-LIF beats all CNN/GRU baselines.",
            "- Do not say HAPT-12 K=2 is a complete 12-class benchmark.",
            "- Do not say the project measures real energy or hardware power.",
            "",
            "## Self-Prompt For The Next Work Session",
            "",
            "```text",
            "Use the current completed UCI-HAR and HAPT-6 experiments to draft the paper results section.",
            "Keep claims conservative and explicitly separate supported claims from diagnostic findings.",
            "Use UCI-HAR + HAPT-6 as main datasets, HAPT-12 K2 as a transition-aware diagnostic.",
            "Include parameter-matched LIF-SNN as a limitation: CMG-LIF-Lite improves default LIF but not widened LIF.",
            "Do not implement new architectures unless a reviewer-style gap cannot be addressed by analysis.",
            "Generate paper-ready tables for: main results, parameter-matched controls, HAPT-12 coverage, robustness, efficiency proxy.",
            "Do not claim measured power consumption.",
            "```",
        ]
    )
    report_path = results_dir / "final_experiment_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")
    print(f"Saved {results_dir / 'final_claim_checklist.csv'}")
    print(f"Saved {results_dir / 'final_core_result_snapshot.csv'}")


def main() -> None:
    write_report()


if __name__ == "__main__":
    main()
