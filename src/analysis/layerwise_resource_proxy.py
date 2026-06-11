from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
SUMMARY_PATH = V3_DIR / "main_results_summary.csv"
RAW_PATH = V3_DIR / "main_results_raw.csv"
PAIRWISE_PATH = V3_DIR / "pairwise_statistics.csv"
LAYERWISE_PATH = V3_DIR / "layerwise_resource_proxy.csv"
RESOURCE_SUMMARY_PATH = V3_DIR / "layerwise_resource_proxy_summary.csv"
TABLE_PATH = V3_DIR / "table_layerwise_resource_proxy.tex"
FIGURE_PATH = V3_DIR / "fig_macro_f1_vs_resource_proxy.png"
REPORT_PATH = V3_DIR / "layerwise_resource_proxy_report.md"
READINESS_PATH = V3_DIR / "final_readiness_report.md"
CLAIM_PATH = V3_DIR / "v3_final_claim_matrix.md"
CLAIM_CSV_PATH = V3_DIR / "v3_final_claim_matrix.csv"

DATASET_ORDER = ["ucihar", "hapt6", "pamap2", "mhealth"]
MODEL_ORDER = ["cnn1d", "window_gru", "ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
NON_SPIKING = {"cnn1d", "window_gru", "ms_ann_tcn"}


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv(RAW_PATH)
    summary_with_meta = attach_metadata(summary, raw)
    layerwise = build_layerwise_proxy(summary_with_meta)
    resource_summary = summarize_layerwise(layerwise)
    layerwise.to_csv(LAYERWISE_PATH, index=False)
    resource_summary.to_csv(RESOURCE_SUMMARY_PATH, index=False)
    write_latex_table(resource_summary)
    write_figure(resource_summary)
    write_report(resource_summary, layerwise)
    claims = build_claim_matrix()
    claims.to_csv(CLAIM_CSV_PATH, index=False)
    write_claim_matrix_markdown(claims)
    write_readiness_report(resource_summary, claims)
    print(f"Wrote layerwise resource proxy to {LAYERWISE_PATH}")
    print(f"Wrote readiness report to {READINESS_PATH}")


def attach_metadata(summary: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_key",
        "model",
        "context_len",
        "window_size",
        "num_channels",
        "num_classes",
        "hidden_dim",
        "branch_dim",
        "tcn_layers",
    ]
    available = [column for column in columns if column in raw.columns]
    meta = (
        raw[available]
        .groupby(["dataset_key", "model"], as_index=False)
        .agg({column: "median" for column in available if column not in {"dataset_key", "model"}})
    )
    return summary.merge(meta, on=["dataset_key", "model"], how="left")


def build_layerwise_proxy(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        spec = model_resource_spec(row)
        base = {
            "dataset": row["dataset"],
            "dataset_key": str(row["dataset_key"]),
            "model": str(row["model"]),
            "context_len": int(spec["context_len"]),
            "window_size": int(spec["window_size"]),
            "num_channels": int(spec["num_channels"]),
            "num_classes": int(spec["num_classes"]),
            "hidden_dim": int(spec["hidden_dim"]),
            "branch_dim": int(spec["branch_dim"]),
            "tcn_layers": int(spec["tcn_layers"]),
            "params": int(row["params"]),
            "model_size_mb": float(int(row["params"]) * 4.0 / (1024.0 * 1024.0)),
            "macro_f1_mean": float(row["macro_f1_mean"]),
            "macro_f1_std": float(row["macro_f1_std"]),
            "spike_rate_mean": maybe_float(row.get("spike_rate_mean", np.nan)),
            "num_seeds": int(row["num_seeds"]),
        }
        for component, op_type, value, note in component_estimates(spec, base["spike_rate_mean"]):
            rows.append(
                {
                    **base,
                    "component": component,
                    "op_type": op_type,
                    "ops_per_sample_proxy": value,
                    "note": note,
                }
            )
    frame = pd.DataFrame(rows)
    frame["dataset_order"] = frame["dataset_key"].map({name: idx for idx, name in enumerate(DATASET_ORDER)})
    frame["model_order"] = frame["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    return frame.sort_values(["dataset_order", "model_order", "component"]).drop(columns=["dataset_order", "model_order"])


def model_resource_spec(row: pd.Series) -> dict[str, Any]:
    model = str(row["model"])
    hidden_dim = int(round(row.get("hidden_dim", 128))) if "hidden_dim" in row else 128
    branch_dim = int(round(row.get("branch_dim", 32))) if "branch_dim" in row else 32
    if model == "ms_lif_snn_wide":
        hidden_dim = 224
        branch_dim = 64
    tcn_layers = 0
    if model in {"ms_lif_tcn", "ms_ann_tcn"}:
        tcn_layers = int(round(row.get("tcn_layers", 2))) if "tcn_layers" in row else 2
        if tcn_layers == 0:
            tcn_layers = 2
    return {
        "model": model,
        "context_len": int(row.get("context_len", 8)),
        "window_size": int(row.get("window_size", 128)),
        "num_channels": int(row.get("num_channels", 6)),
        "num_classes": int(row.get("num_classes", 6)),
        "hidden_dim": hidden_dim,
        "branch_dim": branch_dim,
        "tcn_layers": tcn_layers,
        "tcn_kernel_size": 3,
    }


def component_estimates(spec: dict[str, Any], spike_rate: float) -> list[tuple[str, str, float, str]]:
    model = spec["model"]
    k = int(spec["context_len"])
    t = int(spec["window_size"])
    c = int(spec["num_channels"])
    classes = int(spec["num_classes"])
    h = int(spec["hidden_dim"])
    b = int(spec["branch_dim"])
    tcn_layers = int(spec["tcn_layers"])
    components: list[tuple[str, str, float, str]] = []

    if model == "cnn1d":
        mid = max(32, h // 2)
        conv1 = k * t * c * mid * 5
        conv2 = k * (t // 2) * mid * h * 5
        classifier = k * h * classes
        activation = k * (t * mid + (t // 2) * h + h + classes)
        components.extend(
            [
                ("cnn_conv1", "dense_mac", float(conv1), "Conv1d over each window."),
                ("cnn_conv2", "dense_mac", float(conv2), "Post-pooling Conv1d over each window."),
                ("classifier", "dense_mac", float(classifier), "Dense classifier MACs for all context windows."),
                ("activation_memory", "activation_elements", float(activation), "Approximate retained activation elements, not bytes of a specific runtime."),
            ]
        )
        return components

    if model == "window_gru":
        conv = k * t * c * h * 5
        gru = k * 3 * (h * h + h * h + h)
        classifier = k * h * classes
        activation = k * (t * h + h + classes)
        components.extend(
            [
                ("window_conv_encoder", "dense_mac", float(conv), "Per-window dense Conv1d encoder."),
                ("cross_window_gru", "dense_mac", float(gru), "Approximate GRU MACs across context windows."),
                ("classifier", "dense_mac", float(classifier), "Dense classifier MACs for all context windows."),
                ("activation_memory", "activation_elements", float(activation), "Approximate retained activation elements, not bytes of a specific runtime."),
            ]
        )
        return components

    encoder = multiscale_encoder_macs(k, t, c, h, b)
    classifier = k * h * classes
    activation = k * (t * h + h + classes)
    components.append(("multiscale_encoder", "dense_mac", float(encoder), "Dense multi-scale Conv1d window encoder MACs."))

    if model in {"ms_ann_tcn", "ms_lif_tcn"}:
        tcn = window_tcn_macs(k, h, tcn_layers, int(spec["tcn_kernel_size"]))
        components.append(("window_tcn", "dense_mac", float(tcn), "Dense causal window-level TCN MACs; not event-driven SynOps."))
        activation += k * h

    components.append(("classifier", "dense_mac", float(classifier), "Dense classifier MACs for all context windows."))

    if model not in NON_SPIKING and not np.isnan(spike_rate):
        spike_count = float(spike_rate * k * t * h)
        synops = float(spike_count * h)
        components.extend(
            [
                ("lif_spike_count", "spike_count", spike_count, "Estimated LIF spikes from measured mean spike rate."),
                ("lif_synops_proxy", "synops_proxy", synops, "Coarse hidden-fanout SynOps proxy; not measured hardware energy."),
            ]
        )
        activation += k * t * h

    components.append(("activation_memory", "activation_elements", float(activation), "Approximate retained activation elements, not bytes of a specific runtime."))
    return components


def multiscale_encoder_macs(context_len: int, window_size: int, channels: int, hidden_dim: int, branch_dim: int) -> int:
    branch = window_size * channels * branch_dim * (3 + 5 + 9)
    project = window_size * (branch_dim * 3) * hidden_dim
    return int(context_len * (branch + project))


def window_tcn_macs(context_len: int, hidden_dim: int, tcn_layers: int, kernel_size: int = 3) -> int:
    depthwise = context_len * hidden_dim * kernel_size
    pointwise = context_len * hidden_dim * hidden_dim
    return int(tcn_layers * (depthwise + pointwise))


def summarize_layerwise(layerwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, model), group in layerwise.groupby(["dataset_key", "model"], sort=False):
        def total(op_type: str) -> float:
            values = pd.to_numeric(group[group["op_type"].eq(op_type)]["ops_per_sample_proxy"], errors="coerce")
            return float(values.sum()) if len(values) else 0.0

        dense_macs = total("dense_mac")
        spike_count = total("spike_count")
        synops = total("synops_proxy")
        activation_elements = total("activation_elements")
        first = group.iloc[0]
        rows.append(
            {
                "dataset": first["dataset"],
                "dataset_key": dataset_key,
                "model": model,
                "num_seeds": int(first["num_seeds"]),
                "macro_f1_mean": float(first["macro_f1_mean"]),
                "macro_f1_std": float(first["macro_f1_std"]),
                "params": int(first["params"]),
                "model_size_mb": float(first["model_size_mb"]),
                "spike_rate_mean": maybe_float(first["spike_rate_mean"]),
                "dense_macs_per_sample_proxy": dense_macs,
                "lif_spike_count_per_sample_proxy": spike_count if spike_count > 0 else np.nan,
                "synops_per_sample_proxy": synops if synops > 0 else np.nan,
                "activation_elements_per_sample_proxy": activation_elements,
                "activation_memory_mb_proxy": activation_elements * 4.0 / (1024.0 * 1024.0),
                "resource_note": "algorithmic_proxy_not_measured_energy",
            }
        )
    frame = pd.DataFrame(rows)
    frame["dataset_order"] = frame["dataset_key"].map({name: idx for idx, name in enumerate(DATASET_ORDER)})
    frame["model_order"] = frame["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    return frame.sort_values(["dataset_order", "model_order"]).drop(columns=["dataset_order", "model_order"])


def write_latex_table(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "dataset_key",
            "model",
            "macro_f1_mean",
            "params",
            "model_size_mb",
            "spike_rate_mean",
            "dense_macs_per_sample_proxy",
            "lif_spike_count_per_sample_proxy",
            "synops_per_sample_proxy",
        ]
    ].copy()
    table.to_latex(TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dataset in DATASET_ORDER:
        group = summary[summary["dataset_key"].eq(dataset)]
        axes[0].scatter(group["dense_macs_per_sample_proxy"], group["macro_f1_mean"], label=dataset)
        snn = group[group["synops_per_sample_proxy"].notna()]
        axes[1].scatter(snn["synops_per_sample_proxy"], snn["macro_f1_mean"], label=dataset)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Dense MACs/sample proxy (log scale)")
    axes[0].set_ylabel("Macro-F1 mean")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("SNN SynOps/sample proxy (log scale)")
    axes[1].set_ylabel("Macro-F1 mean")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Macro-F1 vs algorithmic resource proxies")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_report(summary: pd.DataFrame, layerwise: pd.DataFrame) -> None:
    lines = [
        "# Layerwise Resource Proxy",
        "",
        "This is algorithmic resource characterization, not measured hardware energy.",
        "MS-LIF-TCN is decomposed as a dense multi-scale encoder, LIF spike representation, dense causal window-level TCN, and dense classifier.",
        "The dense TCN is reported as MACs, not event-driven SynOps.",
        "",
        "## Completeness",
        "",
        f"- Layerwise component rows: {len(layerwise)}.",
        f"- Summary rows: {len(summary)}.",
        "- Models: cnn1d, window_gru, ms_lif_snn, ms_lif_snn_wide, ms_ann_tcn, ms_lif_tcn.",
        "- Datasets: ucihar, hapt6, pamap2, mhealth.",
        "",
        "## Summary",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "model",
                    "macro_f1_mean",
                    "params",
                    "spike_rate_mean",
                    "dense_macs_per_sample_proxy",
                    "synops_per_sample_proxy",
                    "activation_memory_mb_proxy",
                ]
            ]
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- Dense MACs are estimated from layer dimensions and sequence length; they are not profiler measurements.",
        "- SynOps are a coarse proxy derived from mean spike rate and hidden fanout; they are not measured neuromorphic hardware energy.",
        "- The dense TCN and classifier remain dense operations in the current PyTorch implementation.",
        "- Use this table to discuss resource characterization, not low-power hardware claims.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_claim_matrix() -> pd.DataFrame:
    pairwise = pd.read_csv(PAIRWISE_PATH) if PAIRWISE_PATH.exists() else pd.DataFrame()
    claims: list[dict[str, str]] = []
    claims.append(
        claim_from_pairwise(
            pairwise,
            "MS-LIF-TCN has higher 10-seed mean Macro-F1 than compact MS-LIF-SNN.",
            "ms_lif_tcn - ms_lif_snn",
            expected_positive=4,
            support_when_positive="supported",
        )
    )
    claims.append(
        claim_from_pairwise(
            pairwise,
            "MS-LIF-TCN has higher 10-seed mean Macro-F1 than parameter-matched MS-LIF-SNN-wide.",
            "ms_lif_tcn - ms_lif_snn_wide",
            expected_positive=4,
            support_when_positive="supported",
        )
    )
    claims.append(
        claim_from_pairwise(
            pairwise,
            "MS-LIF-TCN universally outperforms MS-ANN-TCN.",
            "ms_lif_tcn - ms_ann_tcn",
            expected_positive=4,
            support_when_positive="supported",
            default_not_all_positive="not_supported",
        )
    )
    claims.append(
        claim_from_pairwise(
            pairwise,
            "MS-LIF-TCN universally outperforms Window-GRU.",
            "ms_lif_tcn - window_gru",
            expected_positive=4,
            support_when_positive="supported",
            default_not_all_positive="not_supported",
        )
    )
    claims.extend(
        [
            {
                "claim": "Aligned context-length diagnostic supports cross-window context.",
                "support": "diagnostic_supported",
                "evidence": "MS-LIF-TCN and MS-ANN-TCN have best K > 1 on 4/4 datasets under aligned_kmax=8 reduced diagnostic.",
            },
            {
                "claim": "Window-level TCN block is not merely decorative.",
                "support": "diagnostic_supported",
                "evidence": "TCN-depth diagnostic shows at least one TCN depth above TCN-0 on 4/4 datasets.",
            },
            {
                "claim": "Multi-scale encoder is the main independent performance source.",
                "support": "not_supported_as_primary",
                "evidence": "Single-scale diagnostic shows multi-scale is best on 1/4 datasets but above average single-k variants on 4/4; treat as robust design component.",
            },
            {
                "claim": "Spike regularization gives strong energy-efficiency gains.",
                "support": "not_supported",
                "evidence": "Spike-reg diagnostic shows only small sparsity effects and no measured energy.",
            },
            {
                "claim": "Transition diagnostic supports HAPT-12 as a full 12-class primary result.",
                "support": "not_supported",
                "evidence": "HAPT-12 support audit has low/zero transition-class support after K filtering.",
            },
            {
                "claim": "Transition binary diagnostic suggests temporal context helps transition-sensitive settings.",
                "support": "diagnostic_supported",
                "evidence": "K=2 support is adequate and temporal models improve over SNN-wide; K=4 is weak-support diagnostic.",
            },
            {
                "claim": "Layerwise resource proxy supports algorithmic resource characterization.",
                "support": "supported_as_proxy",
                "evidence": "Dense MACs, LIF spike count, SynOps proxy, parameters, and activation proxy are reported separately.",
            },
            {
                "claim": "SNN has measured neuromorphic low-power hardware advantage.",
                "support": "not_supported",
                "evidence": "No neuromorphic hardware measurement is performed; resource and spike results are proxies.",
            },
        ]
    )
    return pd.DataFrame(claims)


def claim_from_pairwise(
    pairwise: pd.DataFrame,
    claim: str,
    comparison: str,
    expected_positive: int,
    support_when_positive: str,
    default_not_all_positive: str = "mixed",
) -> dict[str, str]:
    if pairwise.empty or "comparison" not in pairwise.columns:
        return {"claim": claim, "support": "not_evaluated", "evidence": "Pairwise statistics unavailable."}
    subset = pairwise[pairwise["comparison"].astype(str).eq(comparison)]
    if subset.empty:
        return {"claim": claim, "support": "not_evaluated", "evidence": f"No rows for {comparison}."}
    positive = int((subset["mean_delta_macro_f1"] > 0).sum())
    evidence = "; ".join(
        f"{row.dataset_key}: Δ={float(row.mean_delta_macro_f1):.4f}, wins={int(row.win_count)}/{int(row.num_pairs)}"
        for row in subset.itertuples()
    )
    support = support_when_positive if positive >= expected_positive else default_not_all_positive
    return {"claim": claim, "support": support, "evidence": evidence}


def write_claim_matrix_markdown(claims: pd.DataFrame) -> None:
    lines = [
        "# v3 Final Claim Matrix",
        "",
        "This matrix constrains the v3 manuscript wording. Claims not supported here should not be promoted in the abstract or conclusion.",
        "",
        markdown_table(claims),
        "",
    ]
    CLAIM_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_readiness_report(resource_summary: pd.DataFrame, claims: pd.DataFrame) -> None:
    status_rows = [
        ("10-seed main run", "complete"),
        ("per-class/confusion analysis", "complete"),
        ("aligned context-length diagnostic", "complete"),
        ("TCN-depth diagnostic", "complete"),
        ("spike-regularization diagnostic", "complete_weak_evidence"),
        ("single-scale diagnostic", "complete_design_component"),
        ("HAPT transition diagnostic", "complete_supplementary"),
        ("layerwise resource proxy", "complete_proxy_only"),
        ("v3 manuscript rewrite", "not_started"),
        ("submission metadata", "requires_manual_completion"),
    ]
    lines = [
        "# v3 Final Readiness Report",
        "",
        "Status: experiment package is complete for v3 evidence synthesis; manuscript rewrite and submission metadata are still pending.",
        "",
        "## Evidence Status",
        "",
        markdown_table(pd.DataFrame(status_rows, columns=["item", "status"])),
        "",
        "## Safe Core Claim",
        "",
        "Across four subject-disjoint HAR datasets and ten seeds, `ms_lif_tcn` has higher mean Macro-F1 than compact and parameter-matched SNN baselines, while remaining competitive rather than universally superior to strong ANN temporal baselines.",
        "",
        "## Resource Boundary",
        "",
        "The layerwise resource table reports algorithmic dense MACs, LIF spike count, SynOps proxy, parameter count, model size, and activation-memory proxy. It is not measured hardware energy and must not be written as neuromorphic low-power evidence.",
        "",
        "## Claims",
        "",
        markdown_table(claims),
        "",
        "## Remaining Writing Work",
        "",
        "- Rewrite the manuscript from v2 to v3 around causal window-level spiking context modeling.",
        "- Consider retitling away from a primary multi-scale claim.",
        "- Update all tables/figures to the v3 package.",
        "- Fill author order, affiliations, funding, data availability, ethics/AI-use statements, and code link manually.",
        "- Keep HAPT-12 multiclass as skipped/unsupported due transition support; use transition-binary only as supplementary diagnostic.",
        "",
    ]
    READINESS_PATH.write_text("\n".join(lines), encoding="utf-8")


def maybe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows = [[format_cell(row[column]) for column in headers] for _, row in df.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(float(value)):
            return "N/A"
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
