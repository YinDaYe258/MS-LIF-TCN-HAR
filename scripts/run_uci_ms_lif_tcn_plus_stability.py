from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


CONFIG_PATH = "configs/ucihar_ms_lif_tcn_attn.yaml"
OUTPUT_PATH = Path("results/uci_ms_lif_tcn_plus_stability.csv")
SUMMARY_PATH = Path("results/uci_ms_lif_tcn_plus_stability_summary.csv")
REPORT_PATH = Path("results/uci_ms_lif_tcn_plus_stability_report.md")

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_ms_lif_tcn": {
        "model": "ms_lif_tcn",
        "attention_enabled": False,
        "attention_hidden_dim": 0,
        "supcon_enabled": False,
        "supcon_weight": 0.0,
        "tcn_layers": 2,
        "reuse_variant": "baseline_ms_lif_tcn",
    },
    "attn_only": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 64,
        "supcon_enabled": False,
        "supcon_weight": 0.0,
        "tcn_layers": 2,
        "reuse_variant": "attn_ce",
    },
    "supcon_only_0.05": {
        "model": "ms_lif_tcn",
        "attention_enabled": False,
        "attention_hidden_dim": 0,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "tcn_layers": 2,
    },
    "attn_supcon_0.03": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 64,
        "supcon_enabled": True,
        "supcon_weight": 0.03,
        "tcn_layers": 2,
    },
    "attn_supcon_0.05": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 64,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "tcn_layers": 2,
        "reuse_variant": "attn_supcon_0.05",
    },
    "attn_supcon_0.07": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 64,
        "supcon_enabled": True,
        "supcon_weight": 0.07,
        "tcn_layers": 2,
    },
    "attn_supcon_0.05_smallattn": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 16,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "tcn_layers": 2,
    },
    "attn_supcon_0.05_tcn1": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 64,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "tcn_layers": 1,
    },
    "attn_supcon_0.05_smallattn_tcn1": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "attention_hidden_dim": 16,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "tcn_layers": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCI-HAR MS-LIF-TCN+ stability diagnostics.")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for variant_name in selected_variants(args.variants):
        if not args.force and row_exists(output, variant_name, int(args.seed)):
            print(f"Skipping existing stability row: {variant_name} seed{args.seed}")
            continue
        reused = load_reusable_row(variant_name, int(args.seed))
        if reused is not None and not args.force:
            append_csv_row(output, reused)
            print(f"Reused plus diagnostic row for {variant_name}")
            continue
        row = run_one(variant_name, int(args.seed), args)
        append_csv_row(output, row)
        print(f"Saved row to {output}")
        print(row)
    write_report(pd.read_csv(output))


def selected_variants(args: list[str]) -> list[str]:
    if len(args) == 1 and args[0].lower() == "all":
        return list(VARIANTS)
    unknown = [name for name in args if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return args


def run_one(variant_name: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    variant = VARIANTS[variant_name]
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["seed"] = int(seed)
    apply_variant_config(config, variant)
    apply_arg_overrides(config, args)
    set_seed(seed)
    loaders, meta = create_ucihar_dataloaders(config, model_name=str(variant["model"]), smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(str(variant["model"]), meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"uci_stability_{variant_name}_k{meta.context_len}_{target_mode}_seed{seed}"
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        run_name,
        results_dir=config.get("results", {}).get("dir", "results"),
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    return make_row(variant_name, variant, seed, meta.context_len, target_mode, count_parameters(model), metrics)


def apply_variant_config(config: dict[str, Any], variant: dict[str, Any]) -> None:
    model_cfg = config.setdefault("model", {})
    model_cfg["attention_hidden_dim"] = int(variant["attention_hidden_dim"] or model_cfg.get("attention_hidden_dim", 64))
    model_cfg["tcn_layers"] = int(variant["tcn_layers"])
    training = config.setdefault("training", {})
    training["loss_type"] = "ce"
    training["class_weight_source"] = "train_labels"
    supcon = training.setdefault("aux_loss", {}).setdefault("supervised_contrastive", {})
    supcon["enabled"] = bool(variant["supcon_enabled"])
    supcon["weight"] = float(variant["supcon_weight"])
    supcon.setdefault("temperature", 0.2)
    training.setdefault("augmentation", {})["enabled"] = False


def apply_arg_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)


def make_row(
    variant_name: str,
    variant: dict[str, Any],
    seed: int,
    context_len: int,
    target_mode: str,
    params: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    val_test_gap = float(metrics["best_val_macro_f1"]) - float(metrics["macro_f1"])
    return {
        "dataset": "UCI-HAR",
        "variant": variant_name,
        "model": variant["model"],
        "seed": int(seed),
        "context_len": int(context_len),
        "target_mode": target_mode,
        "attention_enabled": bool(variant["attention_enabled"]),
        "attention_hidden_dim": int(variant["attention_hidden_dim"]),
        "supcon_enabled": bool(variant["supcon_enabled"]),
        "supcon_weight": float(variant["supcon_weight"]),
        "tcn_layers": int(variant["tcn_layers"]),
        "params": int(params),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "spike_rate": metrics.get("spike_rate", 0.0),
        "best_epoch": metrics["best_epoch"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "val_test_macro_f1_gap": val_test_gap,
        "passes_decision_rule": False,
        "checkpoint": metrics["checkpoint"],
        "epoch_log": metrics["epoch_log"],
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
    }


def row_exists(path: Path, variant: str, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"variant", "seed"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    return bool(rows["variant"].astype(str).eq(variant).mul(rows["seed"].astype(int).eq(int(seed))).any())


def load_reusable_row(variant_name: str, seed: int) -> dict[str, Any] | None:
    reuse_variant = VARIANTS[variant_name].get("reuse_variant")
    if not reuse_variant:
        return None
    source = Path("results/ms_lif_tcn_plus_diagnostic.csv")
    if not source.exists():
        return None
    rows = pd.read_csv(source)
    match = rows[
        rows["dataset_key"].astype(str).eq("ucihar")
        & rows["variant"].astype(str).eq(str(reuse_variant))
        & rows["seed"].astype(int).eq(int(seed))
    ]
    if match.empty:
        return None
    source_row = match.iloc[-1].to_dict()
    variant = VARIANTS[variant_name]
    metrics = {
        "accuracy": float(source_row["accuracy"]),
        "macro_f1": float(source_row["macro_f1"]),
        "weighted_f1": float(source_row["weighted_f1"]),
        "balanced_accuracy": float(source_row.get("balanced_accuracy", 0.0)),
        "spike_rate": float(source_row.get("spike_rate", 0.0)),
        "best_epoch": int(source_row.get("best_epoch", 0)),
        "best_val_macro_f1": float(source_row.get("best_val_macro_f1", 0.0)),
        "checkpoint": source_row.get("checkpoint", ""),
        "epoch_log": source_row.get("epoch_log", ""),
        "confusion_matrix_path": source_row.get("confusion_matrix_path", ""),
    }
    return make_row(
        variant_name,
        variant,
        seed,
        int(source_row["context_len"]),
        str(source_row.get("target_mode", "last")),
        int(source_row["params"]),
        metrics,
    )


def write_report(rows: pd.DataFrame) -> None:
    rows = rows.copy()
    for column in ("macro_f1", "best_val_macro_f1", "val_test_macro_f1_gap", "spike_rate", "params"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    baseline = rows[rows["variant"].astype(str).eq("baseline_ms_lif_tcn")]
    baseline_macro = float(baseline.iloc[-1]["macro_f1"]) if not baseline.empty else float("nan")
    current_plus_params = 61895
    rows["passes_decision_rule"] = (
        (rows["macro_f1"] >= baseline_macro + 0.005)
        & (rows["params"] <= current_plus_params)
        & (rows["spike_rate"] <= 0.30)
        & (rows["val_test_macro_f1_gap"] <= 0.03)
    )
    rows.to_csv(OUTPUT_PATH, index=False)
    summary = build_variant_summary(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    best = rows.sort_values(["passes_decision_rule", "macro_f1"], ascending=[False, False]).iloc[0]
    lines = [
        "# UCI MS-LIF-TCN+ Stability Report",
        "",
        "This diagnostic only targets UCI-HAR stability and parameter overhead.",
        "It does not claim measured neuromorphic power.",
        "",
        f"Baseline `ms_lif_tcn` seed42 Macro-F1: {baseline_macro:.4f}",
        "",
        "## Results",
        "",
        dataframe_to_markdown(
            rows[
                [
                    "variant",
                    "macro_f1",
                    "best_val_macro_f1",
                    "val_test_macro_f1_gap",
                    "params",
                    "spike_rate",
                    "passes_decision_rule",
                ]
            ].sort_values("macro_f1", ascending=False)
        ),
        "",
        "## Variant Summary",
        "",
        dataframe_to_markdown(
            summary[
                [
                    "variant",
                    "num_seeds",
                    "seeds",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "macro_f1_min",
                    "macro_f1_max",
                    "params",
                    "spike_rate_mean",
                ]
            ]
        ),
        "",
        "## Recommendation",
        "",
    ]
    expanded = summary[summary["variant"].astype(str).eq("attn_supcon_0.05")]
    if not expanded.empty and int(expanded.iloc[0]["num_seeds"]) >= 3:
        expanded_row = expanded.iloc[0]
        lines.extend(
            [
                (
                    "- `attn_supcon_0.05` was expanded to seeds 42/43/44. "
                    f"Its Macro-F1 is {expanded_row['macro_f1_mean']:.4f} +/- "
                    f"{expanded_row['macro_f1_std']:.4f}, with a minimum of "
                    f"{expanded_row['macro_f1_min']:.4f}."
                ),
                (
                    "- The seed44 drop remains substantial, so this variant does not solve "
                    "UCI-HAR stability. Keep `MS-LIF-TCN` as the global main model."
                ),
                (
                    "- Keep `MS-LIF-TCN+` as an enhanced/diagnostic variant, especially for "
                    "HAPT-6 where its multiseed result is stronger."
                ),
            ]
        )
    elif bool(best["passes_decision_rule"]):
        lines.append(
            f"- `{best['variant']}` passes the seed42 decision rule and is worth expanding to seeds 43/44."
        )
    else:
        lines.append(
            "- No variant passes the full decision rule. Keep `MS-LIF-TCN` as the global main model and keep `MS-LIF-TCN+` as an HAPT-6 enhanced diagnostic."
        )
    lines.extend(
        [
            "- Do not select a variant based only on test Macro-F1 if its validation-test gap is large.",
            "- A smaller attention head or one-layer TCN may reduce parameters, but it should only be kept if it also improves stability.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_variant_summary(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for column in ("seed", "macro_f1", "params", "spike_rate"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    def seed_list(series: pd.Series) -> str:
        seeds = sorted({int(seed) for seed in series.dropna().tolist()})
        return ",".join(str(seed) for seed in seeds)

    summary = (
        rows.groupby("variant", as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            seeds=("seed", seed_list),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            params=("params", "mean"),
            spike_rate_mean=("spike_rate", "mean"),
            spike_rate_std=("spike_rate", "std"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )
    summary["macro_f1_std"] = summary["macro_f1_std"].fillna(0.0)
    summary["spike_rate_std"] = summary["spike_rate_std"].fillna(0.0)
    summary["params"] = summary["params"].round().astype(int)
    return summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
