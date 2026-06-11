from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


V2_DIR = Path("results/final_paper_v2")
ARTIFACT_DIR = V2_DIR / "artifacts"
RAW_PATH = V2_DIR / "main_results_raw.csv"
SUMMARY_PATH = V2_DIR / "main_results_summary.csv"
TABLE_PATH = V2_DIR / "table_main_results_summary.tex"
REPORT_PATH = V2_DIR / "v2_experiment_report.md"

DATASETS: dict[str, dict[str, Any]] = {
    "ucihar": {
        "display": "UCI-HAR",
        "task": "ucihar",
        "base_config": "configs/ucihar_k8_last.yaml",
        "tcn_config": "configs/ucihar_ms_tcn_snn.yaml",
        "plus_config": "configs/ucihar_ms_lif_tcn_attn.yaml",
        "loader": create_ucihar_dataloaders,
    },
    "hapt6": {
        "display": "HAPT",
        "task": "hapt6",
        "base_config": "configs/hapt6_k8_last.yaml",
        "tcn_config": "configs/hapt6_ms_tcn_snn.yaml",
        "plus_config": "configs/hapt6_ms_lif_tcn_attn.yaml",
        "loader": create_hapt_dataloaders,
    },
}

MAIN_MODELS = ["cnn1d", "gru", "window_gru", "ms_lif_snn", "ms_lif_tcn", "ms_lif_tcn_plus"]
CONTROL_MODELS = ["ms_lif_snn_wide", "ms_ann_tcn"]
NON_SPIKING_MODELS = {"cnn1d", "gru", "window_gru", "ms_ann_tcn"}
MODEL_ORDER = MAIN_MODELS + CONTROL_MODELS
SOURCE_FILES = [
    Path("results/ucihar_formal_multiseed_results.csv"),
    Path("results/hapt6_multiseed_results.csv"),
    Path("results/ms_tcn_multiseed_results.csv"),
    Path("results/ms_lif_tcn_plus_multiseed.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled final_paper_v2 extensions.")
    parser.add_argument("--datasets", nargs="+", default=["ucihar", "hapt6"], choices=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["main"],
        help="Model names, or groups: main, controls, all.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V2_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    selected = selected_models(args.models)
    if not args.summarize_only:
        for dataset_key in args.datasets:
            for seed in args.seeds:
                for model_label in selected:
                    if not args.force and row_exists(RAW_PATH, dataset_key, model_label, seed):
                        print(f"Skipping v2 existing row: {dataset_key} {model_label} seed{seed}")
                        continue
                    if not args.force:
                        reused = find_existing_row(dataset_key, model_label, seed)
                        if reused is not None:
                            append_csv_row(RAW_PATH, reused)
                            print(f"Reused existing row: {dataset_key} {model_label} seed{seed}")
                            continue
                    row = run_one(dataset_key, model_label, seed, args)
                    append_csv_row(RAW_PATH, row)
                    print(f"Saved v2 row: {dataset_key} {model_label} seed{seed}")
    summarize()


def selected_models(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        normalized = value.lower().replace("-", "_")
        if normalized == "main":
            expanded.extend(MAIN_MODELS)
        elif normalized == "controls":
            expanded.extend(CONTROL_MODELS)
        elif normalized == "all":
            expanded.extend(MODEL_ORDER)
        elif normalized in MODEL_ORDER:
            expanded.append(normalized)
        else:
            raise ValueError(f"Unknown v2 model/group: {value}")
    deduped: list[str] = []
    for model in expanded:
        if model not in deduped:
            deduped.append(model)
    return deduped


def run_one(dataset_key: str, model_label: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(dataset_key, model_label, seed, args)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    build_name = build_name_for(model_label)
    loaders, meta = loader_fn(config, model_name=build_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(build_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"v2_{dataset_key}_{model_label}_k{meta.context_len}_{target_mode}_seed{seed}"
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        run_name,
        results_dir=ARTIFACT_DIR,
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    return make_row(
        dataset_key=dataset_key,
        spec=spec,
        model_label=model_label,
        build_name=build_name,
        seed=seed,
        context_len=int(meta.context_len),
        target_mode=target_mode,
        synthetic_data=bool(meta.synthetic),
        params=count_parameters(model),
        metrics=metrics,
        config=config,
        source="trained_v2",
    )


def make_config(dataset_key: str, model_label: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    if model_label == "ms_lif_tcn_plus":
        config_path = spec["plus_config"]
    elif model_label in {"ms_lif_tcn", "ms_ann_tcn"}:
        config_path = spec["tcn_config"]
    else:
        config_path = spec["base_config"]
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("results", {})["dir"] = str(ARTIFACT_DIR)
    if model_label == "ms_lif_snn_wide":
        # Close to MS-LIF-TCN's ~52-54k params without adding cross-window TCN.
        config.setdefault("model", {})["hidden_dim"] = 224
        config.setdefault("model", {})["branch_dim"] = 64
    if model_label == "ms_lif_tcn_plus":
        apply_plus_variant(config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def apply_plus_variant(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    aux_loss = training.setdefault("aux_loss", {})
    supcon = aux_loss.setdefault("supervised_contrastive", {})
    supcon["enabled"] = True
    supcon["weight"] = 0.10
    supcon["temperature"] = 0.2
    training["loss_type"] = "ce"
    aug = training.setdefault("augmentation", {})
    aug["enabled"] = False


def build_name_for(model_label: str) -> str:
    if model_label == "ms_lif_tcn_plus":
        return "ms_lif_tcn_attn"
    if model_label == "ms_lif_snn_wide":
        return "ms_lif_snn"
    return model_label


def make_row(
    dataset_key: str,
    spec: dict[str, Any],
    model_label: str,
    build_name: str,
    seed: int,
    context_len: int,
    target_mode: str,
    synthetic_data: bool,
    params: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    source: str,
    variant: str = "",
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    spike_rate: float | None = None
    if model_label not in NON_SPIKING_MODELS:
        spike_rate = float(metrics.get("spike_rate", 0.0))
    return {
        "dataset": spec["display"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "model": model_label,
        "build_model": build_name,
        "variant": variant,
        "seed": int(seed),
        "context_len": int(context_len),
        "target_mode": target_mode,
        "synthetic_data": synthetic_data,
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "branch_dim": int(model_cfg.get("branch_dim", 32)),
        "params": int(params),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "loss": float(metrics.get("loss", 0.0)),
        "spike_rate": spike_rate,
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "best_val_macro_f1": float(metrics.get("best_val_macro_f1", 0.0)),
        "checkpoint": metrics.get("checkpoint", ""),
        "epoch_log": metrics.get("epoch_log", ""),
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
        "source": source,
    }


def row_exists(path: Path, dataset_key: str, model_label: str, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    required = {"dataset_key", "model", "seed"}
    if not required.issubset(rows.columns):
        return False
    mask = (
        rows["dataset_key"].astype(str).eq(dataset_key)
        & rows["model"].astype(str).eq(model_label)
        & rows["seed"].astype(int).eq(int(seed))
    )
    return bool(mask.any())


def find_existing_row(dataset_key: str, model_label: str, seed: int) -> dict[str, Any] | None:
    source_rows = load_source_rows()
    if source_rows.empty:
        return None
    mask = (
        source_rows["dataset_key"].astype(str).eq(dataset_key)
        & source_rows["model"].astype(str).eq(model_label)
        & source_rows["seed"].astype(int).eq(int(seed))
    )
    match = source_rows[mask]
    if match.empty:
        return None
    return match.iloc[-1].to_dict()


def load_source_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            normalized = normalize_source_row(row.to_dict(), path)
            if normalized is not None:
                rows.append(normalized)
    return pd.DataFrame(rows)


def normalize_source_row(row: dict[str, Any], source_path: Path) -> dict[str, Any] | None:
    dataset_key = infer_dataset_key(row)
    if dataset_key not in DATASETS:
        return None
    source_model = str(row.get("model", "")).lower()
    variant = str(row.get("variant", ""))
    if source_model == "ms_lif_tcn_attn" and variant == "attn_supcon_0.1":
        model_label = "ms_lif_tcn_plus"
    else:
        model_label = source_model
    if model_label not in MODEL_ORDER:
        return None
    spec = DATASETS[dataset_key]
    hidden_dim = row.get("hidden_dim", 128)
    branch_dim = row.get("branch_dim", 32)
    spike_rate = row.get("spike_rate", None)
    if model_label in NON_SPIKING_MODELS:
        spike_rate = None
    balanced_accuracy = row.get("balanced_accuracy", None)
    if is_missing(balanced_accuracy):
        balanced_accuracy = balanced_accuracy_from_confusion(row.get("confusion_matrix_path", ""))
    return {
        "dataset": spec["display"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "model": model_label,
        "build_model": source_model,
        "variant": variant,
        "seed": int(row["seed"]),
        "context_len": int(row.get("context_len", 8)),
        "target_mode": str(row.get("target_mode", "last")),
        "synthetic_data": bool_from_value(row.get("synthetic_data", False)),
        "hidden_dim": int(float(hidden_dim)) if not is_missing(hidden_dim) else 128,
        "branch_dim": int(float(branch_dim)) if not is_missing(branch_dim) else 32,
        "params": int(row["params"]),
        "accuracy": float(row["accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "weighted_f1": float(row["weighted_f1"]),
        "balanced_accuracy": float(0.0 if is_missing(balanced_accuracy) else balanced_accuracy),
        "loss": float(row.get("loss", 0.0)),
        "spike_rate": None if is_missing(spike_rate) else float(spike_rate),
        "best_epoch": int(float(row.get("best_epoch", 0))),
        "best_val_macro_f1": float(row.get("best_val_macro_f1", 0.0)),
        "checkpoint": row.get("checkpoint", ""),
        "epoch_log": row.get("epoch_log", ""),
        "confusion_matrix_path": row.get("confusion_matrix_path", ""),
        "source": str(source_path),
    }


def infer_dataset_key(row: dict[str, Any]) -> str:
    if "dataset_key" in row and not is_missing(row["dataset_key"]):
        return str(row["dataset_key"])
    dataset = str(row.get("dataset", "")).lower()
    task = str(row.get("task", "")).lower()
    if "uci" in dataset:
        return "ucihar"
    if task == "hapt6" or "hapt" in dataset:
        return "hapt6"
    return ""


def balanced_accuracy_from_confusion(path_value: Any) -> float | None:
    if is_missing(path_value):
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    try:
        matrix = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))).to_numpy(dtype=float)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    support = matrix.sum(axis=1)
    valid = support > 0
    if not valid.any():
        return None
    recall = matrix.diagonal()[valid] / support[valid]
    return float(recall.mean())


def summarize() -> None:
    if not RAW_PATH.exists():
        print(f"No v2 raw rows at {RAW_PATH}")
        return
    rows = pd.read_csv(RAW_PATH)
    if rows.empty:
        print(f"No v2 rows to summarize at {RAW_PATH}")
        return
    summary_rows: list[dict[str, Any]] = []
    for (dataset_key, model), group in rows.groupby(["dataset_key", "model"], sort=False):
        group = group.sort_values("seed")
        summary_rows.append(
            {
                "dataset": group["dataset"].iloc[0],
                "dataset_key": dataset_key,
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "seeds": " ".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1) if len(group) > 1 else 0.0,
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1) if len(group) > 1 else 0.0,
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1) if len(group) > 1 else 0.0,
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std": group["balanced_accuracy"].std(ddof=1) if len(group) > 1 else 0.0,
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].dropna().mean() if group["spike_rate"].notna().any() else math.nan,
                "spike_rate_std": group["spike_rate"].dropna().std(ddof=1)
                if group["spike_rate"].dropna().shape[0] > 1
                else 0.0,
                "best_epoch_mean": group["best_epoch"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["model_order"] = summary["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    summary["dataset_order"] = summary["dataset_key"].map({"ucihar": 0, "hapt6": 1})
    summary = summary.sort_values(["dataset_order", "model_order", "model"]).drop(columns=["dataset_order", "model_order"])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_latex(summary)
    write_report(summary)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {REPORT_PATH}")


def write_latex(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Model & Seeds & Macro-F1 & Bal. Acc. & Params & Spike Rate \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        spike = "N/A" if is_missing(row["spike_rate_mean"]) else mean_std(row["spike_rate_mean"], row["spike_rate_std"], 4)
        lines.append(
            f"{escape_latex(row['dataset'])} & {escape_latex(row['model'])} & {int(row['num_seeds'])} & "
            f"{mean_std(row['macro_f1_mean'], row['macro_f1_std'])} & "
            f"{mean_std(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])} & "
            f"{int(row['params']):,} & {spike} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Final Paper V2 Extension Report",
        "",
        "This directory extends the locked v1 result package without overwriting `results/final_paper/`.",
        "",
        "## Scope",
        "",
        "- Adds controlled 5-seed rows when available for the final main models.",
        "- Adds `ms_lif_snn_wide` as a parameter-matched MS-LIF-SNN control.",
        "- Adds `ms_ann_tcn` / `ms_cnn_tcn` as a non-spiking structural TCN control.",
        "- CMG variants remain diagnostic and are not expanded here.",
        "",
        "## Summary",
        "",
    ]
    for dataset_key in ["ucihar", "hapt6"]:
        subset = summary[summary["dataset_key"].eq(dataset_key)]
        if subset.empty:
            continue
        best = subset.sort_values("macro_f1_mean", ascending=False).iloc[0]
        lines.append(
            f"- {best['dataset']}: best current v2 mean Macro-F1 is `{best['model']}` "
            f"({best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}, n={int(best['num_seeds'])})."
        )
    lines.extend(["", "## Caveats", "", "- V2 rows should replace v1 claims only after all selected models have equal seed coverage."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def escape_latex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


if __name__ == "__main__":
    main()
