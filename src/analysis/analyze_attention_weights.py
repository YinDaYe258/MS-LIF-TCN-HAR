from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.utils import build_model, get_device, load_config


RESULTS_DIR = Path("results")
ATTENTION_DIR = RESULTS_DIR / "attention_analysis"
INPUT_CSV = RESULTS_DIR / "ms_lif_tcn_plus_multiseed.csv"
SUMMARY_CSV = ATTENTION_DIR / "attention_weights_summary.csv"
FIG_PATH = ATTENTION_DIR / "fig_attention_by_class.png"


CONFIGS = {
    "ucihar": "configs/ucihar_ms_lif_tcn_attn.yaml",
    "hapt6": "configs/hapt6_ms_lif_tcn_attn.yaml",
}


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing plus multiseed CSV: {INPUT_CSV}")
    rows = pd.read_csv(INPUT_CSV)
    selected = rows[rows["variant"].astype(str).eq("attn_supcon_0.1")]
    if selected.empty:
        raise ValueError("No attn_supcon_0.1 rows found for attention analysis")
    ATTENTION_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected.itertuples(index=False):
        records.extend(analyze_one(row))
    summary = pd.DataFrame(records)
    summary.to_csv(SUMMARY_CSV, index=False)
    plot_attention(summary)
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {FIG_PATH}")


def analyze_one(row) -> list[dict]:
    dataset_key = str(row.dataset_key)
    config = load_config(CONFIGS[dataset_key])
    if dataset_key == "ucihar":
        loaders, meta = create_ucihar_dataloaders(config, model_name="ms_lif_tcn_attn", smoke_test=False)
    elif dataset_key == "hapt6":
        loaders, meta = create_hapt_dataloaders(config, model_name="ms_lif_tcn_attn", smoke_test=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    device = get_device(config.get("device", "auto"))
    model = build_model("ms_lif_tcn_attn", meta.num_channels, meta.num_classes, config.get("model", {})).to(device)
    checkpoint = torch.load(str(row.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    overall_sums = torch.zeros(meta.context_len, dtype=torch.float64)
    overall_count = 0
    with torch.no_grad():
        for batch in loaders["test"]:
            x = batch["x"].to(device)
            labels = batch["y"][:, -1].to(torch.long)
            outputs = model(x)
            weights = outputs["attention_weights"].detach().cpu().to(torch.float64)
            overall_sums += weights.sum(dim=0)
            overall_count += int(weights.shape[0])
            for class_id in range(meta.num_classes):
                mask = labels.cpu().eq(class_id)
                if not mask.any():
                    continue
                class_weights = weights[mask]
                for pos in range(meta.context_len):
                    key = (class_id, pos)
                    sums[key] = sums.get(key, 0.0) + float(class_weights[:, pos].sum().item())
                    counts[key] = counts.get(key, 0) + int(class_weights.shape[0])

    records = []
    for pos in range(meta.context_len):
        records.append(
            {
                "dataset_key": dataset_key,
                "seed": int(row.seed),
                "class_id": "overall",
                "relative_position": pos - meta.context_len + 1,
                "attention_weight_mean": float(overall_sums[pos].item() / max(1, overall_count)),
                "support": int(overall_count),
            }
        )
    for (class_id, pos), value in sorted(sums.items()):
        records.append(
            {
                "dataset_key": dataset_key,
                "seed": int(row.seed),
                "class_id": int(class_id),
                "relative_position": pos - meta.context_len + 1,
                "attention_weight_mean": float(value / max(1, counts[(class_id, pos)])),
                "support": int(counts[(class_id, pos)]),
            }
        )
    return records


def plot_attention(summary: pd.DataFrame) -> None:
    overall = summary[summary["class_id"].astype(str).eq("overall")]
    grouped = (
        overall.groupby(["dataset_key", "relative_position"], as_index=False)["attention_weight_mean"].mean()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset_key, group in grouped.groupby("dataset_key"):
        ax.plot(group["relative_position"], group["attention_weight_mean"], marker="o", label=str(dataset_key))
    ax.set_xlabel("Relative window position")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Window attention distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
