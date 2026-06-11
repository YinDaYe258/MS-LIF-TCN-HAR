from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SELECTED_MODELS = ["lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif", "cnn1d", "ms_cnn1d", "window_gru"]


def _level_mask(values: pd.Series, target: str) -> pd.Series:
    target = str(target)
    if target in {"acc", "gyro"}:
        return values.astype(str) == target
    numeric_values = pd.to_numeric(values, errors="coerce")
    return (numeric_values - float(target)).abs() < 1e-9


def _plot_lines(rows: pd.DataFrame, perturbation_type: str, levels: list[str], out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for model in SELECTED_MODELS:
        values = []
        for level in levels:
            matched = rows[
                (rows["model"] == model)
                & (rows["perturbation_type"] == perturbation_type)
                & _level_mask(rows["perturbation_level"], level)
            ]
            values.append(float(matched.iloc[-1]["macro_f1"]) if not matched.empty else float("nan"))
        plt.plot(levels, values, marker="o", label=model)
    plt.xlabel("Perturbation strength")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_modality(rows: pd.DataFrame, out_path: Path) -> None:
    levels = ["clean", "acc", "gyro"]
    plt.figure(figsize=(8, 5))
    for model in SELECTED_MODELS:
        values = []
        for level in levels:
            if level == "clean":
                matched = rows[(rows["model"] == model) & (rows["perturbation_type"] == "clean")]
            else:
                matched = rows[
                    (rows["model"] == model)
                    & (rows["perturbation_type"] == "modality_dropout")
                    & (rows["perturbation_level"].astype(str) == level)
                ]
            values.append(float(matched.iloc[-1]["macro_f1"]) if not matched.empty else float("nan"))
        plt.plot(levels, values, marker="o", label=model)
    plt.xlabel("Condition")
    plt.ylabel("Macro-F1")
    plt.title("Modality Dropout Robustness")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    results_dir = Path("results")
    input_path = results_dir / "ucihar_robustness_suite.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing robustness suite CSV: {input_path}")
    rows = pd.read_csv(input_path)
    _plot_lines(
        rows,
        "gaussian_noise",
        ["0.05", "0.1", "0.2"],
        results_dir / "fig_noise_robustness_macro_f1.png",
        "Gaussian Noise Robustness",
    )
    _plot_lines(
        rows,
        "channel_dropout",
        ["0.1", "0.2", "0.3"],
        results_dir / "fig_channel_dropout_macro_f1.png",
        "Channel Dropout Robustness",
    )
    _plot_modality(rows, results_dir / "fig_modality_dropout_macro_f1.png")
    print("Saved robustness figures")


if __name__ == "__main__":
    main()
