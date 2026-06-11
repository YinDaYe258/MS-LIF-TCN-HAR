from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")


def save_context_length_plot(results_dir: Path) -> None:
    path = results_dir / "ucihar_ablation_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["ablation"].astype(str).str.startswith("context_len_")]
    if df.empty:
        return
    plt.figure(figsize=(5, 3))
    sns.lineplot(data=df, x="context_len", y="macro_f1", marker="o", errorbar="sd")
    plt.xlabel("Context length K")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    out = results_dir / "fig_context_len_vs_f1.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def save_noise_plot(results_dir: Path) -> None:
    path = results_dir / "ucihar_robustness_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    noise = df[df["perturbation"].astype(str).str.startswith("gaussian_noise_")].copy()
    if noise.empty:
        return
    noise["noise_std"] = noise["perturbation"].str.replace("gaussian_noise_", "", regex=False).astype(float)
    plt.figure(figsize=(5, 3))
    sns.lineplot(data=noise, x="noise_std", y="macro_f1", hue="model", marker="o", errorbar="sd")
    plt.xlabel("Gaussian noise std")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    out = results_dir / "fig_noise_robustness.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def save_spike_accuracy_plot(results_dir: Path) -> None:
    path = results_dir / "ucihar_main_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "spike_rate" not in df or "accuracy" not in df:
        return
    plt.figure(figsize=(5, 3))
    sns.scatterplot(data=df, x="spike_rate", y="accuracy", hue="model", style="model", s=70)
    plt.xlabel("Spike rate")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    out = results_dir / "fig_spike_rate_vs_accuracy.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def save_confusion_matrix(results_dir: Path) -> None:
    main_path = results_dir / "ucihar_main_results.csv"
    if not main_path.exists():
        return
    df = pd.read_csv(main_path)
    cmg = df[df["model"].astype(str).isin(["cmg_lif", "cmg_lif_snn"])]
    if cmg.empty or "confusion_matrix_path" not in cmg:
        return
    best = cmg.sort_values("macro_f1", ascending=False).iloc[0]
    cm_path = Path(str(best["confusion_matrix_path"]))
    if not cm_path.exists():
        return
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = results_dir / "confusion_matrix_cmg_lif.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from experiment CSV files.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    save_context_length_plot(results_dir)
    save_noise_plot(results_dir)
    save_spike_accuracy_plot(results_dir)
    save_confusion_matrix(results_dir)


if __name__ == "__main__":
    main()
