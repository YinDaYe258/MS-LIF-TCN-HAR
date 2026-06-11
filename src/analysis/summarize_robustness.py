from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "model",
    "clean_macro_f1",
    "noise_0.05_macro_f1",
    "noise_0.10_macro_f1",
    "noise_0.20_macro_f1",
    "dropout_0.10_macro_f1",
    "dropout_0.20_macro_f1",
    "dropout_0.30_macro_f1",
    "acc_dropout_macro_f1",
    "gyro_dropout_macro_f1",
    "noise_0.20_drop",
    "dropout_0.30_drop",
    "acc_dropout_drop",
    "gyro_dropout_drop",
    "average_robustness_drop",
]


def _level_mask(values: pd.Series, target: str) -> pd.Series:
    target = str(target)
    if target in {"acc", "gyro"}:
        return values.astype(str) == target
    numeric_values = pd.to_numeric(values, errors="coerce")
    return (numeric_values - float(target)).abs() < 1e-9


def _metric(rows: pd.DataFrame, model: str, perturbation_type: str, perturbation_level: str) -> float:
    mask = (rows["model"] == model) & (rows["perturbation_type"] == perturbation_type)
    if perturbation_type != "clean":
        mask = mask & _level_mask(rows["perturbation_level"], perturbation_level)
    matched = rows[mask]
    if matched.empty:
        return float("nan")
    return float(matched.iloc[-1]["macro_f1"])


def summarize_robustness_table(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for model in rows["model"].drop_duplicates():
        clean = _metric(rows, model, "clean", "0")
        noise_005 = _metric(rows, model, "gaussian_noise", "0.05")
        noise_010 = _metric(rows, model, "gaussian_noise", "0.1")
        if pd.isna(noise_010):
            noise_010 = _metric(rows, model, "gaussian_noise", "0.10")
        noise_020 = _metric(rows, model, "gaussian_noise", "0.2")
        if pd.isna(noise_020):
            noise_020 = _metric(rows, model, "gaussian_noise", "0.20")
        dropout_010 = _metric(rows, model, "channel_dropout", "0.1")
        if pd.isna(dropout_010):
            dropout_010 = _metric(rows, model, "channel_dropout", "0.10")
        dropout_020 = _metric(rows, model, "channel_dropout", "0.2")
        if pd.isna(dropout_020):
            dropout_020 = _metric(rows, model, "channel_dropout", "0.20")
        dropout_030 = _metric(rows, model, "channel_dropout", "0.3")
        if pd.isna(dropout_030):
            dropout_030 = _metric(rows, model, "channel_dropout", "0.30")
        acc_dropout = _metric(rows, model, "modality_dropout", "acc")
        gyro_dropout = _metric(rows, model, "modality_dropout", "gyro")

        drops = {
            "noise_0.20_drop": clean - noise_020,
            "dropout_0.30_drop": clean - dropout_030,
            "acc_dropout_drop": clean - acc_dropout,
            "gyro_dropout_drop": clean - gyro_dropout,
        }
        avg_drop = pd.Series(drops).mean(skipna=True)
        summaries.append(
            {
                "model": model,
                "clean_macro_f1": clean,
                "noise_0.05_macro_f1": noise_005,
                "noise_0.10_macro_f1": noise_010,
                "noise_0.20_macro_f1": noise_020,
                "dropout_0.10_macro_f1": dropout_010,
                "dropout_0.20_macro_f1": dropout_020,
                "dropout_0.30_macro_f1": dropout_030,
                "acc_dropout_macro_f1": acc_dropout,
                "gyro_dropout_macro_f1": gyro_dropout,
                **drops,
                "average_robustness_drop": avg_drop,
            }
        )
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def main() -> None:
    results_dir = Path("results")
    input_path = results_dir / "ucihar_robustness_suite.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing robustness suite CSV: {input_path}")
    rows = pd.read_csv(input_path)
    summary = summarize_robustness_table(rows)
    csv_path = results_dir / "ucihar_robustness_summary.csv"
    tex_path = results_dir / "table_ucihar_robustness_summary.tex"
    summary.to_csv(csv_path, index=False)
    summary.to_latex(tex_path, index=False, float_format="%.4f")
    findings_path = results_dir / "ucihar_key_findings.md"
    findings_path.write_text(build_key_findings(summary), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(f"Saved {findings_path}")
    print(summary)


def _best_model(summary: pd.DataFrame, column: str, candidates: list[str] | None = None) -> str:
    rows = summary if candidates is None else summary[summary["model"].isin(candidates)]
    if rows.empty:
        return "n/a"
    row = rows.sort_values(column, ascending=False).iloc[0]
    return f"{row['model']} ({row[column]:.4f})"


def build_key_findings(summary: pd.DataFrame) -> str:
    lightweight = ["lif_snn", "cmg_lif_lite"]
    lines = [
        "# UCI-HAR Key Findings",
        "",
        "These findings are based on the current UCI-HAR seed-42 experiments unless a multi-seed table is stated explicitly.",
        "",
        f"- Best clean model: {_best_model(summary, 'clean_macro_f1')}.",
        f"- Best Gaussian noise std=0.20 model: {_best_model(summary, 'noise_0.20_macro_f1')}.",
        f"- Best channel dropout p=0.30 model: {_best_model(summary, 'dropout_0.30_macro_f1')}.",
        f"- Best gyroscope-dropout model: {_best_model(summary, 'gyro_dropout_macro_f1')}.",
        f"- Best lightweight SNN on clean data: {_best_model(summary, 'clean_macro_f1', lightweight)}.",
        f"- Best lightweight SNN under noise std=0.20: {_best_model(summary, 'noise_0.20_macro_f1', lightweight)}.",
        f"- Best lightweight SNN under channel dropout p=0.30: {_best_model(summary, 'dropout_0.30_macro_f1', lightweight)}.",
        "",
        "Conservative interpretation:",
        "",
        "1. CMG-LIF-Lite improves over LIF-SNN in clean and most perturbation absolute Macro-F1 values.",
        "2. MS-LIF-SNN is stronger than MS-CMG-LIF on clean UCI-HAR in the current results.",
        "3. MS-CMG-LIF does not show a robustness advantage over MS-LIF-SNN in the current robustness suite.",
        "4. Window-GRU is the strongest non-SNN baseline in the current seed-42 table, but it has far more parameters.",
        "5. Efficiency results are proxy-only and must not be described as measured power.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
