from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    results_dir = Path("results")
    formal = _load_optional(results_dir / "ucihar_formal_multiseed_summary.csv")
    hapt6 = _load_optional(results_dir / "hapt6_multiseed_summary.csv")
    distill = _load_optional(results_dir / "distill_multiseed_summary.csv")
    transition = _load_optional(results_dir / "hapt_transition_binary_summary.csv")
    balanced = _load_optional(results_dir / "class_balanced_summary.csv")

    lines: list[str] = [
        "# Final Readiness After Enhancement",
        "",
        "This report is generated from completed experiment CSV files. It is an audit note, not a paper conclusion.",
        "",
        "## Supported Findings",
        "",
    ]

    if not distill.empty:
        lines.append("### Distillation")
        for row in distill.itertuples():
            lines.append(
                f"- {row.dataset_key} {row.model}: Macro-F1 {_fmt(row.macro_f1_mean)} "
                f"+/- {_fmt(row.macro_f1_std)} over {int(row.num_seeds)} seeds "
                f"(T={row.temperature}, kd={row.kd_weight})."
            )
        for row in distill.itertuples():
            baseline = formal if row.dataset_key == "ucihar" else hapt6 if row.dataset_key == "hapt6" else pd.DataFrame()
            if baseline.empty:
                continue
            base = baseline[baseline["model"].astype(str).eq(str(row.base_model))]
            if base.empty:
                continue
            diff = float(row.macro_f1_mean - base.iloc[0]["macro_f1_mean"])
            dataset_name = "UCI-HAR" if row.dataset_key == "ucihar" else "HAPT-6"
            lines.append(
                f"- {dataset_name} distillation changes {row.base_model} mean Macro-F1 by {_fmt(diff)}."
            )
        lines.extend(["", ""])

    if not transition.empty:
        lines.append("### HAPT Transition Binary")
        k2 = transition[transition["context_len"].astype(int).eq(2)]
        for row in k2.itertuples():
            lines.append(
                f"- K=2 {row.model}: Macro-F1 {_fmt(row.macro_f1_mean)} +/- {_fmt(row.macro_f1_std)}, "
                f"transition recall {_fmt(row.transition_recall_mean)}."
            )
        lines.append(
            "- This task is a diagnostic for basic-vs-transition detection, not a replacement for HAPT-6/HAPT-12 main classification."
        )
        lines.extend(["", ""])

    if not balanced.empty:
        lines.append("### Class-Balanced Loss")
        for row in balanced.itertuples():
            lines.append(
                f"- {row.dataset_key} {row.model} {row.loss_type}: Macro-F1 {_fmt(row.macro_f1)}, "
                f"balanced accuracy {_fmt(row.balanced_accuracy)}."
            )
        lines.append("- Current weighted-CE diagnostics do not support claiming that class-balanced loss reliably improves CMG-LIF-Lite.")
        lines.extend(["", ""])

    lines.extend(
        [
            "## Unsupported Or Risky Claims",
            "",
            "- Do not claim CMG-LIF-Lite is better than parameter-matched LIF-SNN.",
            "- Do not claim HAPT-12 K2 is a complete 12-class benchmark; class coverage is incomplete.",
            "- Do not claim measured power or measured energy. Report spike rate and operation proxies only.",
            "- Do not claim distillation beats Window-GRU unless a formal same-protocol comparison supports it.",
            "- Do not claim transition-binary results are stable for lightweight CMG-LIF-Lite; the K=2 result is seed-sensitive.",
            "",
            "## Recommended Paper Direction",
            "",
            "- Main model line: multi-scale SNN for HAR, optionally with Window-GRU distillation.",
            "- Secondary analysis: lightweight context gates and transition-detection diagnostics.",
            "- Conservative title direction: Distilled Multi-Scale Spiking Neural Networks for Wearable Human Activity Recognition.",
            "",
            "## Remaining Work Before Writing",
            "",
            "- Run a small validation-selected KD grid before declaring T=2.0, kd=0.3 final.",
            "- MS-CMG-LIF has now been repeated as a distilled student; use it as a cautious comparison, not an automatic main claim.",
            "- Keep HAPT transition-binary as diagnostic unless full 3-seed stability is strong under the final selected protocol.",
            "- Add final per-class tables for any model claimed as best.",
        ]
    )

    output = results_dir / "final_readiness_after_enhancement.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
