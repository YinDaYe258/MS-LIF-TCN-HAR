from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULT_SPECS = (
    ("hapt6", Path("results/hapt6_multiseed_results.csv"), Path("results/hapt6_per_class_metrics.csv")),
    ("hapt12", Path("results/hapt12_k2_multiseed_results.csv"), Path("results/hapt12_k2_per_class_metrics.csv")),
)


def per_class_from_confusion(
    matrix: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=np.float64)
    rows = []
    for class_id, class_name in enumerate(class_names):
        support = int(matrix[class_id].sum()) if class_id < matrix.shape[0] else 0
        predicted = int(matrix[:, class_id].sum()) if class_id < matrix.shape[1] else 0
        tp = float(matrix[class_id, class_id]) if class_id < matrix.shape[0] and class_id < matrix.shape[1] else 0.0
        precision = tp / predicted if predicted > 0 else np.nan
        recall = tp / support if support > 0 else np.nan
        if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
            f1 = np.nan
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "note": "filtered_by_sequence_protocol" if support == 0 else "",
            }
        )
    return pd.DataFrame(rows)


def read_hapt_class_names(task: str, root: Path = Path("data/HAPT Dataset")) -> list[str]:
    labels = {}
    path = root / "activity_labels.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                labels[int(parts[0]) - 1] = parts[1]
    num_classes = 6 if task == "hapt6" else 12
    return [labels.get(class_id, f"class_{class_id}") for class_id in range(num_classes)]


def rows_for_results(results_path: Path, task: str) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing HAPT result file: {results_path}")
    results = pd.read_csv(results_path)
    class_names = read_hapt_class_names(task)
    records = []
    for row in results.itertuples(index=False):
        matrix_path = Path(str(row.confusion_matrix_path))
        if not matrix_path.exists():
            continue
        matrix = np.asarray(json.loads(matrix_path.read_text(encoding="utf-8")), dtype=np.float64)
        class_rows = per_class_from_confusion(matrix, class_names)
        for class_row in class_rows.to_dict("records"):
            records.append(
                {
                    "dataset": getattr(row, "dataset", "HAPT"),
                    "task": getattr(row, "task", task),
                    "model": row.model,
                    "seed": int(row.seed),
                    "context_len": int(row.context_len),
                    "target_mode": row.target_mode,
                    **class_row,
                }
            )
    return pd.DataFrame(records)


def save_latex(per_class: pd.DataFrame, output_path: Path) -> None:
    if per_class.empty:
        output_path.write_text("", encoding="utf-8")
        return
    table = (
        per_class.groupby(["model", "class_name"], dropna=False)
        .agg(support=("support", "mean"), f1=("f1", "mean"))
        .reset_index()
    )
    table["support"] = table["support"].round(1)
    table["f1"] = table["f1"].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
    table.to_latex(output_path, index=False, escape=False)


def main() -> None:
    for task, input_path, output_path in RESULT_SPECS:
        if not input_path.exists():
            print(f"Skipping missing {input_path}")
            continue
        per_class = rows_for_results(input_path, task)
        per_class.to_csv(output_path, index=False)
        tex_path = Path("results") / (
            "table_hapt6_per_class_f1.tex" if task == "hapt6" else "table_hapt12_k2_per_class_f1.tex"
        )
        save_latex(per_class, tex_path)
        print(f"Saved {output_path}")
        print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
