"""Create multilabel-stratified splits for the human annotation JSONL export."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from skmultilearn.model_selection import IterativeStratification

from ekt_hmltc.preprocessing.filter_annotations import PROJECTED_LABEL_FIELD, RAW_LABEL_FIELD, SELECTED_LABEL_FIELD
from ekt_hmltc.preprocessing.taxonomy import (
    expand_with_ancestors,
    load_parent_map,
    remove_redundant_ancestors,
)


DEFAULT_SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitResult:
    records: list[dict]
    labels: list[str]
    matrix: np.ndarray
    split_indices: dict[str, np.ndarray]
    ratios: tuple[float, ...]
    label_projection: str
    label_field: str


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record.get(RAW_LABEL_FIELD), list):
                raise ValueError(f"Record at line {line_number} has no list {RAW_LABEL_FIELD} field.")
            if not record[RAW_LABEL_FIELD]:
                raise ValueError(f"Record at line {line_number} has an empty {RAW_LABEL_FIELD} list.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def get_record_labels(
    record: dict,
    label_field: str,
    parent_by_label: dict[str, str] | None,
) -> set[str]:
    labels = record.get(label_field)
    if isinstance(labels, list) and labels:
        return set(labels)

    if parent_by_label is None:
        return set(record[RAW_LABEL_FIELD])

    selected_labels = record.get(SELECTED_LABEL_FIELD)
    if not isinstance(selected_labels, list) or not selected_labels:
        selected_labels = sorted(remove_redundant_ancestors(record[RAW_LABEL_FIELD], parent_by_label))
    return expand_with_ancestors(selected_labels, parent_by_label)


def build_label_matrix(
    records: list[dict],
    taxonomy_path: Path | None = None,
    label_field: str = PROJECTED_LABEL_FIELD,
) -> tuple[list[str], np.ndarray, str]:
    parent_by_label = load_parent_map(taxonomy_path) if taxonomy_path is not None else None
    projected_record_labels = [
        get_record_labels(record, label_field=label_field, parent_by_label=parent_by_label) for record in records
    ]
    labels = sorted({label for record_labels in projected_record_labels for label in record_labels})
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=np.int8)

    for row_index, record_labels in enumerate(projected_record_labels):
        for label in record_labels:
            matrix[row_index, label_to_index[label]] = 1

    if label_field == PROJECTED_LABEL_FIELD:
        label_projection = PROJECTED_LABEL_FIELD
    elif taxonomy_path is not None:
        label_projection = f"{label_field}_plus_ancestors"
    else:
        label_projection = label_field
    return labels, matrix, label_projection


def shuffle_records(records: list[dict], seed: int | None) -> list[dict]:
    if seed is None:
        return records
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    return [records[int(index)] for index in order]


def normalize_ratios(ratios: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(float(ratio) for ratio in ratios)
    if len(normalized) != 3:
        raise ValueError("Exactly three ratios are expected: train dev test.")
    if any(ratio <= 0 or ratio >= 1 for ratio in normalized):
        raise ValueError("Ratios must be decimal values between 0 and 1, for example: 0.8 0.1 0.1.")
    total = sum(normalized)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0. Got {total:.12g} from {normalized}.")
    return normalized


def make_split(
    records: list[dict],
    ratios: tuple[float, ...],
    order: int,
    seed: int | None,
    taxonomy_path: Path | None = None,
    label_field: str = PROJECTED_LABEL_FIELD,
    split_names: tuple[str, ...] = DEFAULT_SPLIT_NAMES,
) -> SplitResult:
    labels, matrix, label_projection = build_label_matrix(
        records,
        taxonomy_path=taxonomy_path,
        label_field=label_field,
    )
    features = np.zeros((len(records), 1), dtype=np.int8)
    if seed is not None:
        np.random.seed(seed)

    stratifier = IterativeStratification(
        n_splits=len(ratios),
        order=order,
        sample_distribution_per_fold=list(ratios),
        random_state=None,
    )

    # scikit-multilearn yields each fold as the test indices. Because we pass
    # sample_distribution_per_fold, fold 0/1/2 correspond to train/dev/test.
    fold_indices = [test_index for _, test_index in stratifier.split(features, matrix)]
    if len(fold_indices) != len(split_names):
        raise RuntimeError(f"Expected {len(split_names)} folds, got {len(fold_indices)}.")

    return SplitResult(
        records=records,
        labels=labels,
        matrix=matrix,
        split_indices=dict(zip(split_names, fold_indices, strict=True)),
        ratios=ratios,
        label_projection=label_projection,
        label_field=label_field,
    )


def write_jsonl(records: list[dict], indices: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for index in indices:
            fp.write(json.dumps(records[int(index)], ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")


def write_labels(labels: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for label in labels:
            fp.write(label)
            fp.write("\n")


def write_label_distribution(result: SplitResult, path: Path) -> dict:
    totals = result.matrix.sum(axis=0).astype(int)
    summary_abs_deviations: list[float] = []
    missing_by_split: dict[str, int] = {}

    fieldnames = ["label", "total"]
    for split_name in result.split_indices:
        fieldnames.extend(
            [
                f"{split_name}_target",
                f"{split_name}_actual",
                f"{split_name}_deviation",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for label_index, label in enumerate(result.labels):
            row: dict[str, int | float | str] = {
                "label": label,
                "total": int(totals[label_index]),
            }
            for split_index, (split_name, indices) in enumerate(result.split_indices.items()):
                target = totals[label_index] * result.ratios[split_index]
                actual = int(result.matrix[indices, label_index].sum())
                deviation = actual - target
                row[f"{split_name}_target"] = round(float(target), 3)
                row[f"{split_name}_actual"] = actual
                row[f"{split_name}_deviation"] = round(float(deviation), 3)
                summary_abs_deviations.append(abs(float(deviation)))
            writer.writerow(row)

    for split_name, indices in result.split_indices.items():
        split_counts = result.matrix[indices].sum(axis=0)
        missing_by_split[split_name] = int(((totals > 0) & (split_counts == 0)).sum())

    return {
        "mean_abs_label_count_deviation": round(float(np.mean(summary_abs_deviations)), 6),
        "max_abs_label_count_deviation": round(float(np.max(summary_abs_deviations)), 6),
        "missing_labels_by_split": missing_by_split,
    }


def write_summary(result: SplitResult, diagnostics: dict, output_path: Path, order: int, seed: int | None) -> None:
    split_sizes = {name: int(len(indices)) for name, indices in result.split_indices.items()}
    total_labels_per_split = {
        name: int(result.matrix[indices].sum()) for name, indices in result.split_indices.items()
    }
    summary = {
        "records": len(result.records),
        "labels": len(result.labels),
        "order": order,
        "seed": seed,
        "label_field": result.label_field,
        "label_projection": result.label_projection,
        "ratios": dict(zip(result.split_indices.keys(), result.ratios, strict=True)),
        "split_sizes": split_sizes,
        "total_labels_per_split": total_labels_per_split,
        **diagnostics,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split(result: SplitResult, output_dir: Path, order: int, seed: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, indices in result.split_indices.items():
        write_jsonl(result.records, indices, output_dir / f"{split_name}.jsonl")

    write_labels(result.labels, output_dir / "labels.txt")
    diagnostics = write_label_distribution(result, output_dir / "label_distribution.csv")
    write_summary(result, diagnostics, output_dir / "summary.json", order=order, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/human_annotations.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/subsets/human_order1_80_10_10_seed16"),
    )
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "DEV", "TEST"))
    parser.add_argument("--order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/EKTSubjects_new.xml"))
    parser.add_argument("--label-field", default=PROJECTED_LABEL_FIELD)
    parser.add_argument("--direct-labels-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = normalize_ratios(args.ratios)
    records = shuffle_records(load_jsonl(args.input), args.seed)
    taxonomy_path = None if args.direct_labels_only else args.taxonomy
    result = make_split(
        records,
        ratios=ratios,
        order=args.order,
        seed=args.seed,
        taxonomy_path=taxonomy_path,
        label_field=args.label_field,
    )
    write_split(result, args.output_dir, order=args.order, seed=args.seed)

    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
