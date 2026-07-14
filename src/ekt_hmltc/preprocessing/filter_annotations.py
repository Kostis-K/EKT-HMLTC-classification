"""Filter annotation records before stratification/training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ekt_hmltc.preprocessing.taxonomy import (
    expand_with_ancestors,
    load_parent_map,
    remove_redundant_ancestors,
)


RAW_LABEL_FIELD = "ekt_subjects"
SELECTED_LABEL_FIELD = "ekt_selected_subjects"
PROJECTED_LABEL_FIELD = "ekt_projected_subjects"


@dataclass(frozen=True)
class FilterResult:
    kept_records: list[dict]
    dropped_records: list[dict]
    raw_label_counts: Counter[str]
    selected_label_counts: Counter[str]
    support_label_counts: Counter[str]
    supported_selected_labels: set[str]
    unsupported_selected_labels: set[str]
    support_counting: str


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels = record.get(RAW_LABEL_FIELD)
            if not isinstance(labels, list):
                raise ValueError(f"Record at line {line_number} has no list {RAW_LABEL_FIELD} field.")
            if not labels:
                raise ValueError(f"Record at line {line_number} has an empty {RAW_LABEL_FIELD} list.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")


def count_labels(records: list[dict], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(record[field]))
    return counts


def prepare_record_labels(record: dict, parent_by_label: dict[str, str] | None) -> dict:
    prepared = dict(record)
    raw_labels = list(dict.fromkeys(record[RAW_LABEL_FIELD]))

    if parent_by_label is None:
        selected_labels = set(raw_labels)
        projected_labels = set(raw_labels)
    else:
        selected_labels = remove_redundant_ancestors(raw_labels, parent_by_label)
        projected_labels = expand_with_ancestors(selected_labels, parent_by_label)

    prepared[SELECTED_LABEL_FIELD] = sorted(selected_labels)
    prepared[PROJECTED_LABEL_FIELD] = sorted(projected_labels)
    return prepared


def prepare_records(records: list[dict], taxonomy_path: Path | None) -> tuple[list[dict], str]:
    parent_by_label = load_parent_map(taxonomy_path) if taxonomy_path is not None else None
    prepared_records = [prepare_record_labels(record, parent_by_label) for record in records]
    support_counting = "selected_plus_ancestors" if parent_by_label is not None else "selected"
    return prepared_records, support_counting


def filter_by_min_label_support(
    records: list[dict],
    min_support: int,
    taxonomy_path: Path | None = None,
) -> FilterResult:
    if min_support < 1:
        raise ValueError("min_support must be at least 1.")

    prepared_records, support_counting = prepare_records(records, taxonomy_path)
    raw_label_counts = count_labels(prepared_records, RAW_LABEL_FIELD)
    selected_label_counts = count_labels(prepared_records, SELECTED_LABEL_FIELD)
    support_label_counts = count_labels(prepared_records, PROJECTED_LABEL_FIELD)

    selected_labels = set(selected_label_counts)
    supported_selected_labels = {
        label for label in selected_labels if support_label_counts[label] >= min_support
    }
    unsupported_selected_labels = selected_labels - supported_selected_labels

    kept_records: list[dict] = []
    dropped_records: list[dict] = []
    for record in prepared_records:
        selected = set(record[SELECTED_LABEL_FIELD])
        if selected <= supported_selected_labels:
            kept_records.append(record)
        else:
            dropped_records.append(record)

    return FilterResult(
        kept_records=kept_records,
        dropped_records=dropped_records,
        raw_label_counts=raw_label_counts,
        selected_label_counts=selected_label_counts,
        support_label_counts=support_label_counts,
        supported_selected_labels=supported_selected_labels,
        unsupported_selected_labels=unsupported_selected_labels,
        support_counting=support_counting,
    )


def write_label_support(result: FilterResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "label",
                "raw_count",
                "selected_count",
                "support_count",
                "appears_raw",
                "appears_selected",
                "supported_selected_label",
            ],
        )
        writer.writeheader()
        labels = set(result.support_label_counts) | set(result.selected_label_counts) | set(result.raw_label_counts)
        for label in sorted(labels):
            writer.writerow(
                {
                    "label": label,
                    "raw_count": result.raw_label_counts.get(label, 0),
                    "selected_count": result.selected_label_counts.get(label, 0),
                    "support_count": result.support_label_counts.get(label, 0),
                    "appears_raw": label in result.raw_label_counts,
                    "appears_selected": label in result.selected_label_counts,
                    "supported_selected_label": label in result.supported_selected_labels,
                }
            )


def write_summary(result: FilterResult, path: Path, min_support: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "min_support": min_support,
        "input_records": len(result.kept_records) + len(result.dropped_records),
        "kept_records": len(result.kept_records),
        "dropped_records": len(result.dropped_records),
        "raw_export_labels": len(result.raw_label_counts),
        "selected_input_labels": len(result.selected_label_counts),
        "projected_support_labels": len(result.support_label_counts),
        "supported_selected_labels": len(result.supported_selected_labels),
        "unsupported_selected_labels": len(result.unsupported_selected_labels),
        "raw_label_field": RAW_LABEL_FIELD,
        "selected_label_field": SELECTED_LABEL_FIELD,
        "projected_label_field": PROJECTED_LABEL_FIELD,
        "support_counting": result.support_counting,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/human_annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/human_annotations_min50.jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/preprocessing/human_min50"))
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/EKTSubjects_new.xml"))
    parser.add_argument("--direct-support-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    taxonomy_path = None if args.direct_support_only else args.taxonomy
    result = filter_by_min_label_support(records, args.min_support, taxonomy_path=taxonomy_path)

    write_jsonl(result.kept_records, args.output)
    write_label_support(result, args.report_dir / "label_support.csv")
    write_summary(result, args.report_dir / "summary.json", args.min_support)

    print((args.report_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
