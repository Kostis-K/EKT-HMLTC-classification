"""Filter annotation records before stratification/training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ekt_hmltc.preprocessing.taxonomy import expand_with_ancestors, load_parent_map


@dataclass(frozen=True)
class FilterResult:
    kept_records: list[dict]
    dropped_records: list[dict]
    direct_label_counts: Counter[str]
    support_label_counts: Counter[str]
    supported_direct_labels: set[str]
    unsupported_direct_labels: set[str]
    support_counting: str


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels = record.get("ekt_subjects")
            if not isinstance(labels, list):
                raise ValueError(f"Record at line {line_number} has no list ekt_subjects field.")
            if not labels:
                raise ValueError(f"Record at line {line_number} has an empty ekt_subjects list.")
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


def count_direct_labels(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(record["ekt_subjects"]))
    return counts


def count_labels_with_ancestors(records: list[dict], taxonomy_path: Path) -> Counter[str]:
    parent_by_label = load_parent_map(taxonomy_path)
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(expand_with_ancestors(record["ekt_subjects"], parent_by_label))
    return counts


def filter_by_min_label_support(
    records: list[dict],
    min_support: int,
    taxonomy_path: Path | None = None,
) -> FilterResult:
    if min_support < 1:
        raise ValueError("min_support must be at least 1.")

    direct_label_counts = count_direct_labels(records)
    if taxonomy_path is None:
        support_label_counts = direct_label_counts
        support_counting = "direct"
    else:
        support_label_counts = count_labels_with_ancestors(records, taxonomy_path)
        support_counting = "direct_plus_ancestors"

    direct_labels = set(direct_label_counts)
    supported_direct_labels = {
        label for label in direct_labels if support_label_counts[label] >= min_support
    }
    unsupported_direct_labels = direct_labels - supported_direct_labels

    kept_records: list[dict] = []
    dropped_records: list[dict] = []
    for record in records:
        labels = set(record["ekt_subjects"])
        if labels <= supported_direct_labels:
            kept_records.append(record)
        else:
            dropped_records.append(record)

    return FilterResult(
        kept_records=kept_records,
        dropped_records=dropped_records,
        direct_label_counts=direct_label_counts,
        support_label_counts=support_label_counts,
        supported_direct_labels=supported_direct_labels,
        unsupported_direct_labels=unsupported_direct_labels,
        support_counting=support_counting,
    )


def write_label_support(result: FilterResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "label",
                "direct_count",
                "support_count",
                "appears_directly",
                "supported_direct_label",
            ],
        )
        writer.writeheader()
        labels = set(result.support_label_counts) | set(result.direct_label_counts)
        for label in sorted(labels):
            writer.writerow(
                {
                    "label": label,
                    "direct_count": result.direct_label_counts.get(label, 0),
                    "support_count": result.support_label_counts.get(label, 0),
                    "appears_directly": label in result.direct_label_counts,
                    "supported_direct_label": label in result.supported_direct_labels,
                }
            )


def write_summary(result: FilterResult, path: Path, min_support: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "min_support": min_support,
        "input_records": len(result.kept_records) + len(result.dropped_records),
        "kept_records": len(result.kept_records),
        "dropped_records": len(result.dropped_records),
        "direct_input_labels": len(result.direct_label_counts),
        "supported_direct_labels": len(result.supported_direct_labels),
        "unsupported_direct_labels": len(result.unsupported_direct_labels),
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
