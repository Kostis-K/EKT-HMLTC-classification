"""Run the default EKT HMLTC dataset preparation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ekt_hmltc.preprocessing.filter_annotations import (
    filter_by_min_label_support,
    load_jsonl,
    write_jsonl,
    write_label_support,
    write_summary as write_filter_summary,
)
from ekt_hmltc.stratification.human_annotations_split import (
    make_split,
    normalize_ratios,
    shuffle_records,
    write_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/human_annotations.jsonl"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/EKTSubjects_new.xml"))
    parser.add_argument("--direct-support-only", action="store_true")
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "DEV", "TEST"))
    parser.add_argument("--order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--prepared-output", type=Path, default=Path("data/processed/human_annotations_min50.jsonl"))
    parser.add_argument("--preprocess-report-dir", type=Path, default=Path("outputs/preprocessing/human_min50"))
    parser.add_argument("--split-output-dir", type=Path, default=Path("outputs/subsets/human_min50_order1_80_10_10_seed16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = normalize_ratios(args.ratios)

    records = load_jsonl(args.input)
    taxonomy_path = None if args.direct_support_only else args.taxonomy
    filter_result = filter_by_min_label_support(records, args.min_support, taxonomy_path=taxonomy_path)

    write_jsonl(filter_result.kept_records, args.prepared_output)
    write_label_support(filter_result, args.preprocess_report_dir / "label_support.csv")
    write_filter_summary(filter_result, args.preprocess_report_dir / "summary.json", args.min_support)

    shuffled_records = shuffle_records(filter_result.kept_records, args.seed)
    split_result = make_split(
        shuffled_records,
        ratios=ratios,
        order=args.order,
        seed=args.seed,
        taxonomy_path=taxonomy_path,
    )
    write_split(split_result, args.split_output_dir, order=args.order, seed=args.seed)

    pipeline_summary = {
        "preprocessing": json.loads((args.preprocess_report_dir / "summary.json").read_text(encoding="utf-8")),
        "split": json.loads((args.split_output_dir / "summary.json").read_text(encoding="utf-8")),
    }
    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
