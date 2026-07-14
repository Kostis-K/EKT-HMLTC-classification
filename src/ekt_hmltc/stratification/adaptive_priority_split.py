"""Adaptive-priority multilabel stratification.

This splitter is inspired by the old Java ``Classifier.stratifiedMultiSplit``
implementation. It repeatedly chooses the label/subset pair with the strongest
current need and assigns one remaining record carrying that label to that
subset.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from ekt_hmltc.stratification.human_annotations_split import SplitResult, build_label_matrix


@dataclass(frozen=True)
class AdaptiveSplitDiagnostics:
    target_subset_sizes: dict[str, int]
    actual_subset_sizes: dict[str, int]
    assignments_from_priority: int
    assignments_from_fill: int


def _target_sizes(total_records: int, ratios: tuple[float, ...]) -> list[int]:
    raw_targets = [total_records * ratio for ratio in ratios]
    sizes = [int(np.floor(target)) for target in raw_targets]
    remainder = total_records - sum(sizes)
    fractional_order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_targets[index] - sizes[index], ratios[index]),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        sizes[index] += 1
    return sizes


def _label_indices_by_record(matrix: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(matrix[row_index]) for row_index in range(matrix.shape[0])]


def _record_queues_by_label(matrix: np.ndarray, record_order: np.ndarray) -> list[deque[int]]:
    queues: list[deque[int]] = []
    for label_index in range(matrix.shape[1]):
        records = [int(index) for index in record_order if matrix[int(index), label_index] == 1]
        queues.append(deque(records))
    return queues


def _drop_assigned_from_front(queue: deque[int], assigned: np.ndarray) -> None:
    while queue and assigned[queue[0]]:
        queue.popleft()


def _priority(
    subset_ratio: float,
    subset_label_count: int,
    assigned_label_count: int,
    remaining_label_count: int,
) -> float:
    if remaining_label_count <= 0:
        return float("-inf")
    if assigned_label_count <= 0 or subset_label_count <= 0:
        return subset_ratio / remaining_label_count
    return (subset_ratio - (subset_label_count / assigned_label_count)) / remaining_label_count


def _heap_item(
    label_index: int,
    subset_index: int,
    score: float,
    remaining_label_count: int,
    total_label_count: int,
    version: int,
) -> tuple[float, int, int, int, int, int]:
    # heapq is a min-heap. Negating the score gives us max-priority behavior.
    return (-score, remaining_label_count, total_label_count, label_index, subset_index, version)


def _push_label_priorities(
    heap: list[tuple[float, int, int, int, int, int]],
    label_index: int,
    ratios: tuple[float, ...],
    total_label_counts: np.ndarray,
    remaining_label_counts: np.ndarray,
    label_versions: np.ndarray,
    subset_label_counts: np.ndarray,
    subset_sizes: np.ndarray,
    subset_targets: np.ndarray,
) -> None:
    remaining_label_count = int(remaining_label_counts[label_index])
    if remaining_label_count <= 0:
        return

    assigned_label_count = int(subset_label_counts[:, label_index].sum())
    total_label_count = int(total_label_counts[label_index])
    version = int(label_versions[label_index])

    for subset_index, subset_ratio in enumerate(ratios):
        if subset_sizes[subset_index] >= subset_targets[subset_index]:
            continue
        score = _priority(
            subset_ratio=subset_ratio,
            subset_label_count=int(subset_label_counts[subset_index, label_index]),
            assigned_label_count=assigned_label_count,
            remaining_label_count=remaining_label_count,
        )
        heapq.heappush(
            heap,
            _heap_item(
                label_index=label_index,
                subset_index=subset_index,
                score=score,
                remaining_label_count=remaining_label_count,
                total_label_count=total_label_count,
                version=version,
            ),
        )


def _initialize_priority_heap(
    ratios: tuple[float, ...],
    total_label_counts: np.ndarray,
    remaining_label_counts: np.ndarray,
    label_versions: np.ndarray,
    subset_label_counts: np.ndarray,
    subset_sizes: np.ndarray,
    subset_targets: np.ndarray,
) -> list[tuple[float, int, int, int, int, int]]:
    heap: list[tuple[float, int, int, int, int, int]] = []
    for label_index in range(len(total_label_counts)):
        _push_label_priorities(
            heap=heap,
            label_index=label_index,
            ratios=ratios,
            total_label_counts=total_label_counts,
            remaining_label_counts=remaining_label_counts,
            label_versions=label_versions,
            subset_label_counts=subset_label_counts,
            subset_sizes=subset_sizes,
            subset_targets=subset_targets,
        )
    return heap


def _pop_best_label_subset(
    heap: list[tuple[float, int, int, int, int, int]],
    assigned: np.ndarray,
    label_queues: list[deque[int]],
    label_versions: np.ndarray,
    remaining_label_counts: np.ndarray,
    subset_sizes: np.ndarray,
    subset_targets: np.ndarray,
) -> tuple[int, int] | None:
    while heap:
        _, _, _, label_index, subset_index, version = heapq.heappop(heap)
        if version != int(label_versions[label_index]):
            continue
        if subset_sizes[subset_index] >= subset_targets[subset_index]:
            continue
        if remaining_label_counts[label_index] <= 0:
            continue
        _drop_assigned_from_front(label_queues[label_index], assigned)
        if not label_queues[label_index]:
            continue
        return label_index, subset_index
    return None


def _next_unassigned_with_label(queue: deque[int], assigned: np.ndarray) -> int | None:
    _drop_assigned_from_front(queue, assigned)
    if not queue:
        return None
    return queue.popleft()


def _next_subset_with_capacity(subset_sizes: np.ndarray, subset_targets: np.ndarray) -> int | None:
    deficits = subset_targets - subset_sizes
    if np.max(deficits) <= 0:
        return None
    return int(np.argmax(deficits))


def adaptive_priority_split_indices(
    matrix: np.ndarray,
    ratios: tuple[float, ...],
    seed: int | None,
    split_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], AdaptiveSplitDiagnostics]:
    if len(ratios) != len(split_names):
        raise ValueError("The number of ratios must match the number of split names.")

    rng = np.random.default_rng(seed)
    record_order = rng.permutation(matrix.shape[0])
    label_queues = _record_queues_by_label(matrix, record_order)
    record_labels = _label_indices_by_record(matrix)
    total_label_counts = matrix.sum(axis=0).astype(np.int64)
    remaining_label_counts = total_label_counts.copy()
    label_versions = np.zeros(matrix.shape[1], dtype=np.int64)
    subset_targets = np.array(_target_sizes(matrix.shape[0], ratios), dtype=np.int64)
    subset_sizes = np.zeros(len(ratios), dtype=np.int64)
    subset_label_counts = np.zeros((len(ratios), matrix.shape[1]), dtype=np.int64)
    assigned = np.zeros(matrix.shape[0], dtype=bool)
    split_indices: list[list[int]] = [[] for _ in ratios]
    assignments_from_priority = 0
    assignments_from_fill = 0
    heap = _initialize_priority_heap(
        ratios=ratios,
        total_label_counts=total_label_counts,
        remaining_label_counts=remaining_label_counts,
        label_versions=label_versions,
        subset_label_counts=subset_label_counts,
        subset_sizes=subset_sizes,
        subset_targets=subset_targets,
    )

    while int(assigned.sum()) < matrix.shape[0]:
        selected = _pop_best_label_subset(
            heap=heap,
            assigned=assigned,
            label_queues=label_queues,
            label_versions=label_versions,
            remaining_label_counts=remaining_label_counts,
            subset_sizes=subset_sizes,
            subset_targets=subset_targets,
        )

        if selected is None:
            subset_index = _next_subset_with_capacity(subset_sizes, subset_targets)
            if subset_index is None:
                raise RuntimeError("No subset capacity left while records remain unassigned.")
            remaining_records = [int(index) for index in record_order if not assigned[int(index)]]
            if not remaining_records:
                break
            record_index = remaining_records[0]
            assignments_from_fill += 1
        else:
            label_index, subset_index = selected
            record_index = _next_unassigned_with_label(label_queues[label_index], assigned)
            if record_index is None:
                continue
            assignments_from_priority += 1

        assigned[record_index] = True
        split_indices[subset_index].append(record_index)
        subset_sizes[subset_index] += 1
        affected_labels = record_labels[record_index]
        for label_index in affected_labels:
            remaining_label_counts[label_index] -= 1
            subset_label_counts[subset_index, label_index] += 1
            label_versions[label_index] += 1
        for label_index in affected_labels:
            _push_label_priorities(
                heap=heap,
                label_index=int(label_index),
                ratios=ratios,
                total_label_counts=total_label_counts,
                remaining_label_counts=remaining_label_counts,
                label_versions=label_versions,
                subset_label_counts=subset_label_counts,
                subset_sizes=subset_sizes,
                subset_targets=subset_targets,
            )

    diagnostics = AdaptiveSplitDiagnostics(
        target_subset_sizes=dict(zip(split_names, (int(size) for size in subset_targets), strict=True)),
        actual_subset_sizes=dict(zip(split_names, (int(size) for size in subset_sizes), strict=True)),
        assignments_from_priority=assignments_from_priority,
        assignments_from_fill=assignments_from_fill,
    )
    return {
        split_name: np.array(indices, dtype=np.int64)
        for split_name, indices in zip(split_names, split_indices, strict=True)
    }, diagnostics


def make_adaptive_split(
    records: list[dict],
    ratios: tuple[float, ...],
    seed: int | None,
    taxonomy_path,
    label_field: str,
    split_names: tuple[str, ...],
) -> tuple[SplitResult, AdaptiveSplitDiagnostics]:
    labels, matrix, label_projection = build_label_matrix(
        records,
        taxonomy_path=taxonomy_path,
        label_field=label_field,
    )
    split_indices, diagnostics = adaptive_priority_split_indices(
        matrix=matrix,
        ratios=ratios,
        seed=seed,
        split_names=split_names,
    )
    return (
        SplitResult(
            records=records,
            labels=labels,
            matrix=matrix,
            split_indices=split_indices,
            ratios=ratios,
            label_projection=label_projection,
            label_field=label_field,
        ),
        diagnostics,
    )
