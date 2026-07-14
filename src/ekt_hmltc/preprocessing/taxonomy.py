"""Helpers for reading and using the EKT/ThetaEPE SKOS taxonomy."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
SKOS_BROADER = "{http://www.w3.org/2004/02/skos/core#}broader"
SKOS_CONCEPT = "{http://www.w3.org/2004/02/skos/core#}Concept"


def load_parent_map(taxonomy_path: Path) -> dict[str, str]:
    tree = ElementTree.parse(taxonomy_path)
    parent_by_label: dict[str, str] = {}

    for concept in tree.getroot().iter(SKOS_CONCEPT):
        uri = concept.attrib.get(RDF_ABOUT)
        if not uri:
            continue

        broader = concept.find(SKOS_BROADER)
        if broader is None:
            continue

        parent = broader.attrib.get(RDF_RESOURCE)
        if parent:
            parent_by_label[uri] = parent

    return parent_by_label


def ancestors_of(label: str, parent_by_label: dict[str, str]) -> list[str]:
    ancestors: list[str] = []
    current = label
    seen = {label}

    while current in parent_by_label:
        parent = parent_by_label[current]
        if parent in seen:
            raise ValueError(f"Cycle detected in taxonomy around {label}.")
        ancestors.append(parent)
        seen.add(parent)
        current = parent

    return ancestors


def expand_with_ancestors(labels: list[str] | set[str], parent_by_label: dict[str, str]) -> set[str]:
    expanded = set(labels)
    for label in labels:
        expanded.update(ancestors_of(label, parent_by_label))
    return expanded


def remove_redundant_ancestors(labels: list[str] | set[str], parent_by_label: dict[str, str]) -> set[str]:
    """Keep the most specific labels selected for one record.

    Human annotations are not forced to be leaves: an internal node can be a
    deliberate annotation. We remove a label only when a descendant of that same
    label is also present in the same record.
    """

    unique_labels = list(dict.fromkeys(labels))
    label_set = set(unique_labels)
    redundant_ancestors: set[str] = set()

    for label in unique_labels:
        for ancestor in ancestors_of(label, parent_by_label):
            if ancestor in label_set:
                redundant_ancestors.add(ancestor)

    return {label for label in unique_labels if label not in redundant_ancestors}
