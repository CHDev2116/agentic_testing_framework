"""Versioned oracle outcomes: collect, diff, and emit semantic change summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from eval.oracle_regression import load_oracle_cases, run_oracle_case

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseOutcome:
    release: str
    conflict: str
    semantic_errors: List[str]
    override_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release": self.release,
            "conflict": self.conflict,
            "semantic_errors": list(self.semantic_errors),
            "override_applied": self.override_applied,
        }

    @classmethod
    def from_run(cls, actual: Dict[str, Any]) -> "CaseOutcome":
        return cls(
            release=str(actual["release"]),
            conflict=str(actual["conflict"]),
            semantic_errors=list(actual.get("semantic_errors") or []),
            override_applied=bool(actual.get("override_applied")),
        )


@dataclass
class CaseSemanticChange:
    case_id: str
    description: str
    fields: List[str]
    before: Dict[str, Any]
    after: Dict[str, Any]
    taxonomy_hint: str = "LD"

    def summary_line(self) -> str:
        parts = [f"{self.case_id}: {', '.join(self.fields)} changed"]
        if "release" in self.fields:
            parts.append(f"release {self.before.get('release')!r} → {self.after.get('release')!r}")
        if "conflict" in self.fields:
            parts.append(
                f"conflict {self.before.get('conflict')!r} → {self.after.get('conflict')!r}"
            )
        return " | ".join(parts)


@dataclass
class SemanticDiffReport:
    baseline_label: str
    current_label: str
    unchanged_count: int
    changed: List[CaseSemanticChange] = field(default_factory=list)
    added_cases: List[str] = field(default_factory=list)
    removed_cases: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.added_cases or self.removed_cases)

    def to_markdown(self) -> str:
        lines = [
            "# Oracle semantic change report",
            "",
            f"- Baseline: `{self.baseline_label}`",
            f"- Current: `{self.current_label}`",
            f"- Unchanged cases: {self.unchanged_count}",
            f"- Changed: {len(self.changed)}",
            f"- Added case ids: {len(self.added_cases)}",
            f"- Removed case ids: {len(self.removed_cases)}",
            "",
        ]
        if not self.has_changes:
            lines.append("No semantic drift detected.")
            return "\n".join(lines) + "\n"

        if self.changed:
            lines.append("## Release / conflict / semantic drift")
            lines.append("")
            for item in self.changed:
                lines.append(f"- **{item.summary_line()}**")
                lines.append(f"  - Description: {item.description}")
                lines.append(f"  - Taxonomy hint: `{item.taxonomy_hint}` (see docs/FailureTaxonomy.md)")
            lines.append("")

        if self.added_cases:
            lines.append("## New cases (not in baseline)")
            for case_id in self.added_cases:
                lines.append(f"- `{case_id}`")
            lines.append("")

        if self.removed_cases:
            lines.append("## Removed cases (in baseline only)")
            for case_id in self.removed_cases:
                lines.append(f"- `{case_id}`")
            lines.append("")

        lines.append("## Interpretation")
        lines.append("")
        lines.append(
            "- If you **intentionally** changed `semantic_asserts` / `arbitrator`, refresh the "
            "snapshot after review: `python scripts/refresh_oracle_snapshot.py`."
        )
        lines.append(
            "- If you **only** edited `oracle_cases.jsonl` expectations, update the JSONL or "
            "revert — the snapshot compares **code output**, not file expectations."
        )
        return "\n".join(lines) + "\n"


def collect_suite_outcomes(cases_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Run current code on all cases; return per-id outcome dicts (no expected_* compare)."""
    outcomes: Dict[str, Dict[str, Any]] = {}
    for case in load_oracle_cases(cases_path):
        case_id = str(case["id"])
        actual = run_oracle_case(case)
        outcome = CaseOutcome.from_run(actual)
        outcomes[case_id] = {
            **outcome.to_dict(),
            "description": str(case.get("description", "")),
        }
    return outcomes


def build_snapshot_document(
    outcomes: Dict[str, Dict[str, Any]],
    *,
    label: str,
    cases_path: str,
    oracle_rules_tag: str = "HEAD",
) -> Dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "label": label,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": cases_path,
        "oracle_rules_tag": oracle_rules_tag,
        "case_count": len(outcomes),
        "cases": outcomes,
    }


def load_snapshot(path: str | Path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(doc.get("schema_version", 0)) != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema_version in {path}: {doc.get('schema_version')}"
        )
    return doc


def _outcome_fields_changed(
    before: Dict[str, Any], after: Dict[str, Any]
) -> List[str]:
    fields: List[str] = []
    for key in ("release", "conflict", "override_applied"):
        if before.get(key) != after.get(key):
            fields.append(key)
    if list(before.get("semantic_errors") or []) != list(after.get("semantic_errors") or []):
        fields.append("semantic_errors")
    return fields


def _taxonomy_hint_for_change(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """Heuristic label for triage (not authoritative)."""
    before_err = list(before.get("semantic_errors") or [])
    after_err = list(after.get("semantic_errors") or [])
    if before_err != after_err and before.get("release") == after.get("release"):
        return "LD"
    if before.get("release") != after.get("release"):
        return "LD"
    return "LD"


def diff_snapshots(
    baseline_doc: Dict[str, Any],
    current_doc: Dict[str, Any],
) -> SemanticDiffReport:
    baseline_cases = baseline_doc.get("cases") or {}
    current_cases = current_doc.get("cases") or {}
    baseline_label = str(baseline_doc.get("label", "baseline"))
    current_label = str(current_doc.get("label", "current"))

    all_ids = sorted(set(baseline_cases) | set(current_cases))
    changed: List[CaseSemanticChange] = []
    unchanged = 0
    added: List[str] = []
    removed: List[str] = []

    for case_id in all_ids:
        if case_id not in baseline_cases:
            added.append(case_id)
            continue
        if case_id not in current_cases:
            removed.append(case_id)
            continue
        before = baseline_cases[case_id]
        after = current_cases[case_id]
        fields = _outcome_fields_changed(before, after)
        if not fields:
            unchanged += 1
            continue
        changed.append(
            CaseSemanticChange(
                case_id=case_id,
                description=str(after.get("description") or before.get("description") or ""),
                fields=fields,
                before={
                    "release": before.get("release"),
                    "conflict": before.get("conflict"),
                    "semantic_errors": before.get("semantic_errors"),
                    "override_applied": before.get("override_applied"),
                },
                after={
                    "release": after.get("release"),
                    "conflict": after.get("conflict"),
                    "semantic_errors": after.get("semantic_errors"),
                    "override_applied": after.get("override_applied"),
                },
                taxonomy_hint=_taxonomy_hint_for_change(before, after),
            )
        )

    return SemanticDiffReport(
        baseline_label=baseline_label,
        current_label=current_label,
        unchanged_count=unchanged,
        changed=changed,
        added_cases=added,
        removed_cases=removed,
    )


def diff_against_baseline_file(
    baseline_path: str | Path,
    *,
    cases_path: str | Path,
    current_label: str = "working_tree",
) -> SemanticDiffReport:
    baseline_doc = load_snapshot(baseline_path)
    current_outcomes = collect_suite_outcomes(cases_path)
    current_doc = build_snapshot_document(
        current_outcomes,
        label=current_label,
        cases_path=str(cases_path),
    )
    return diff_snapshots(baseline_doc, current_doc)
