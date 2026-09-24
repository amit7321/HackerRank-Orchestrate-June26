"""CSV I/O helpers: reading claims/history/evidence, writing output.csv."""

import csv
from pathlib import Path
from typing import Dict, List

from . import config

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def read_claims_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_evidence_requirements(path: Path = config.EVIDENCE_REQUIREMENTS_CSV) -> List[Dict[str, str]]:
    return read_claims_csv(path)


def read_user_history(path: Path = config.USER_HISTORY_CSV) -> Dict[str, Dict[str, str]]:
    rows = read_claims_csv(path)
    return {row["user_id"]: row for row in rows}


def split_semicolon_field(value: str) -> List[str]:
    value = (value or "").strip()
    if not value or value.lower() == "none":
        return []
    return [v.strip() for v in value.split(";") if v.strip()]


def join_semicolon_field(values: List[str]) -> str:
    values = [v for v in values if v]
    return ";".join(values) if values else "none"


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def write_output_csv(rows: List[Dict[str, str]], path: Path = config.OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})
