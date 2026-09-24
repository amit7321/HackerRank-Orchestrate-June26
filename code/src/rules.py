"""Deterministic (no-model) logic: matching an evidence-requirement row,
computing evidence_standard_met / valid_image from per-image analysis, and
looking up user-history risk. Keeping this rule-based makes the pipeline
reproducible and keeps model calls focused on genuinely visual judgments.
"""

from typing import Dict, List, Optional

from . import io_utils

# Keyword hints used to pick the most specific evidence_requirements row for
# a claim, based on the object and the issue types the claim parser found.
_ISSUE_FAMILY_KEYWORDS = {
    "car": [
        (("dent", "scratch"), "REQ_CAR_BODY_PANEL"),
        (("crack", "glass_shatter", "broken_part", "missing_part"), "REQ_CAR_GLASS_LIGHT_MIRROR"),
    ],
    "laptop": [
        (("crack", "stain", "water_damage"), "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"),
        (("dent", "scratch", "broken_part", "missing_part"), "REQ_LAPTOP_BODY_HINGE_PORT"),
    ],
    "package": [
        (("torn_packaging", "crushed_packaging"), "REQ_PACKAGE_EXTERIOR"),
        (("water_damage", "stain"), "REQ_PACKAGE_LABEL_OR_STAIN"),
        (("missing_part",), "REQ_PACKAGE_CONTENTS"),
    ],
}

_FALLBACK_REQUIREMENT_ID = "REQ_GENERAL_OBJECT_PART"
_MULTI_IMAGE_REQUIREMENT_ID = "REQ_GENERAL_MULTI_IMAGE"


def load_requirements_by_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["requirement_id"]: row for row in rows}


def select_requirement(
    requirements_by_id: Dict[str, Dict[str, str]],
    claim_object: str,
    claimed_issue_types: List[str],
    num_images: int,
) -> Dict[str, str]:
    if num_images > 1:
        base_id = _MULTI_IMAGE_REQUIREMENT_ID
    else:
        base_id = _FALLBACK_REQUIREMENT_ID

    for keywords, req_id in _ISSUE_FAMILY_KEYWORDS.get(claim_object, []):
        if any(issue in keywords for issue in claimed_issue_types):
            base_id = req_id
            break

    return requirements_by_id.get(base_id, requirements_by_id.get(_FALLBACK_REQUIREMENT_ID))


def compute_evidence_standard(image_analyses: List[Dict]) -> tuple:
    """Returns (evidence_standard_met: bool, reason: str), based purely on the
    per-image object_match / part_match signals already produced by stage 2.
    """
    if not image_analyses:
        return False, "No usable image was submitted, so the claimed condition cannot be evaluated."

    matching = [a for a in image_analyses if a["object_match"] and a["part_match"]]
    total = len(image_analyses)
    if matching:
        ids = ", ".join(a["image_id"] for a in matching)
        return (
            True,
            f"{len(matching)} of {total} submitted image(s) ({ids}) clearly show the claimed object and part.",
        )

    object_matching = [a for a in image_analyses if a["object_match"]]
    if object_matching:
        return (
            False,
            "The submitted images show the claimed object, but none clearly show the specific claimed part.",
        )
    return False, "None of the submitted images clearly show the claimed object."


def compute_valid_image(image_analyses: List[Dict], any_load_failed: bool, all_failed: bool) -> bool:
    if all_failed:
        return False
    usable = [a for a in image_analyses if not a["non_original_image"]]
    return len(usable) > 0


def get_user_history_risk(history_row: Optional[Dict[str, str]]) -> tuple:
    """Returns (history_risk: bool, manual_review_required: bool, summary: str)."""
    if not history_row:
        return False, False, ""
    flags = set(io_utils.split_semicolon_field(history_row.get("history_flags", "")))
    history_risk = "user_history_risk" in flags
    manual_review_required = "manual_review_required" in flags
    return history_risk, manual_review_required, history_row.get("history_summary", "")


def deterministic_risk_flags(
    image_analyses: List[Dict],
    text_instruction_present: bool,
    history_risk: bool,
    history_manual_review: bool,
    evidence_standard_met: bool,
) -> set:
    flags = set()
    for a in image_analyses:
        flags.update(a["quality_flags"])
        if a["possible_manipulation"]:
            flags.add("possible_manipulation")
        if a["non_original_image"]:
            flags.add("non_original_image")

    if text_instruction_present:
        flags.add("text_instruction_present")
    if history_risk:
        flags.add("user_history_risk")
    if history_manual_review:
        flags.add("manual_review_required")

    if not evidence_standard_met and image_analyses:
        if not any(a["object_match"] for a in image_analyses):
            flags.add("wrong_object")
        elif not any(a["object_match"] and a["part_match"] for a in image_analyses):
            flags.add("wrong_object_part")

    return flags
