"""Allowed-value enums and dynamic pydantic models for Gemini structured output.

Object-part enums are object-specific (car/laptop/package), so the pydantic
models used as `response_schema` are built per-row via the `build_*` factory
functions below, constrained to the correct part list for that row's
`claim_object`.
"""

from typing import List, Literal

from pydantic import BaseModel, create_model

CLAIM_STATUS = ("supported", "contradicted", "not_enough_information")

ISSUE_TYPES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

OBJECT_PARTS = {
    "car": (
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    ),
    "laptop": (
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    ),
    "package": (
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    ),
}

RISK_FLAGS = (
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

# Subset of risk flags that stage-2 (per-image) analysis is allowed to raise.
QUALITY_FLAGS = ("blurry_image", "low_light_or_glare", "cropped_or_obstructed", "wrong_angle")

SEVERITY = ("none", "low", "medium", "high", "unknown")


def _parts_for(claim_object: str) -> tuple:
    return OBJECT_PARTS.get(claim_object, OBJECT_PARTS["car"] + OBJECT_PARTS["laptop"] + OBJECT_PARTS["package"])


def build_claim_parse_model(claim_object: str) -> type[BaseModel]:
    parts = _parts_for(claim_object)
    return create_model(
        "ClaimParseResult",
        claim_summary=(str, ...),
        claimed_issue_types=(List[Literal[ISSUE_TYPES]], ...),
        claimed_parts=(List[Literal[parts]], ...),
        text_instruction_present=(bool, ...),
        language=(str, ...),
    )


def build_image_analysis_model(claim_object: str) -> type[BaseModel]:
    parts = _parts_for(claim_object)
    return create_model(
        "ImageAnalysisResult",
        visible_issue_type=(Literal[ISSUE_TYPES], ...),
        object_part=(Literal[parts], ...),
        severity=(Literal[SEVERITY], ...),
        object_match=(bool, ...),
        part_match=(bool, ...),
        quality_flags=(List[Literal[QUALITY_FLAGS]], ...),
        possible_manipulation=(bool, ...),
        non_original_image=(bool, ...),
        caption=(str, ...),
    )


def build_decision_model(claim_object: str) -> type[BaseModel]:
    parts = _parts_for(claim_object)
    return create_model(
        "DecisionResult",
        evidence_standard_met=(bool, ...),
        evidence_standard_met_reason=(str, ...),
        risk_flags=(List[Literal[RISK_FLAGS]], ...),
        issue_type=(Literal[ISSUE_TYPES], ...),
        object_part=(Literal[parts], ...),
        claim_status=(Literal[CLAIM_STATUS], ...),
        claim_status_justification=(str, ...),
        supporting_image_ids=(List[str], ...),
        valid_image=(bool, ...),
        severity=(Literal[SEVERITY], ...),
    )


def coerce_issue_type(value: str) -> str:
    return value if value in ISSUE_TYPES else "unknown"


def coerce_object_part(value: str, claim_object: str) -> str:
    parts = _parts_for(claim_object)
    return value if value in parts else "unknown"


def coerce_severity(value: str) -> str:
    return value if value in SEVERITY else "unknown"


def coerce_claim_status(value: str) -> str:
    return value if value in CLAIM_STATUS else "not_enough_information"


def coerce_risk_flags(values) -> list:
    return [v for v in dict.fromkeys(values) if v in RISK_FLAGS and v != "none"]
