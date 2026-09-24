"""Stage 1-4 orchestration for one claim row.

  (1) claim parser (text)      -> claimed issue/parts/summary/instruction flag
  (2) image analyzer (vision)  -> per-image visible issue/part/severity/flags (cached)
  (3) evidence + history rules -> evidence_standard_met, valid_image, history risk (no model)
  (4) decision fusion (text)   -> final 14-column record
"""

from typing import Dict, List, Optional

from . import cache, config, images, io_utils, prompts, rules, schema


def _claimed_parts_for_prompt(claim_object: str, parts: List[str]) -> List[str]:
    return [p for p in parts if p in schema.OBJECT_PARTS.get(claim_object, ())]


def _analyze_image(
    client,
    model: str,
    loaded: images.LoadedImage,
    claim_object: str,
    claimed_parts: List[str],
    claim_summary: str,
) -> Dict:
    if not loaded.load_ok:
        return {
            "image_id": loaded.image_id,
            "visible_issue_type": "unknown",
            "object_part": "unknown",
            "severity": "unknown",
            "object_match": False,
            "part_match": False,
            "quality_flags": [],
            "possible_manipulation": False,
            "non_original_image": True,
            "caption": f"Image could not be loaded: {loaded.error}",
        }

    cache_key = cache.make_key(
        "image_analysis",
        config.PROMPT_VERSION,
        model,
        claim_object,
        cache.image_bytes_hash(loaded.data),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        result = dict(cached)
        result["image_id"] = loaded.image_id
        return result

    prompt_text = prompts.image_analysis_prompt(claim_object, claimed_parts, claim_summary)
    response_model = schema.build_image_analysis_model(claim_object)
    parsed = client.generate_structured(
        model=model,
        contents=[prompt_text, client.image_part(loaded.data, loaded.mime_type)],
        response_schema=response_model,
    )

    result = {
        "visible_issue_type": schema.coerce_issue_type(parsed.visible_issue_type),
        "object_part": schema.coerce_object_part(parsed.object_part, claim_object),
        "severity": schema.coerce_severity(parsed.severity),
        "object_match": bool(parsed.object_match),
        "part_match": bool(parsed.part_match),
        "quality_flags": list(parsed.quality_flags),
        "possible_manipulation": bool(parsed.possible_manipulation),
        "non_original_image": bool(parsed.non_original_image),
        "caption": parsed.caption,
    }
    cache.set(cache_key, result)

    result_with_id = dict(result)
    result_with_id["image_id"] = loaded.image_id
    return result_with_id


def process_claim(
    row: Dict[str, str],
    client,
    requirements_by_id: Dict[str, Dict[str, str]],
    history_by_user: Dict[str, Dict[str, str]],
    model: str = config.DEFAULT_CONFIG["model"],
) -> Dict[str, str]:
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    user_claim = row["user_claim"]
    image_paths_field = row["image_paths"]

    # Stage 1: claim parsing
    claim_parse_model = schema.build_claim_parse_model(claim_object)
    stage1 = client.generate_structured(
        model=model,
        contents=[prompts.claim_parser_prompt(user_claim, claim_object)],
        response_schema=claim_parse_model,
    )
    claimed_issue_types = [schema.coerce_issue_type(i) for i in stage1.claimed_issue_types]
    claimed_parts = _claimed_parts_for_prompt(claim_object, list(stage1.claimed_parts))
    text_instruction_present = bool(stage1.text_instruction_present)

    # Stage 2: per-image vision analysis (cached)
    loaded_images = images.load_images(image_paths_field)
    image_analyses = [
        _analyze_image(client, model, li, claim_object, claimed_parts, stage1.claim_summary)
        for li in loaded_images
    ]
    all_failed = len(loaded_images) > 0 and all(not li.load_ok for li in loaded_images)
    any_load_failed = any(not li.load_ok for li in loaded_images)

    # Stage 3: deterministic evidence + history rules
    requirement = rules.select_requirement(requirements_by_id, claim_object, claimed_issue_types, len(loaded_images))
    evidence_standard_met, evidence_reason = rules.compute_evidence_standard(image_analyses)
    valid_image = rules.compute_valid_image(image_analyses, any_load_failed, all_failed)
    history_row = history_by_user.get(user_id)
    history_risk, history_manual_review, history_summary = rules.get_user_history_risk(history_row)
    det_flags = rules.deterministic_risk_flags(
        image_analyses, text_instruction_present, history_risk, history_manual_review, evidence_standard_met
    )

    # Stage 4: decision fusion
    decision_model = schema.build_decision_model(claim_object)
    decision = client.generate_structured(
        model=model,
        contents=[
            prompts.decision_prompt(
                claim_object=claim_object,
                claim_summary=stage1.claim_summary,
                claimed_issue_types=claimed_issue_types,
                claimed_parts=claimed_parts,
                text_instruction_present=text_instruction_present,
                image_analyses=image_analyses,
                evidence_requirement_text=requirement["minimum_image_evidence"] if requirement else "",
                history_risk=history_risk,
                history_summary=history_summary,
            )
        ],
        response_schema=decision_model,
    )

    model_flags = set(schema.coerce_risk_flags(decision.risk_flags))
    final_flags = sorted(det_flags | model_flags)

    valid_image_ids = {a["image_id"] for a in image_analyses}
    supporting_ids = [i for i in decision.supporting_image_ids if i in valid_image_ids]

    return {
        "user_id": user_id,
        "image_paths": image_paths_field,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": io_utils.format_bool(evidence_standard_met),
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": io_utils.join_semicolon_field(final_flags),
        "issue_type": schema.coerce_issue_type(decision.issue_type),
        "object_part": schema.coerce_object_part(decision.object_part, claim_object),
        "claim_status": schema.coerce_claim_status(decision.claim_status),
        "claim_status_justification": decision.claim_status_justification,
        "supporting_image_ids": io_utils.join_semicolon_field(supporting_ids),
        "valid_image": io_utils.format_bool(valid_image),
        "severity": schema.coerce_severity(decision.severity),
    }


def process_claims(
    rows: List[Dict[str, str]],
    client,
    requirements_by_id: Optional[Dict[str, Dict[str, str]]] = None,
    history_by_user: Optional[Dict[str, Dict[str, str]]] = None,
    model: str = config.DEFAULT_CONFIG["model"],
    on_row_done=None,
) -> List[Dict[str, str]]:
    if requirements_by_id is None:
        requirements_by_id = rules.load_requirements_by_id(io_utils.read_evidence_requirements())
    if history_by_user is None:
        history_by_user = io_utils.read_user_history()

    results = []
    for i, row in enumerate(rows):
        result = process_claim(row, client, requirements_by_id, history_by_user, model=model)
        results.append(result)
        if on_row_done:
            on_row_done(i, row, result)
    return results
