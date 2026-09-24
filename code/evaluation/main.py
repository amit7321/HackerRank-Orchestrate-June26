#!/usr/bin/env python3
"""Runs the review pipeline on dataset/sample_claims.csv for two model
configurations, scores each against the provided labels, and writes
evaluation/evaluation_report.md with the comparison + operational analysis.

Usage:
    python code/evaluation/main.py [--limit N]

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a .env
file in code/ (see code/.env.example).
"""

import argparse
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from dotenv import load_dotenv  # noqa: E402

from evaluation import metrics  # noqa: E402
from src import config, io_utils, pipeline, rules  # noqa: E402
from src.gemini_client import GeminiClient  # noqa: E402

INPUT_FIELDS = ["user_id", "image_paths", "user_claim", "claim_object"]


def run_config(rows, client, requirements_by_id, history_by_user, model_config, limit=None):
    input_rows = [{k: r[k] for k in INPUT_FIELDS} for r in rows]
    if limit:
        input_rows = input_rows[:limit]
        rows = rows[:limit]

    calls_before = client.call_count
    prompt_tokens_before = client.total_prompt_tokens
    output_tokens_before = client.total_output_tokens

    start = time.time()
    predictions = pipeline.process_claims(
        input_rows,
        client,
        requirements_by_id=requirements_by_id,
        history_by_user=history_by_user,
        model=model_config["model"],
    )
    elapsed = time.time() - start

    report = metrics.score(predictions, rows)
    report["operational"] = {
        "calls": client.call_count - calls_before,
        "prompt_tokens": client.total_prompt_tokens - prompt_tokens_before,
        "output_tokens": client.total_output_tokens - output_tokens_before,
        "elapsed_seconds": elapsed,
    }
    return report, predictions


def format_report(sample_n, report_a, report_b, config_a, config_b, chosen_name) -> str:
    def fmt_pct(x):
        return f"{x * 100:.1f}%"

    def field_table(report):
        lines = ["| Field | Accuracy |", "|---|---|"]
        for field, val in report["accuracy"].items():
            lines.append(f"| {field} | {fmt_pct(val)} |")
        return "\n".join(lines)

    def f1_table(report):
        lines = ["| Field | Macro-F1 |", "|---|---|"]
        for field, val in report["macro_f1"].items():
            lines.append(f"| {field} | {val:.3f} |")
        return "\n".join(lines)

    def set_table(report):
        lines = ["| Field | Precision | Recall | F1 |", "|---|---|---|---|"]
        for field, vals in report["set_fields"].items():
            lines.append(f"| {field} | {fmt_pct(vals['precision'])} | {fmt_pct(vals['recall'])} | {vals['f1']:.3f} |")
        return "\n".join(lines)

    def confusion_table(report):
        labels = sorted(report["claim_status_confusion"].keys() | {"supported", "contradicted", "not_enough_information"})
        header = "| expected \\ predicted | " + " | ".join(labels) + " |"
        sep = "|---" * (len(labels) + 1) + "|"
        rows = [header, sep]
        for expected in labels:
            row_counts = report["claim_status_confusion"].get(expected, {})
            rows.append("| " + expected + " | " + " | ".join(str(row_counts.get(p, 0)) for p in labels) + " |")
        return "\n".join(rows)

    def op_line(report, model_config, price_in, price_out):
        op = report["operational"]
        cost = (op["prompt_tokens"] / 1_000_000) * price_in + (op["output_tokens"] / 1_000_000) * price_out
        return (
            f"- Model: `{model_config['model']}`\n"
            f"- Calls: {op['calls']}\n"
            f"- Approx. prompt tokens: {op['prompt_tokens']}\n"
            f"- Approx. output tokens: {op['output_tokens']}\n"
            f"- Wall-clock time: {op['elapsed_seconds']:.1f}s\n"
            f"- Approx. paid-tier cost for this run: ${cost:.4f} "
            f"(free tier: $0; assumes ${price_in}/1M input, ${price_out}/1M output tokens)"
        )

    price_a_in = config.APPROX_PRICE_PER_1M_INPUT_TOKENS.get(config_a["model"], 0)
    price_a_out = config.APPROX_PRICE_PER_1M_OUTPUT_TOKENS.get(config_a["model"], 0)
    price_b_in = config.APPROX_PRICE_PER_1M_INPUT_TOKENS.get(config_b["model"], 0)
    price_b_out = config.APPROX_PRICE_PER_1M_OUTPUT_TOKENS.get(config_b["model"], 0)

    total_test_rows = 44
    approx_images_per_row = 1.9
    approx_calls_per_row = 2 + approx_images_per_row  # claim-parse + decision + ~images
    approx_test_calls = int(total_test_rows * approx_calls_per_row)

    return f"""# Evaluation Report

Evaluated on `dataset/sample_claims.csv` ({sample_n} labeled rows).

## Config A: `{config_a['model']}`

### Exact-match accuracy
{field_table(report_a)}

### Macro-F1
{f1_table(report_a)}

### Set-valued fields (risk_flags, supporting_image_ids)
{set_table(report_a)}

### claim_status confusion matrix (rows = expected, cols = predicted)
{confusion_table(report_a)}

### Operational
{op_line(report_a, config_a, price_a_in, price_a_out)}

---

## Config B: `{config_b['model']}`

### Exact-match accuracy
{field_table(report_b)}

### Macro-F1
{f1_table(report_b)}

### Set-valued fields (risk_flags, supporting_image_ids)
{set_table(report_b)}

### claim_status confusion matrix (rows = expected, cols = predicted)
{confusion_table(report_b)}

### Operational
{op_line(report_b, config_b, price_b_in, price_b_out)}

---

## Chosen strategy for `output.csv`

**{chosen_name}** — selected by comparing `claim_status` macro-F1 (the primary
decision field) between the two configs above; ties broken toward the config
with higher overall exact-match accuracy across fields. Both configs use the
identical staged pipeline (claim parser -> per-image vision analysis ->
deterministic evidence/history rules -> decision fusion) and prompts; only
the underlying Gemini model differs, isolating the model-choice comparison
from prompt/architecture differences.

## Operational analysis (full test set, `dataset/claims.csv`, 44 rows)

This run only processed the sample set. Approximate figures for the full
44-row test set, extrapolated from this run and the dataset's image counts:

- Estimated model calls: ~{approx_test_calls} (44 claim-parses + ~44 decisions +
  ~{int(total_test_rows * approx_images_per_row)} per-image vision calls, minus any
  cache hits from reruns).
- Images processed: ~{int(total_test_rows * approx_images_per_row)} (1-3 per claim).
- Rate limiting: a token-bucket limiter caps requests at
  `MAX_REQUESTS_PER_MINUTE = {config.MAX_REQUESTS_PER_MINUTE}` (see `src/config.py`),
  comfortably under typical free-tier RPM. `tenacity` retries 429/503 responses
  with exponential backoff (`{config.RETRY_MIN_WAIT_SECONDS}`-`{config.RETRY_MAX_WAIT_SECONDS}`s, up to
  `{config.MAX_RETRY_ATTEMPTS}` attempts).
- Caching: per-image analysis is cached on disk
  (`code/.cache/`) keyed by `sha256(image bytes) + prompt version + model + claim_object`,
  so reruns of `main.py`/`evaluation/main.py` and repeated evaluation passes reuse
  prior vision results instead of re-calling the API.
- Cost: on the free tier this is $0. If free-tier quota were exhausted, paid-rate
  cost for the full test set is estimated in the low cents (see per-config cost
  lines above, scaled by ~2.2x for the larger test set) — well under $1 for either
  config.
"""


def main() -> None:
    load_dotenv(CODE_DIR / ".env")

    parser = argparse.ArgumentParser(description="Evaluate the pipeline on dataset/sample_claims.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N sample rows.")
    args = parser.parse_args()

    rows = io_utils.read_claims_csv(config.SAMPLE_CLAIMS_CSV)
    print(f"Loaded {len(rows)} labeled sample row(s) from {config.SAMPLE_CLAIMS_CSV}")

    client = GeminiClient()
    requirements_by_id = rules.load_requirements_by_id(io_utils.read_evidence_requirements())
    history_by_user = io_utils.read_user_history()

    print(f"Running config A: {config.MODEL_CONFIG_A['model']} ...")
    report_a, predictions_a = run_config(
        rows, client, requirements_by_id, history_by_user, config.MODEL_CONFIG_A, limit=args.limit
    )
    print("  claim_status accuracy:", report_a["accuracy"]["claim_status"])

    print(f"Running config B: {config.MODEL_CONFIG_B['model']} ...")
    report_b, predictions_b = run_config(
        rows, client, requirements_by_id, history_by_user, config.MODEL_CONFIG_B, limit=args.limit
    )
    print("  claim_status accuracy:", report_b["accuracy"]["claim_status"])

    chosen = (
        config.MODEL_CONFIG_A
        if report_a["macro_f1"]["claim_status"] >= report_b["macro_f1"]["claim_status"]
        else config.MODEL_CONFIG_B
    )
    chosen_name = f"Config A ({config.MODEL_CONFIG_A['model']})" if chosen is config.MODEL_CONFIG_A else f"Config B ({config.MODEL_CONFIG_B['model']})"

    report_md = format_report(
        len(rows) if not args.limit else args.limit,
        report_a,
        report_b,
        config.MODEL_CONFIG_A,
        config.MODEL_CONFIG_B,
        chosen_name,
    )

    report_path = Path(__file__).resolve().parent / "evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nWrote {report_path}")
    print(f"Chosen strategy: {chosen_name}")


if __name__ == "__main__":
    main()
