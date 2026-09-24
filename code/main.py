#!/usr/bin/env python3
"""Entry point: reads dataset/claims.csv, runs the review pipeline, writes
output.csv at the repo root.

Usage:
    python code/main.py [--limit N] [--model MODEL_NAME]

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a .env
file next to this script (see .env.example).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from src import config, io_utils, pipeline, rules  # noqa: E402
from src.gemini_client import GeminiClient  # noqa: E402


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="Run the evidence-review pipeline on dataset/claims.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke testing).")
    parser.add_argument("--model", type=str, default=config.DEFAULT_CONFIG["model"], help="Gemini model to use.")
    parser.add_argument("--output", type=str, default=str(config.OUTPUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    rows = io_utils.read_claims_csv(config.CLAIMS_CSV)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} claim(s) from {config.CLAIMS_CSV}")

    client = GeminiClient()
    requirements_by_id = rules.load_requirements_by_id(io_utils.read_evidence_requirements())
    history_by_user = io_utils.read_user_history()

    start = time.time()

    def on_row_done(i, row, result):
        print(f"[{i + 1}/{len(rows)}] {row['user_id']} -> claim_status={result['claim_status']}")

    results = pipeline.process_claims(
        rows,
        client,
        requirements_by_id=requirements_by_id,
        history_by_user=history_by_user,
        model=args.model,
        on_row_done=on_row_done,
    )

    elapsed = time.time() - start

    output_path = Path(args.output)
    io_utils.write_output_csv(results, output_path)

    print(f"\nWrote {len(results)} row(s) to {output_path}")
    print(f"Model calls: {client.call_count}  |  elapsed: {elapsed:.1f}s")
    print(f"Prompt tokens (approx): {client.total_prompt_tokens}  |  output tokens (approx): {client.total_output_tokens}")


if __name__ == "__main__":
    main()
