# Multi-Modal Evidence Review — Solution

A staged Gemini pipeline that reads `dataset/claims.csv` and produces
`output.csv` at the repo root, deciding whether submitted images support,
contradict, or provide insufficient evidence for each damage claim.

## Setup

```bash
pip install -r code/requirements.txt
cp code/.env.example code/.env   # then fill in GEMINI_API_KEY
```

Get a free-tier key at https://aistudio.google.com/apikey. The key is read
only from the environment (`GEMINI_API_KEY`, falling back to
`GOOGLE_API_KEY`) or a `.env` file next to `main.py` — never hardcoded.

## Run

```bash
# Smoke test on a few rows first
python code/main.py --limit 3

# Full run: produces output.csv at the repo root
python code/main.py

# Evaluation on dataset/sample_claims.csv (both model configs), writes
# code/evaluation/evaluation_report.md
python code/evaluation/main.py
```

## Architecture

```
claims.csv row
  (1) CLAIM PARSER (Gemini, text)   -> claim_summary, claimed_issue_types,
                                        claimed_parts, text_instruction_present,
                                        language
  (2) IMAGE ANALYZER (Gemini vision,-> per image: visible_issue_type, object_part,
      one call per image, cached)      severity, object_match, part_match,
                                        quality_flags, manipulation signals, caption
  (3) EVIDENCE + HISTORY (rules,    -> evidence_standard_met, valid_image,
      no model)                        user_history_risk (deterministic)
  (4) DECISION FUSION (Gemini,      -> final 14-column record
      text, structured JSON)
```

Rationale for staging:

- **Cheap text parsing is separated from expensive vision calls.** The claim
  parser and decision fusion are text-only; only stage 2 sends image bytes.
- **Per-image vision analysis is cached on disk** (`code/.cache/`, keyed by
  `sha256(image bytes) + prompt version + model + claim_object`), so reruns
  and the two evaluation configs reuse prior vision results instead of
  re-calling the API.
- **Prompt injection is isolated.** The transcript text is only ever shown to
  the text-only claim parser and decision-fusion stages with an explicit
  instruction to ignore embedded instructions; the vision stage is told the
  same about text visible inside photos. `text_instruction_present` is
  recorded as a risk flag but never allowed to change `claim_status`.
- **Evidence sufficiency and user-history risk are deterministic**
  (`src/rules.py`), not model calls: `evidence_standard_met` is computed
  directly from the `object_match`/`part_match` booleans that stage 2
  already produced, and `user_history_risk` / `manual_review_required` are
  looked up straight from `dataset/user_history.csv`. This keeps the
  pipeline reproducible and keeps model calls focused on genuinely visual
  judgments (per the "deterministic where possible" project contract).
- **Every image mime type is sniffed from magic bytes**, not the file
  extension — several dataset images are WebP despite a `.jpg` extension.

All enum-valued outputs (`issue_type`, `object_part`, `severity`,
`claim_status`, `risk_flags`) are constrained at generation time with Gemini
`response_schema` (via dynamic per-row pydantic models scoped to the correct
object-part list), and are coerced again in `src/schema.py` before being
written, so an illegal value can never reach `output.csv`.

## File layout

```
code/
├── main.py                     # entry: reads claims.csv -> writes output.csv
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py                # model names, paths, enums, rate-limit constants
│   ├── io_utils.py               # CSV read/write with the exact 14-column schema
│   ├── images.py                 # image path resolution + real-mime-type sniffing
│   ├── gemini_client.py          # rate-limited, retried, structured-output client
│   ├── prompts.py                # claim-parser / image-analyzer / decision prompts
│   ├── schema.py                 # allowed-value enums + dynamic pydantic models
│   ├── cache.py                  # disk cache for per-image analysis
│   ├── rules.py                  # evidence-requirement matching + history risk
│   └── pipeline.py               # orchestrates stages 1-4 per claim
└── evaluation/
    ├── main.py                   # runs the pipeline on sample_claims.csv, scores it
    ├── metrics.py                # accuracy / macro-F1 / confusion matrix / set P-R
    └── evaluation_report.md      # generated: metrics + 2-config comparison + ops analysis
```

## Model comparison

`evaluation/main.py` runs the identical pipeline under two configurations
(only the model differs) and reports both in `evaluation_report.md`:

- **Config A** — `gemini-3.5-flash-lite` (used to produce the final `output.csv`
  unless the report shows Config B scoring higher on `claim_status` macro-F1).
- **Config B** — `gemini-3.1-flash-lite` (prior-generation lite model, comparison point).

Both configs use a **flash-lite** model, not "flash vs flash-lite" as you
might expect. As of this writing, Gemini's free tier caps full "flash"
models (including the `gemini-flash-latest` alias) at roughly 20
requests/day, while flash-lite models get roughly 500/day. This pipeline
needs on the order of 150-250 calls to process the sample evaluation plus
the full 44-row test set, so a full flash model's free-tier quota cannot
complete even one real run — only flash-lite is viable for a $0 run today.
See `src/config.py` to change either model if you have paid-tier access or
Google's quotas change.

## Cost / rate-limit strategy

- Token-bucket rate limiter (`src/gemini_client.py`) caps requests at
  `MAX_REQUESTS_PER_MINUTE` (default 8) to stay comfortably under typical
  free-tier RPM limits.
- `tenacity`-based exponential backoff retries 429/503/rate-limit errors.
- Disk cache on per-image vision analysis avoids repeated vision calls
  across reruns and across the two evaluation configs (when they share a
  cache key).
- See `evaluation/evaluation_report.md` for measured call counts, token
  usage, and approximate cost once you run the evaluation.

## Notes

- Nothing is hardcoded from `dataset/sample_claims.csv` labels; the pipeline
  only ever reads the four input columns for any row, sample or test.
- Boolean output fields are written as lowercase `true`/`false`;
  `risk_flags` and `supporting_image_ids` are semicolon-joined or `none`, per
  the required schema.
