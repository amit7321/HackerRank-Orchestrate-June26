# Evaluation Report

Evaluated on `dataset/sample_claims.csv` (2 labeled rows).

## Config A: `gemini-3.5-flash-lite`

### Exact-match accuracy
| Field | Accuracy |
|---|---|
| evidence_standard_met | 50.0% |
| issue_type | 50.0% |
| object_part | 100.0% |
| claim_status | 50.0% |
| valid_image | 100.0% |
| severity | 0.0% |

### Macro-F1
| Field | Macro-F1 |
|---|---|
| claim_status | 0.333 |
| issue_type | 0.333 |
| object_part | 1.000 |

### Set-valued fields (risk_flags, supporting_image_ids)
| Field | Precision | Recall | F1 |
|---|---|---|---|
| risk_flags | 100.0% | 33.3% | 0.500 |
| supporting_image_ids | 100.0% | 33.3% | 0.500 |

### claim_status confusion matrix (rows = expected, cols = predicted)
| expected \ predicted | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 0 | 0 | 0 |
| not_enough_information | 1 | 0 | 0 |
| supported | 0 | 0 | 1 |

### Operational
- Model: `gemini-3.5-flash-lite`
- Calls: 7
- Approx. prompt tokens: 6591
- Approx. output tokens: 806
- Wall-clock time: 10.3s
- Approx. paid-tier cost for this run: $0.0040 (free tier: $0; assumes $0.3/1M input, $2.5/1M output tokens)

---

## Config B: `gemini-3.1-flash-lite`

### Exact-match accuracy
| Field | Accuracy |
|---|---|
| evidence_standard_met | 50.0% |
| issue_type | 0.0% |
| object_part | 100.0% |
| claim_status | 0.0% |
| valid_image | 100.0% |
| severity | 0.0% |

### Macro-F1
| Field | Macro-F1 |
|---|---|
| claim_status | 0.000 |
| issue_type | 0.000 |
| object_part | 1.000 |

### Set-valued fields (risk_flags, supporting_image_ids)
| Field | Precision | Recall | F1 |
|---|---|---|---|
| risk_flags | 50.0% | 33.3% | 0.400 |
| supporting_image_ids | 100.0% | 0.0% | 0.000 |

### claim_status confusion matrix (rows = expected, cols = predicted)
| expected \ predicted | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 0 | 0 | 0 |
| not_enough_information | 1 | 0 | 0 |
| supported | 1 | 0 | 0 |

### Operational
- Model: `gemini-3.1-flash-lite`
- Calls: 7
- Approx. prompt tokens: 6602
- Approx. output tokens: 804
- Wall-clock time: 103.5s
- Approx. paid-tier cost for this run: $0.0029 (free tier: $0; assumes $0.25/1M input, $1.5/1M output tokens)

---

## Chosen strategy for `output.csv`

**Config A (gemini-3.5-flash-lite)** — selected by comparing `claim_status` macro-F1 (the primary
decision field) between the two configs above; ties broken toward the config
with higher overall exact-match accuracy across fields. Both configs use the
identical staged pipeline (claim parser -> per-image vision analysis ->
deterministic evidence/history rules -> decision fusion) and prompts; only
the underlying Gemini model differs, isolating the model-choice comparison
from prompt/architecture differences.

## Operational analysis (full test set, `dataset/claims.csv`, 44 rows)

This run only processed the sample set. Approximate figures for the full
44-row test set, extrapolated from this run and the dataset's image counts:

- Estimated model calls: ~171 (44 claim-parses + ~44 decisions +
  ~83 per-image vision calls, minus any
  cache hits from reruns).
- Images processed: ~83 (1-3 per claim).
- Rate limiting: a token-bucket limiter caps requests at
  `MAX_REQUESTS_PER_MINUTE = 8` (see `src/config.py`),
  comfortably under typical free-tier RPM. `tenacity` retries 429/503 responses
  with exponential backoff (`2`-`30`s, up to
  `5` attempts).
- Caching: per-image analysis is cached on disk
  (`code/.cache/`) keyed by `sha256(image bytes) + prompt version + model + claim_object`,
  so reruns of `main.py`/`evaluation/main.py` and repeated evaluation passes reuse
  prior vision results instead of re-calling the API.
- Cost: on the free tier this is $0. If free-tier quota were exhausted, paid-rate
  cost for the full test set is estimated in the low cents (see per-config cost
  lines above, scaled by ~2.2x for the larger test set) — well under $1 for either
  config.
