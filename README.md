# Removing sensitive sentences from Italian expert appraisals

Project Work in NLP (3 CFU), University of Bologna. Student: Klein Tahiraj.

## Task

Italian court-appointed expert appraisals (*perizie*) from real-estate enforcement
proceedings sometimes mention the health, criminal matters, minors or family
circumstances of the people involved. This project removes **whole sentences**
that carry such data, with four multi-label heads (`salute`, `penale`, `minori`,
`famiglia`), and keeps everything else. Under the two-gate policy a sentence is
positive only if (1) it concerns an identifiable living person, masked parties
included, and (2) it asserts the sensitive fact, so negated, hypothetical, generic
and deceased-only statements are kept. LLM labels (two local annotators, then the
GPT-6.1 Sol teacher) are distilled into a local `microsoft/mdeberta-v3-base`
student. Models are `A0`, `it1` (more teacher labels), `it3` (counterfactual
training), `it3m` (plus a pairwise margin loss) and `it3m_c` (`it3m` with a lower
penale threshold). Baselines are Presidio, GLiNER-PII and Gemma 4 12B zero-shot.

## Install and data

Install with `uv sync` (`uv sync --extra baselines` adds Presidio and GLiNER).
The appraisal corpus contains personal data and is **not distributed**. Every
entry point accepts `--help` without it. To run the stages, set

- `PRIVACY_DATA_ROOT`, the private working directory (corpus, labels, models, caches);
- `PERIZIA_TEXT_ROOT`, the extracted document texts (default `$PRIVACY_DATA_ROOT/texts`);
- `PERIZIA_FACTS_ROOT`, per-document metadata used only for corpus statistics.

## Entry points

Run from the repository root, in pipeline order. The remaining scripts
(`phase0_train`, `phase0_eval`, `final_train`, `clean_mining`) cover variants
explored but not reported.

| Stage | Command |
|---|---|
| Identifier mask (names, fiscal codes, birth dates, contacts), applied to every text before any other step | `src/perizia_privacy/masking.py` (`mask`) |
| Corpus, near-duplicate groups, group split | `uv run python -m perizia_privacy.corpus build` / `select` |
| A0-train (local annotators, synthetic rows) | `uv run python -m perizia_privacy.teachers`, `.annotation`, `.synthetic` |
| `A0` training and prediction | `uv run python -m perizia_privacy.training train` / `predict` |
| Teacher decision tree and label probes | `uv run python -m scripts.phase0_label` |
| Teacher rows and Edge suite, `it1` | `uv run python -m scripts.proof_score`, then `proof_package`, `proof_label`, `proof_iter2`, `proof_train`, `proof_eval` |
| Held-out, keyword subset, IPW sample | `uv run python -m scripts.final_instrument build` / `label` / `extend` |
| Twins | `uv run python -m scripts.counterfactuals supplement` / `edit` / `relabel` / `score` |
| Cassazione | `uv run python -m scripts.sentenze fetch` / `mine` / `label` / `score` |
| CF pairs, Cass-train, dev set | `uv run python -m scripts.it3_data <step>` |
| `it3` / `it3m` training and thresholds | `uv run python -m scripts.it3_train train` / `thresholds` |
| `A0` vs `it1` evaluation | `uv run python -m scripts.final_eval` |
| `it3` / `it3m` evaluation | `uv run python -m scripts.it3_eval step0` / `final` |
| `it3m_c` | `uv run python -m scripts.hybrid_eval` |
| Presidio and GLiNER-PII | `uv run python -m scripts.baselines_ner collect` / `score` / `report` |
| GLiNER-PII calibrated on validation | `uv run python -m scripts.gliner_calibrated collect` / `score` / `report` |
| Gemma 4 12B | `uv run python -m scripts.baselines_llm final` / `sentenze` |
| Canonical re-run of `A0` and `it3m_c` | `uv run python -m scripts.canonical_eval score` / `report` |
| Human check of the LLM labels | `uv run python -m scripts.human_agreement` |
| Inference speed | `uv run python -m scripts.inference_speed --out <file>` |
| Corpus statistics | `uv run python -m scripts.corpus_diversity` |
| Numbers table | `uv run python -m scripts.numbers` |

## Reproduce the report numbers

Without private data, `uv run python -m scripts.numbers` rebuilds
`results/numbers.json` and `results/NUMBERS.md` from the aggregate JSON in
`results/`. With the private data, regenerate the aggregates first.

```
uv run python -m scripts.canonical_eval score
uv run python -m scripts.canonical_eval report
uv run python -m scripts.baselines_ner report
uv run python -m scripts.baselines_llm final --deletion 600
uv run python -m scripts.baselines_llm sentenze
uv run python -m scripts.gliner_calibrated collect
uv run python -m scripts.gliner_calibrated score --labels en --part validation   # also it, and --part test
uv run python -m scripts.gliner_calibrated report
uv run python -m scripts.human_agreement
uv run python -m scripts.inference_speed --out results/speed-20261029/report.json
uv run python -m scripts.numbers
```

`results/NUMBERS.md` names the command behind each number. Training and labelling
are not needed for this, since scoring starts from saved checkpoints and stored labels.

## Layout

```
src/perizia_privacy/   library (corpus, annotation, training, evaluation)
scripts/               one module per experiment stage
results/               aggregate JSON only, one dated folder per stage, plus NUMBERS.md and HEADLINE.md
report/                course report
```

`results/` holds aggregates only, with no document text, sentences or identifiers.
