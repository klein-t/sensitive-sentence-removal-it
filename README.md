# Removing special-category sentences from Italian judicial appraisals

Project Work in NLP (3 CFU), University of Bologna. Student: Klein Tahiraj.

## Task

Italian court-appointed expert appraisals (*perizie*) from real-estate enforcement
proceedings are long, boilerplate-heavy documents that sometimes mention health,
criminal matters, minors or family circumstances of the people involved. GDPR
Art. 9 and Art. 10 treat these as special categories. This project trains a small
local classifier that **removes whole sentences** carrying such data about an
**identifiable living person**, and keeps everything else.

- Model: `microsoft/mdeberta-v3-base` with four multi-label heads:
  `salute` (health), `penale` (criminal), `minori` (minors), `famiglia` (family).
- Labels: distilled from LLM teacher labels under a fixed policy. A sentence is
  positive only if (1) it concerns an identifiable living person (masked parties
  count) and (2) it asserts the sensitive fact (negated, hypothetical,
  deceased-only and generic statements are kept).
- Variants: v1 (A0), harder-negative fine-tunes (it1, it2), a factored-input
  student (Phase 0, Run A), and counterfactual training with real-derived
  minimal pairs plus a pairwise margin loss (it3, it3m, it3m_c).
- Baselines: Presidio, GLiNER (multi and PII checkpoints), a local Gemma 12B
  instruct model zero-shot (and as a cascade behind the classifier).
- Evaluation: a 100-document audit, held-out positives on unseen document
  groups, an inverse-probability-weighted (IPW) estimate of harmless deletion,
  a cue-only clean frame, real-derived counterfactual pairs, and an
  out-of-domain test on public Corte di Cassazione decisions.

Headline numbers: [`results/HEADLINE.md`](results/HEADLINE.md).

## Pipeline

```
documents ─► corpus build (normalize, exact hash + MinHash LSH near-dup groups, group split)
          ─► sentence windows + cue mining
          ─► teacher labelling (two blind passes, decision-tree schema)
          ─► student training (v1 recipe; fine-tunes; counterfactual pairs + margin loss)
          ─► thresholds (per-head, chosen on validation / dev only)
          ─► frozen instruments (audit, held-out, IPW deletion, clean frame,
             counterfactual pairs, Cassazione) ─► baselines ─► comparison
```

| Stage | Entry point |
|---|---|
| Corpus build, near-dup grouping, group split | `python -m perizia_privacy.corpus build` / `select` |
| v1 local annotators, synthetic rows | `perizia_privacy.annotation`, `perizia_privacy.teachers`, `perizia_privacy.synthetic` |
| v1 student training / prediction | `python -m perizia_privacy.training train` / `predict` |
| Teacher schema and label probes (Phase 0) | `python -m scripts.phase0_label` |
| Fine-tune from v1 (it1) and its evaluation | `scripts.proof_score`, `proof_package`, `proof_label`, `proof_iter2`, `proof_train`, `proof_eval` |
| Factored student (Phase 0 curve, Run A) | `scripts.phase0_train`, `scripts.phase0_eval`, `scripts.final_train` |
| Frozen instrument (held-out positives, IPW deletion sample) | `python -m scripts.final_instrument build|label|extend` |
| Final comparison and decision rule | `python -m scripts.final_eval` |
| Clean cue-only frame | `python -m scripts.clean_mining frame|mine|label|score` |
| Real-derived counterfactual pairs | `python -m scripts.counterfactuals supplement|edit|relabel|score` |
| Cassazione out-of-domain set | `python -m scripts.sentenze fetch|mine|label|score` |
| Presidio / GLiNER baselines | `python -m scripts.baselines_ner collect|score|report` |
| Local LLM baseline and cascade | `python -m scripts.baselines_llm` |
| it3 / it3m data, training, evaluation | `scripts.it3_data`, `python -m scripts.it3_train train|thresholds`, `python -m scripts.it3_eval step0|final` |
| Hybrids and it3m_c | `python -m scripts.hybrid_eval` |
| Corpus statistics | `python -m scripts.corpus_diversity` |

Run every command from the repository root (`uv sync`, then `uv run python -m ...`;
`uv sync --extra baselines` for Presidio and GLiNER). Training uses Apple MPS
when available.

## Data

The appraisal corpus contains personal data and **cannot be shared**. Only
aggregate reports are published, in `results/`. With access to the private data,
set:

- `PRIVACY_DATA_ROOT`: the private working directory (corpus, labels, models, caches);
- `PERIZIA_TEXT_ROOT`: the extracted document texts (default `$PRIVACY_DATA_ROOT/texts`);
- `PERIZIA_FACTS_ROOT`: per-document metadata used only for corpus statistics.

The teacher labelling steps call a hosted LLM through a command-line client
(`perizia_privacy/retraining/codex_cli.py`) and the local baselines call a model
served on loopback. Nothing in `results/` contains document text, sentences or
document identifiers.

The Cassazione decisions are public acts (art. 5 l. 633/1941) fetched at no
more than one request every 1.5 s.

## Layout

```
src/perizia_privacy/   library: corpus, models, training, inference, evaluation helpers
scripts/               one script per experiment stage (run as python -m scripts.<stage>)
results/               vetted aggregate JSON reports + HEADLINE.md
report/                official report template
```
