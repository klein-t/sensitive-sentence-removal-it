# Headline results

**Single source: `results/NUMBERS.md` (generated from `results/numbers.json`), which gives every number with its source file and the command that reproduces it; where this file and NUMBERS.md disagree, NUMBERS.md wins.** The it3m_c and A0 columns below were re-run on 2026-10-30 from the saved checkpoints (`results/canonical-20261029/eval-canonical.json`); the other rows trace to the aggregate JSON files in this folder. Models: **A0** = v1 student (mDeBERTa-v3-base, 4 heads, ratified per-head thresholds salute .0384, penale .40721, minori .02048, famiglia .07239); **it1** = fine-tune of A0 on harder teacher-labelled rows; **Run A** = factored-input fine-tune; **it3** = fine-tune with real-derived counterfactual pairs; **it3m** = it3 plus pairwise margin loss; **it3m_c** = it3m with the penale threshold its saved checkpoint uses, 0.0098325 (see caveats); salute, minori and famiglia keep A0's thresholds.

Gate definitions: a sentence is positive only if it concerns an identifiable living person AND asserts the sensitive fact. Recall intervals Clopper-Pearson (held-out) or Wilson (clean frame, Cassazione); IPW = inverse-probability-weighted harmless deletion on a stratified sample of 2,000 sentences from 200 never-used documents.

## 1. Corpus

Source: `it3-20261009/corpus-diversity.json`, `final-20261003/instrument-summary.json`, `clean-20261008/frame-summary.json`.

| Item | Value | Source |
|---|---|---|
| Source files | 7,464 | lab notebook 2026-09-15 |
| Unique normalized documents | **5,950** | lab notebook 2026-09-15 |
| Near-duplicate groups | **5,845** (largest 16 documents) | lab notebook 2026-09-15 |
| Document split (before any labelling) | train 3,521 · validation 1,230 · test 1,199 (= 5,950; the 100 reserved audit originals, 99 groups, are inside the 1,199 test documents) | lab notebook 2026-09-15, 2026-09-18 |
| Mined windows (distinct, eligible) | train 546,961 · validation 183,387 · test 190,404 | lab notebook 2026-09-15 |
| Corpus integrity | 920,752/920,752 stored windows reconstructed from the 5,950 sources | lab notebook 2026-09-15 |

| | train | validation | test | audit100 | held-out instrument | clean frame |
|---|---:|---:|---:|---:|---:|---:|
| Documents | 3,521 | 1,230 | 1,199 | 100 | 458 | 116 |
| Distinct tribunals · top-10 share | 152 · 22.9% | 135 · 24.8% | 135 · 26.7% | 49 · 41.0% | 109 · 30.6% | 61 · 41.4% |
| Regions | 20 | 20 | 20 | 17 | 20 | 18 |
| Nord-Ovest / Nord-Est / Centro / Sud e Isole / unknown | 25.5/11.0/26.4/34.8/2.2 | 23.2/11.1/27.8/36.3/1.6 | 24.7/11.6/27.8/33.7/2.3 | 24/14/33/27/2 | 27.3/14.0/26.6/29.7/2.4 | 25.0/14.7/34.5/24.1/1.7 |
| Perizia year (extracted): median [IQR] · coverage | 2025 [2024–2026] · 69% | 2025 [2024–2026] · 70% | 2025 [2024–2026] · 70% | 2025 [2023–2026] · 76% | 2025 [2024–2026] · 95% | 2025 [2023–2025] · 91% |
| Attachment PERIZ / STIMA / AVEND / ORDIN | 2333/107/592/489 | 799/37/222/172 | 785/42/207/165 | 68/2/18/12 | 439/19/0/0 | 113/3/0/0 |
| Sentences/doc median [IQR] | 180 [121–286] | 189 [126–277] | 182 [122–292] | 175 [123–290] | 220 [133–368] | 171 [105–284] |
| Non-lexicon word rate | 1.59% | 1.67% | 1.64% | 2.24% | 1.90% | 1.72% |
| Docs with broken hyphenation · per 10k words | 7.2% · 2.6 | 7.6% · 1.9 | 7.6% · 1.2 | 8.0% · 1.5 | 8.3% · 1.7 | 10.3% · 1.2 |
| Garbage-char rate · docs >1% garbage | 0.17% · 0.5% | 0.13% · 0.2% | 0.17% · 0.8% | 0.06% · 0% | 0.06% · 0.2% | 0.43% · 0.9% |
| Near-dup groups · group sizes (size: count) | 3,461 · 1:3431, 2:16, 3:5, 4:5, 5:3, 8:1 | 1,194 · 1:1182, 2:8, 4:1, 5:1, 7:1, 16:1 | 1,190 · 1:1182, 2:7, 3:1 | 99 · 1:98, 2:1 | 458 · all 1 | 116 · all 1 |

Held-out instrument and clean frame columns count documents with at least one evaluated sentence: the instrument drew 538 documents (200 deletion side + 338 positives side), of which 458 contribute (196 + 262), and its 372 positives sit in 116 documents; the clean frame selected 122 documents, of which 116 yield a mined unit.

## 2. Final comparison at the frozen decision rule (2026-10-03)

Source: `final-20261003/eval-final.json`. Rule frozen before Run A: held-out recall >= 98% AND IPW harmless deletion <= 1.5%.

| Model (threshold) | Audit recall | Audit harmless deleted | Audit bleed /239 | Audit leak docs /39 | IPW harmless deleted [95% CI] | Held-out recall [95% CI] | Cue-mined-only recall (n=115) | Held-out bleed /600 | Held-out leak docs /116 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| A0, per-head ratified | 157/158 | 5.64% | 134 | 1 | 7.25% [6.56, 8.05] | 99.7% [98.5, 100] | 114/115 | 410 | 1 |
| A0, global 0.480 | 156/158 | 4.52% | 119 | 1 | 5.16% [4.64, 5.80] | 99.5% [98.1, 99.9] | 113/115 | 346 | 2 |
| it1, global 0.060 | 156/158 | 2.23% | 86 | 2 | 2.29% [1.98, 2.69] | **96.5%** [94.1, 98.1] | 110/115 | 237 | 4 |
| Run A, global 0.018 | 156/158 | 13.94% | 213 | 2 | **12.80%** [10.74, 15.18] | 99.2% [97.7, 99.8] | 112/115 | 580 | 3 |

## 3. Counterfactual training: it3 / it3m (2026-10-10)

Source: `it3-20261009/eval-it3.json`. All at A0's thresholds. Ship rule: held-out >= 99%, audit >= 156/158, held-out leak <= A0, IPW deletion < A0.

| Test | A0 (v1) | it1 | it3 | it3m (margin) |
|---|---|---|---|---|
| Held-out recall /372 | 99.7% | 96.5% | 99.7% | 99.2% |
| IPW harmless deletion | 7.25% | 2.29% | **1.72%** | 2.37% |
| Held-out doc leak | 1/116 | 4/116 | 1/116 | 3/116 |
| audit100 recall /158 · deletion · bleed | 157 · 5.64% · 134 | 155 · 1.61% · 85 | 155 · 1.64% · 86 | 156 · 2.09% · 100 |
| Cassazione recall salute/penale/minori/famiglia | 97.0/95.7/98.5/98.8 | 83.6/78.3/94.1/89.0 | 98.5/93.2/100/97.1 | 97.0/93.2/100/96.5 |
| Cassazione decision leak | 4.2% | 22.5% | 7.2% | 7.8% |
| Cassazione harmless deleted, non-cue · penale cue | 39.3 · 79.1% | 8.4 · n/a | 11.9 · 58.3% | 14.0 · 60.7% |
| Counterfactual twins kept: negated · deceased · company | 0/54 · 0/24 · 10/15 | 0/54 · 0/24 · 11/15 | 3/54 · 0/24 · 11/15 | **39/54 (72%)** · 0/24 · 11/15 |
| Counterfactual targets removed (all types) | 100% | 100% | 100% | 100% |

## 4. Hybrids and it3m_c (2026-10-10)

Source: `it3-20261009/eval-hybrid.json` (A0, it3m, hybrids); it3m_c column from `canonical-20261029/eval-canonical.json` (`it3m_c@saved`, fresh scores of the saved checkpoint).

| Test | A0 | it3m | hybrid | hybrid_b | it3m_c (penale 0.0098325) |
|---|---|---|---|---|---|
| Held-out recall /372 (fam · sal · pen · min) | 371 (368 · 2/2 · 2/2 · 7/7) | 369 (366 · 2/2 · 2/2 · 7/7) | 369 (366 · same) | 369 (366 · same) | **371** (368 · 2/2 · 2/2 · 7/7) |
| IPW harmless deletion | 7.25% | 2.37% | 4.50% | 4.77% | 5.14% [4.56, 5.89] |
| Held-out doc leak (Wilson) | 1/116 [0.2, 4.7] | 3/116 [0.9, 7.3] | 3/116 | 3/116 | 1/116 [0.2, 4.7] |
| audit100 recall /158 · deletion · bleed · doc leak | 157 · 5.64% · 134 · 1/39 | 156 · 2.09% · 100 · 2/39 | 157 · 4.57% · 104 · 1/39 | 157 · 4.65% · 106 · 1/39 | 157 · 5.00% · 108 · 1/39 |
| Cassazione recall sal/pen/min/fam | 97.0/95.7/98.5/98.8 | 97.0/93.2/100/96.5 | 97.0/97.9/100/97.7 | same as hybrid | **100/100/100/98.8** |
| Cassazione decision leak (Wilson) | 13/307 = 4.2% [2.5, 7.1] | 24/307 = 7.8% [5.3, 11.4] | 11/307 = 3.6% [2.0, 6.3] | 11/307 | **2/307 = 0.7% [0.2, 2.3]** |
| Cassazione harmless deleted, non-cue · penale cue | 39.3 · 79.1% | 14.0 · 60.7% | 38.6 · 81.6% | 38.6 · 81.6% | 41.4 · 92.0% |
| Twins kept: negated · deceased · company | 0/54 · 0/24 · 10/15 | 39/54 · 0/24 · 11/15 | 39/54 · 0/24 · 11/15 | 38/54 · 0/24 · 11/15 | 32/54 · 0/24 · 7/15 |

Caveats: (1) Penale threshold. The research point 0.009772 matched A0's penale firing count on a dev set with zero penale positives (a heuristic) and fires on 45/899 = 5.01% of the all-negative validation rows, over the 5% per-label budget. The saved checkpoint re-selects penale on the validation split: the lowest observed threshold whose in-sample rate on the 899 all-negative rows fits a 0.048 budget (43/899), 0.048 being the largest budget on a 0.001 grid whose pooled out-of-fold rate (5 document-grouped folds, 3 seeds) stays within 0.05. That out-of-fold run was ad hoc and unsaved; a reconstruction of its documented design gives 0.0478 at budget 0.048 but 0.0493 at 0.050 (the original runs reported 0.0486 and 0.0508), so the 0.048 choice depends on the fold assignment, and under the reconstructed folds the rule would pick 0.050 (threshold 0.009821, 44/899). All it3m_c numbers here are at the saved 0.0098325; at 0.009772 they were IPW 5.17%, audit100 deletion 5.02%, Cassazione penale-cue 92.6% (recall and leaks identical). (2) it3m_c was selected among five variants after viewing the frozen tests, which inflates its numbers. Leak McNemar (exact, two-sided): held-out it3m 3 vs A0 1 of 116, p = 0.625 (not significant); Cassazione it3m worse than A0 p = 0.0127, it3m_c better p = 0.0034.

## 5. Baselines on the final instrument

Sources: `baselines-ner-20261009/report.json`, `llm-baseline-20261009/report-final.json`. Presidio and GLiNER see the target sentence alone; the local LLM (Gemma 12B instruct, 4-bit, zero-shot with the teacher's decision-tree prompt) is reported raw (its own remove decision) and composed (the policy applied to its factored answers).

| Method | IPW harmless deletion [95% CI] | Held-out recall (372) | Cue-mined recall (115) | Doc leak |
|---|---:|---:|---:|---:|
| A0 (ratified) | 7.25% [6.56%, 8.05%] | 371/372 = 99.7% [98.5, 100.0] | 99.1% | 1/116 |
| Presidio, any entity | 52.87% [48.66%, 56.73%] | 321/372 = 86.3% [82.4, 89.6] | 86.1% | 33/116 |
| Presidio, PERSON only | 14.04% [11.34%, 17.16%] | 165/372 = 44.4% [39.2, 49.6] | 46.1% | 90/116 |
| GLiNER-multi t=0.3 | 0.62% [0.27%, 1.18%] | 132/372 = 35.5% [30.6, 40.6] | 26.1% | 100/116 |
| GLiNER-multi t=0.5 | 0.34% [0.07%, 0.85%] | 72/372 = 19.4% [15.5, 23.7] | 10.4% | 106/116 |
| GLiNER-multi t=0.7 | 0.01% [0.00%, 0.04%] | 28/372 = 7.5% [5.1, 10.7] | 6.1% | 109/116 |
| GLiNER-PII t=0.3 | 7.42% [5.58%, 9.30%] | 150/372 = 40.3% [35.3, 45.5] | 41.7% | 88/116 |
| GLiNER-PII t=0.5 | 2.52% [1.64%, 3.56%] | 84/372 = 22.6% [18.4, 27.2] | 19.1% | 105/116 |
| GLiNER-PII t=0.7 | 0.51% [0.19%, 1.04%] | 41/372 = 11.0% [8.0, 14.7] | 8.7% | 109/116 |

| Method | IPW harmless deletion (600) [95% CI] | Held-out recall (372) | minori (7) | famiglia (369) | Doc leak |
|---|---:|---:|---:|---:|---:|
| A0 (ratified) | 7.27% [6.00, 8.79] | 371/372 = 99.7% [98.5, 100.0] | 7/7 | 368/369 | 1/116 |
| LLM raw | 0.44% [0.20, 0.71] | 13/372 = 3.5% [1.9, 5.9] | 0/7 | 12/369 | 114/116 |
| LLM composed | 0/600 = 0.00% | 234/372 = 62.9% [57.8, 67.8] | 7/7 | 234/369 = 63.4% | 65/116 |

audit100 and cascade: see `baselines-ner-20261009/report.json` and `llm-baseline-20261009/report-cascade.json`.

## 6. Out of domain: Corte di Cassazione decisions

Source: `sentenze-20261008/report.json`, `baselines-ner-20261009/report.json`, `llm-baseline-20261009/report-sentenze.json`. 847 decisions, 1,180 cue-mined units, two blind teacher passes (remove agreement 98.4%, kappa 0.964).

| Method | salute (67) | penale (235) | minori (68) | famiglia (173) | Decision leak (307) |
|---|---:|---:|---:|---:|---:|
| A0 (ratified) | 65/67 = 97.0% [89.8, 99.2] | 225/235 = 95.7% [92.3, 97.7] | 67/68 = 98.5% [92.1, 99.7] | 171/173 = 98.8% [95.9, 99.7] | 4.2% |
| Presidio, any entity | 47/67 = 70.1% [58.3, 79.8] | 170/235 = 72.3% [66.3, 77.7] | 51/68 = 75.0% [63.6, 83.8] | 141/173 = 81.5% [75.1, 86.6] | 25.1% |
| Presidio, PERSON only | 20/67 = 29.9% [20.2, 41.7] | 87/235 = 37.0% [31.1, 43.4] | 20/68 = 29.4% [19.9, 41.1] | 90/173 = 52.0% [44.6, 59.3] | 59.9% |
| GLiNER-multi t=0.3 | 27/67 = 40.3% [29.4, 52.3] | 82/235 = 34.9% [29.1, 41.2] | 59/68 = 86.8% [76.7, 92.9] | 106/173 = 61.3% [53.8, 68.2] | 65.5% |
| GLiNER-multi t=0.5 | 17/67 = 25.4% [16.5, 36.9] | 49/235 = 20.9% [16.1, 26.5] | 45/68 = 66.2% [54.3, 76.3] | 73/173 = 42.2% [35.1, 49.6] | 78.8% |
| GLiNER-multi t=0.7 | 6/67 = 9.0% [4.2, 18.2] | 18/235 = 7.7% [4.9, 11.8] | 20/68 = 29.4% [19.9, 41.1] | 24/173 = 13.9% [9.5, 19.8] | 93.5% |
| GLiNER-PII t=0.3 | 43/67 = 64.2% [52.2, 74.6] | 143/235 = 60.9% [54.5, 66.9] | 64/68 = 94.1% [85.8, 97.7] | 122/173 = 70.5% [63.3, 76.8] | 44.0% |
| GLiNER-PII t=0.5 | 27/67 = 40.3% [29.4, 52.3] | 98/235 = 41.7% [35.6, 48.1] | 62/68 = 91.2% [82.1, 95.9] | 96/173 = 55.5% [48.0, 62.7] | 62.2% |
| GLiNER-PII t=0.7 | 12/67 = 17.9% [10.6, 28.7] | 54/235 = 23.0% [18.1, 28.8] | 50/68 = 73.5% [62.0, 82.6] | 61/173 = 35.3% [28.5, 42.6] | 80.1% |

| Method | non-cue | any cue | salute | penale | minori | famiglia |
|---|---:|---:|---:|---:|---:|---:|
| A0 (ratified) | 39.3% | 61.7% | 60.1% | 79.1% | 69.0% | 56.7% |
| Presidio, any entity | 51.9% | 74.4% | 79.1% | 72.4% | 71.0% | 74.3% |
| Presidio, PERSON only | 23.2% | 40.4% | 46.0% | 36.8% | 35.9% | 39.2% |
| GLiNER-multi t=0.3 | 1.1% | 13.2% | 16.6% | 16.0% | 30.3% | 14.6% |
| GLiNER-multi t=0.5 | 0.0% | 7.7% | 6.7% | 8.0% | 22.1% | 7.6% |
| GLiNER-multi t=0.7 | 0.0% | 2.0% | 1.2% | 2.5% | 6.2% | 2.9% |
| GLiNER-PII t=0.3 | 12.3% | 41.6% | 45.4% | 54.6% | 44.1% | 43.3% |
| GLiNER-PII t=0.5 | 4.2% | 22.7% | 25.2% | 30.7% | 34.5% | 25.1% |
| GLiNER-PII t=0.7 | 1.4% | 9.3% | 9.2% | 11.0% | 20.7% | 8.2% |

## 7. Robustness: real-derived counterfactual pairs (A0)

Source: `clean-20261008/counterfactuals/report.json`. Robustness only, not a recall estimate.

| Edit | Pairs kept (dropped) / source groups | Expected on edited side | A0 behaves as expected | Original side removed | Realism AUC |
|---|---|---|---|---|---:|
| famiglia -> salute | 68 (2) / 27 | remove | 68/68 = 100% [94.7, 100] | 68/68 | 0.986 |
| famiglia -> penale | 66 (2) / 26 | remove | 64/66 = 97.0% [89.6, 99.2] | 66/66 | 0.996 |
| famiglia -> minori | 62 (0) / 25 | remove | 62/62 = 100% [94.2, 100] | 62/62 | 1.000 |
| positive -> **negated** | 54 (17) / 24 | **keep** | **0/54 kept** [0, 6.6] | 54/54 | 0.949 |
| positive -> **deceased** | 24 (0) / 12 | **keep** | **0/24 kept** [0, 13.8] | 24/24 | 1.000 |
| positive -> **company** | 15 (18) / 10 | keep | 10/15 kept = 66.7% [41.7, 84.8] | 15/15 | 1.000 |
| **All** | | | | | **0.915** |

## 8. Phase-0 real-only learning curve

Source: `phase0-20261003/eval-phase0.json`, `phase0-20261003/student-*/manifest.json`.

| Model | Real positives (salute/penale/minori/famiglia) | Audit deletion at 156/158 recall | At it1-matched deletion (1.6%): recall (salute, penale) | Edge pos. removed / twins kept | Probe held-out "keep" accuracy (gate trap) |
|---|---|---|---|---|---|
| A0 (reference) | n/a | 4.52% | 144/158 at 1.61% | 95.6% / 50.1% | n/a |
| it1 (reference) | n/a | 2.23% | 153/158 at 1.58% | 92.7% / 61.2% | n/a |
| factored 25% real | 2/41/45/247 | 61.58% | 121/158 (0/2, 0/2) | 95.3% / 21.0% | 10/109 |
| factored 50% real | 7/66/93/477 | 58.90% | 142/158 (0/2, 0/2) | 94.8% / 22.4% | 10/109 |
| factored 100% real | 19/99/201/1,011 | **81.00%** | 148/158 (0/2, 2/2) | 99.7% / 5.0% | 0/109 |

