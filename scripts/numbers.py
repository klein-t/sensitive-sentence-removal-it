"""Build results/numbers.json and results/NUMBERS.md: one value and one source per number.

Every value is read from an aggregate JSON in ``results/`` (never typed by hand),
paired with the command that regenerates that JSON, and compared with the value
``results/HEADLINE.md`` printed before the canonical re-run (2026-10-30). A
number is marked changed when its leading figures differ from the old value.
The report rows appended on 2026-11-03 also use ``PRIVATE``: a few aggregates with no
``results/`` JSON, typed by hand from the private data root and marked as such.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

R = Path("results")
ENV = (
    "PRIVACY_DATA_ROOT=<private dir> PERIZIA_TEXT_ROOT=<texts> PERIZIA_FACTS_ROOT=<facts> "
    "PYTHONPATH=src:. python -m"
)
FILES = {  # alias: (file under results/, command, status)
    "CE": (
        "canonical-20261029/eval-canonical.json",
        "scripts.canonical_eval score; then scripts.canonical_eval report",
        "re-run 2026-10-30 from the shipped checkpoint (fresh MPS scores)",
    ),
    "NER": (
        "baselines-ner-20261009/report.json",
        "scripts.baselines_ner report",
        "stored predictions; sentence keys and report re-derived 2026-10-30",
    ),
    "LLMF": (
        "llm-baseline-20261009/report-final.json",
        "scripts.baselines_llm final --deletion 600",
        "stored predictions; report re-derived 2026-10-30",
    ),
    "LLMS": (
        "llm-baseline-20261009/report-sentenze.json",
        "scripts.baselines_llm sentenze",
        "stored predictions; report re-derived 2026-10-30",
    ),
    "FIN": ("final-20261003/eval-final.json", "scripts.final_eval", "stored run"),
    "SUM": ("final-20261003/instrument-summary.json", "scripts.final_instrument build", "frozen"),
    "IT3": ("it3-20261009/eval-it3.json", "scripts.it3_eval final", "stored run"),
    "HYB": ("it3-20261009/eval-hybrid.json", "scripts.hybrid_eval", "stored run"),
    "PROBE": ("phase0-20261003/probe-report.json", "scripts.phase0_label", "hosted teacher"),
    "CF": ("clean-20261008/counterfactuals/report.json", "scripts.counterfactuals score", "stored"),
    "GLC": (
        "gliner-calibrated-20261101/report.json",
        "scripts.gliner_calibrated collect; score (gliner venv); report",
        "GLiNER-PII calibrated (validation, A0 targets, 5% budget); fresh MPS scores 2026-11-01",
    ),
}

# id | file alias | json path | value HEADLINE.md printed before the re-run (blank = none)
TABLE = """
A0.heldout_recall | CE | A0@saved/instrument/held_out_recall | 371/372 = 99.7% [98.5, 100.0]
A0.heldout_recall_cue_mined | CE | A0@saved/instrument/held_out_recall_cue_mined_only | 99.1%
A0.heldout_recall_famiglia | CE | A0@saved/instrument/held_out_recall_by_head/famiglia | 368/369
A0.heldout_doc_leak | CE | A0@saved/leak_stats/instrument/wilson | 1/116 [0.2, 4.7]
A0.heldout_bleed | CE | A0@saved/instrument/neighbour_bleed | 410/600
A0.ipw_harmless_deletion | CE | A0@saved/instrument/ipw_harmless_deleted | 7.25% [6.56, 8.05]
A0.audit_recall | CE | A0@saved/audit100/sensitive_removed | 157/158
A0.audit_harmless_deletion | CE | A0@saved/audit100/harmless_deleted | 5.64%
A0.audit_bleed | CE | A0@saved/audit100/bleed | 134/239
A0.audit_doc_leak | CE | A0@saved/audit100/doc_leak | 1/39
A0.cass_recall_salute | CE | A0@saved/sentenze/salute/recall_removed | 65/67 = 97.0% [89.8, 99.2]
A0.cass_recall_penale | CE | A0@saved/sentenze/penale/recall_removed | 225/235 = 95.7% [92.3, 97.7]
A0.cass_recall_minori | CE | A0@saved/sentenze/minori/recall_removed | 67/68 = 98.5% [92.1, 99.7]
A0.cass_recall_famiglia | CE | A0@saved/sentenze/famiglia/recall_removed | 171/173 = 98.8%
A0.cass_decision_leak | CE | A0@saved/sentenze/decision_leak/all_heads | 13/307 = 4.2% [2.5, 7.1]
A0.cass_harmless_noncue | CE | A0@saved/sentenze/harmless_deleted/non_cue | 39.3%
A0.cass_harmless_anycue | CE | A0@saved/sentenze/harmless_deleted/any_cue | 61.7%
A0.cass_harmless_salute_cue | CE | A0@saved/sentenze/harmless_deleted/salute | 60.1%
A0.cass_harmless_penale_cue | CE | A0@saved/sentenze/harmless_deleted/penale | 79.1%
A0.cass_harmless_minori_cue | CE | A0@saved/sentenze/harmless_deleted/minori | 69.0%
A0.cass_harmless_famiglia_cue | CE | A0@saved/sentenze/harmless_deleted/famiglia | 56.7%
A0.cf_fam_to_salute | CE | A0@saved/counterfactuals/fam_to_salute/{T} | 68/68
A0.cf_fam_to_penale | CE | A0@saved/counterfactuals/fam_to_penale/{T} | 64/66
A0.cf_fam_to_minori | CE | A0@saved/counterfactuals/fam_to_minori/{T} | 62/62
A0.cf_negated_twin_kept | CE | A0@saved/counterfactuals/negated/{K} | 0/54
A0.cf_deceased_twin_kept | CE | A0@saved/counterfactuals/deceased/{K} | 0/24
A0.cf_company_twin_kept | CE | A0@saved/counterfactuals/company/{K} | 10/15
A0.clean_recall_minori | CE | A0@saved/clean/minori/recall_removed | 3/3
A0.clean_recall_famiglia | CE | A0@saved/clean/famiglia/recall_removed | 28/28
it3m_c.heldout_recall | CE | it3m_c@saved/instrument/held_out_recall | 371/372
it3m_c.heldout_recall_cue_mined | CE | it3m_c@saved/instrument/held_out_recall_cue_mined_only |
it3m_c.heldout_recall_salute | CE | it3m_c@saved/instrument/held_out_recall_by_head/salute | 2/2
it3m_c.heldout_recall_penale | CE | it3m_c@saved/instrument/held_out_recall_by_head/penale | 2/2
it3m_c.heldout_recall_minori | CE | it3m_c@saved/instrument/held_out_recall_by_head/minori | 7/7
it3m_c.heldout_recall_famiglia | CE | it3m_c@saved/instrument/held_out_recall_by_head/famiglia | 368
it3m_c.heldout_doc_leak | CE | it3m_c@saved/leak_stats/instrument/wilson | 1/116 [0.2, 4.7]
it3m_c.heldout_bleed | CE | it3m_c@saved/instrument/neighbour_bleed |
it3m_c.ipw | CE | it3m_c@saved/instrument/ipw_harmless_deleted | 5.17% [4.59, 5.91]
it3m_c.audit_recall | CE | it3m_c@saved/audit100/sensitive_removed | 157/158
it3m_c.audit_harmless_deletion | CE | it3m_c@saved/audit100/harmless_deleted | 5.02%
it3m_c.audit_bleed | CE | it3m_c@saved/audit100/bleed | 108/239
it3m_c.audit_doc_leak | CE | it3m_c@saved/audit100/doc_leak | 1/39
it3m_c.cass_recall_salute | CE | it3m_c@saved/sentenze/salute/recall_removed | 100
it3m_c.cass_recall_penale | CE | it3m_c@saved/sentenze/penale/recall_removed | 100
it3m_c.cass_recall_minori | CE | it3m_c@saved/sentenze/minori/recall_removed | 100
it3m_c.cass_recall_famiglia | CE | it3m_c@saved/sentenze/famiglia/recall_removed | 98.8
it3m_c.cass_decision_leak | CE | it3m_c@saved/sentenze/decision_leak/all_heads | 2/307 = 0.7%
it3m_c.cass_harmless_noncue | CE | it3m_c@saved/sentenze/harmless_deleted/non_cue | 41.4
it3m_c.cass_harmless_anycue | CE | it3m_c@saved/sentenze/harmless_deleted/any_cue |
it3m_c.cass_harmless_penale_cue | CE | it3m_c@saved/sentenze/harmless_deleted/penale | 92.6%
it3m_c.cf_fam_to_salute | CE | it3m_c@saved/counterfactuals/fam_to_salute/{T} |
it3m_c.cf_fam_to_penale | CE | it3m_c@saved/counterfactuals/fam_to_penale/{T} |
it3m_c.cf_fam_to_minori | CE | it3m_c@saved/counterfactuals/fam_to_minori/{T} |
it3m_c.cf_negated_twin_kept | CE | it3m_c@saved/counterfactuals/negated/{K} | 32/54
it3m_c.cf_deceased_twin_kept | CE | it3m_c@saved/counterfactuals/deceased/{K} | 0/24
it3m_c.cf_company_twin_kept | CE | it3m_c@saved/counterfactuals/company/{K} | 7/15
it3m_c.clean_recall_minori | CE | it3m_c@saved/clean/minori/recall_removed |
it3m_c.clean_recall_famiglia | CE | it3m_c@saved/clean/famiglia/recall_removed |
it3m_c.mcnemar_heldout_vs_A0 | CE | it3m_c@saved/leak_stats/instrument/vs_A0/p_exact |
it3m_c.mcnemar_cass_vs_A0 | CE | it3m_c@saved/leak_stats/sentenze/vs_A0/p_exact | 0.003
research.ipw | CE | it3m_c@0.009772/instrument/ipw_harmless_deleted | 5.17% [4.59, 5.91]
research.audit_deletion | CE | it3m_c@0.009772/audit100/harmless_deleted | 5.02%
research.cass_penale_cue | CE | it3m_c@0.009772/sentenze/harmless_deleted/penale | 92.6%
research.cf_company_twin_kept | CE | it3m_c@0.009772/counterfactuals/company/{K} | 7/15
corpus.source_files | CE | frames/corpus/source_files | 7,464
corpus.documents | CE | frames/corpus/unique_documents | 5,950
corpus.groups | CE | frames/corpus/near_duplicate_groups | 5,845
corpus.largest_group | CE | frames/corpus/largest_group | 16
split.train | CE | frames/corpus/documents_by_split/train | 3,521
split.validation | CE | frames/corpus/documents_by_split/validation | 1,230
split.test_incl_audit | CE | frames/corpus/documents_by_split/test | 1,199
split.audit_inside_test | CE | frames/corpus/audit_documents_by_split/test |
split.audit_groups | CE | frames/corpus/audit_groups | 99
windows.train | CE | frames/corpus/windows/train | 546,961
windows.validation | CE | frames/corpus/windows/validation | 183,387
windows.test | CE | frames/corpus/windows/test | 190,404
instrument.unseen_documents | SUM | unseen_documents |
instrument.docs_drawn | CE | frames/instrument_documents_drawn/total |
instrument.docs_with_units | CE | frames/instrument_documents_with_units/union | 458
instrument.deletion_docs_with_units | CE | frames/instrument_documents_with_units/deletion_sample |
instrument.population_sentences | SUM | population_sentences |
instrument.deletion_units | SUM | deletion_units |
instrument.heldout_positives | CE | frames/held_out_positives | 372
instrument.heldout_positive_docs | CE | frames/held_out_positive_documents | 116
clean.frame_docs | CE | frames/clean_frame_documents |
clean.frame_docs_with_units | CE | frames/clean_frame_documents_with_units | 116
thr.it3m_c | CE | canonical/thresholds | 0.0384, 0.0098, 0.02048, 0.07239
thr.A0 | CE | canonical/A0_thresholds | 0.0384, 0.40721, 0.02048, 0.07239
val.penale_negatives_fired | CE | penale_selection/saved_threshold_negatives_fired |
val.penale_recall | CE | penale_selection/saved_threshold_recall |
val.research_point_negatives_fired | CE | penale_selection/research_threshold_negatives_fired |
val.saved_equals_rule_at_0.048 | CE | penale_selection/saved_equals_in_sample_rule_at_0.048 |
val.oof_0.048 | CE | penale_selection/budget_sweep/0.048/oof_standalone_removal_rate | 0.0486
val.oof_0.050 | CE | penale_selection/budget_sweep/0.050/oof_standalone_removal_rate | 0.0508
ship_rule.it3m_c | CE | ship_rule/0/ship_candidate |
teacher.cass_agreement | CE | cassazione_teacher_agreement/pass_agreement/remove_any | 98.4%
teacher.cass_kappa | CE | cassazione_teacher_agreement/pass_agreement/remove_any_kappa | 0.964
teacher.cass_uncertain | CE | cassazione_teacher_agreement/uncertain | 29
teacher.label_ceiling | PROBE | label_ceiling_A_vs_B/remove_any | 98.0%
teacher.label_ceiling_kappa | PROBE | label_ceiling_A_vs_B/remove_any_kappa | 0.949
teacher.input_ceiling | PROBE | input_ceiling_C_vs_A/remove_any | 96.7%
cf.realism_auc | CF | realism_auc/all | 0.915
it1.ipw | FIN | it1/instrument/ipw_harmless_deleted | 2.29% [1.98, 2.69]
it1.heldout_recall | FIN | it1/instrument/held_out_recall | 96.5% [94.1, 98.1]
it1.heldout_doc_leak | FIN | it1/instrument/held_out_doc_leak | 4/116
it1.audit_recall | FIN | it1/audit100/sensitive_removed | 156/158
runA.ipw | FIN | RunA/instrument/ipw_harmless_deleted | 12.80% [10.74, 15.18]
runA.heldout_recall | FIN | RunA/instrument/held_out_recall | 99.2% [97.7, 99.8]
runA.heldout_doc_leak | FIN | RunA/instrument/held_out_doc_leak | 3/116
A0_global.ipw | FIN | A0/instrument/ipw_harmless_deleted | 5.16% [4.64, 5.80]
A0_global.heldout_recall | FIN | A0/instrument/held_out_recall | 99.5% [98.1, 99.9]
it3.ipw | IT3 | it3@A0_fixed/instrument/ipw_harmless_deleted | 1.72%
it3.heldout_recall | IT3 | it3@A0_fixed/instrument/held_out_recall | 99.7%
it3.heldout_doc_leak | IT3 | it3@A0_fixed/instrument/held_out_doc_leak | 1/116
it3.audit_recall | IT3 | it3@A0_fixed/audit100/sensitive_removed | 155
it3.cass_decision_leak | IT3 | it3@A0_fixed/sentenze/decision_leak/all_heads | 7.2%
it3.cf_negated_twin_kept | IT3 | it3@A0_fixed/counterfactuals/negated/{K} | 3/54
it3m.ipw | HYB | it3m@A0_fixed/instrument/ipw_harmless_deleted | 2.37%
it3m.heldout_recall | HYB | it3m@A0_fixed/instrument/held_out_recall | 99.2%
it3m.heldout_doc_leak | HYB | it3m@A0_fixed/leak_stats/instrument/wilson | 3/116 [0.9, 7.3]
it3m.audit_recall | HYB | it3m@A0_fixed/audit100/sensitive_removed | 156
it3m.cass_decision_leak | HYB | it3m@A0_fixed/sentenze/decision_leak/all_heads | 24/307 = 7.8%
it3m.cf_negated_twin_kept | HYB | it3m@A0_fixed/counterfactuals/negated/{K} | 39/54
it3m.mcnemar_heldout_vs_A0 | HYB | it3m@A0_fixed/leak_stats/instrument/vs_A0/p_exact | 0.63
it3m.mcnemar_cass_vs_A0 | HYB | it3m@A0_fixed/leak_stats/sentenze/vs_A0/p_exact | 0.013
hybrid.ipw | HYB | hybrid@A0_fixed/instrument/ipw_harmless_deleted | 4.50%
hybrid_b.ipw | HYB | hybrid_b@A0_fixed/instrument/ipw_harmless_deleted | 4.77%
hybrid.cass_decision_leak | HYB | hybrid@A0_fixed/sentenze/decision_leak/all_heads | 11/307 = 3.6%
llm_raw.heldout_recall | LLMF | llm_raw/held_out_recall | 13/372 = 3.5% [1.9, 5.9]
llm_raw.ipw_600 | LLMF | llm_raw/ipw_harmless_deleted | 0.44% [0.20, 0.71]
llm_raw.heldout_doc_leak | LLMF | llm_raw/held_out_doc_leak | 114/116
llm_composed.heldout_recall | LLMF | llm_composed/held_out_recall | 234/372 = 62.9% [57.8, 67.8]
llm_composed.ipw_600 | LLMF | llm_composed/ipw_harmless_deleted | 0.00%
llm_composed.heldout_doc_leak | LLMF | llm_composed/held_out_doc_leak | 65/116
llm_composed.cass_decision_leak | LLMS | llm_composed/decision_leak/all_heads | 158/307 = 51.5%
llm.A0_ipw_same_600 | LLMF | a0_production/ipw_harmless_deleted | 7.27% [6.00, 8.79]
gliner_pii_cal.sanity_t0.3_identical | GLC | sanity_en_t0.3/report_identical_to_stored |
gliner_pii_cal.label_set | GLC | label_choice/chosen |
gliner_pii_cal.val_recalled_en | GLC | label_choice/en |
gliner_pii_cal.val_recalled_it | GLC | label_choice/it |
gliner_pii_cal.thresholds | GLC | calibrated/thresholds |
gliner_pii_cal.val_salute | GLC | validation/en/salute/chosen |
gliner_pii_cal.val_salute_feasible | GLC | validation/en/salute/feasible |
gliner_pii_cal.val_penale | GLC | validation/en/penale/chosen |
gliner_pii_cal.val_penale_feasible | GLC | validation/en/penale/feasible |
gliner_pii_cal.val_minori | GLC | validation/en/minori/chosen |
gliner_pii_cal.val_minori_feasible | GLC | validation/en/minori/feasible |
gliner_pii_cal.val_famiglia | GLC | validation/en/famiglia/chosen |
gliner_pii_cal.val_famiglia_feasible | GLC | validation/en/famiglia/feasible |
gliner_pii_cal.val_or_negatives_fired | GLC | validation/en/or_negatives_fired_at_chosen |
gliner_pii_cal.heldout_recall | GLC | calibrated/final/held_out_recall |
gliner_pii_cal.heldout_recall_cue_mined | GLC | calibrated/final/held_out_recall_cue_mined_only |
gliner_pii_cal.heldout_doc_leak | GLC | calibrated/final/held_out_doc_leak |
gliner_pii_cal.ipw | GLC | calibrated/final/ipw_harmless_deleted |
gliner_pii_cal.audit_recall | GLC | calibrated/audit100/sensitive_removed |
gliner_pii_cal.cass_decision_leak | GLC | calibrated/sentenze/decision_leak/all_heads |
gliner_pii_cal.cf_negated_twin_kept | GLC | calibrated/cf/negated/twin_kept |
gliner_pii_cal.cf_negated_original_removed | GLC | calibrated/cf/negated/target_removed |
gliner_pii_cal.curve_0.05.heldout_recall | GLC | global_curve/0.05/held_out_recall |
gliner_pii_cal.curve_0.05.ipw | GLC | global_curve/0.05/ipw_harmless_deleted |
gliner_pii_cal.curve_0.1.heldout_recall | GLC | global_curve/0.1/held_out_recall |
gliner_pii_cal.curve_0.1.ipw | GLC | global_curve/0.1/ipw_harmless_deleted |
gliner_pii_cal.curve_0.2.heldout_recall | GLC | global_curve/0.2/held_out_recall |
gliner_pii_cal.curve_0.2.ipw | GLC | global_curve/0.2/ipw_harmless_deleted |
gliner_pii_cal.curve_0.3.heldout_recall | GLC | global_curve/0.3/held_out_recall |
gliner_pii_cal.curve_0.3.ipw | GLC | global_curve/0.3/ipw_harmless_deleted |
gliner_pii_cal.curve_0.5.heldout_recall | GLC | global_curve/0.5/held_out_recall |
gliner_pii_cal.curve_0.5.ipw | GLC | global_curve/0.5/ipw_harmless_deleted |
"""

# NER baselines: variant | IPW | held-out recall | cue-mined | doc leak | audit100 | Cass. leak
NER_OLD = """
presidio_any | 52.87% | 321/372 | 86.1% | 33/116 | 122/158 | 25.1%
presidio_person | 14.04% | 165/372 | 46.1% | 90/116 | | 59.9%
gliner_multi@0.3 | 0.62% | 132/372 | 26.1% | 100/116 | | 65.5%
gliner_multi@0.5 | 0.34% | 72/372 | 10.4% | 106/116 | | 78.8%
gliner_multi@0.7 | 0.01% | 28/372 | 6.1% | 109/116 | | 93.5%
gliner_pii@0.3 | 7.42% | 150/372 | 41.7% | 88/116 | 61/158 | 44.0%
gliner_pii@0.5 | 2.52% | 84/372 | 19.1% | 105/116 | | 62.2%
gliner_pii@0.7 | 0.51% | 41/372 | 8.7% | 109/116 | | 80.1%
"""
NER_PATHS = (
    ("ipw", "final/ipw_harmless_deleted"),
    ("heldout_recall", "final/held_out_recall"),
    ("heldout_recall_cue_mined", "final/held_out_recall_cue_mined_only"),
    ("heldout_doc_leak", "final/held_out_doc_leak"),
    ("audit_recall", "audit100/sensitive_removed"),
    ("cass_decision_leak", "sentenze/decision_leak/all_heads"),
)
RESOLVED = """
- Clean frame: 122 documents selected (`clean.frame_docs`); 116 of them yield at least one mined
  unit (`clean.frame_docs_with_units`, the diversity-table column). The 116 validation / 6 test
  split of the 122 is a coincidence of numbers.
- Held-out instrument: 538 documents drawn (200 deletion side + 338 positives side); 458
  contribute at least one evaluated sentence (196 + 262; the neighbours sit in positive
  documents). The 372 positives come from 116 documents.
- Splits: train 3,521 + validation 1,230 + test 1,199 = 5,950; the 100 audit documents are
  inside the 1,199 test documents, not in addition to them.
- it3m_c penale: 0.009772 was the research point (dev firing-count match; 45/899 = 5.01% on
  all-negative validation rows, over the 5% budget). The saved checkpoint uses 0.0098325340077281
  (43/899); every it3m_c number here is at the saved value.
- Out-of-fold penale rate: 0.0508 (`it3m-c-oof-evidence.json`) and 0.0486 (checkpoint manifest)
  come from two ad hoc, unsaved runs at different budgets; the reconstruction here gives 0.0493
  at budget 0.050 and 0.0478 at 0.048, so the fold assignment of those runs is not reproducible
  and the 0.048 budget choice depends on it.
"""
ABBREV = {"{T}": "target_removed (edited, new class)", "{K}": "twin_kept (edited harmless)"}


def rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.split("|")] for line in text.strip().splitlines()]


def specs() -> list[tuple[str, str, str, str]]:
    out = []
    for nid, alias, path, old in rows(TABLE):
        for k, v in ABBREV.items():
            path = path.replace(k, v)
        out.append((nid, alias, path, old))
    for variant, *olds in rows(NER_OLD):
        for (k, path), old in zip(NER_PATHS, olds, strict=True):
            out.append((f"{variant}.{k}", "NER", f"{variant}/{path}", old))
    return out


def lookup(data: Any, path: str) -> Any:
    for part in path.split("/"):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


NUM = re.compile(r"\d[\d,]*\.?\d*")


def figures(value: Any) -> list[Decimal]:
    text = value if isinstance(value, str) else json.dumps(value)
    return [Decimal(x.replace(",", "").rstrip(".")) for x in NUM.findall(text)]


def changed(old: str, new: Any) -> bool:
    """True unless every figure of the old value appears, at its own precision, in the new."""
    if not old or isinstance(new, bool):
        return False
    have = figures(new)
    for x in figures(old):
        step = Decimal(1).scaleb(x.as_tuple().exponent)
        if not any(y.quantize(step, ROUND_HALF_UP) == x for y in have):
            return True
    return False


# Report rows (appended 2026-11-03): every number the report states, labelled with the
# report's names and formatted the way the report prints it. `--` is the LaTeX range dash.
FILES_REPORT = {  # aliases used only by the report rows
    "HUM": (
        "human-agreement-20261103/report.json",
        "scripts.human_agreement",
        "one human annotator (the author), 186 stratified items",
    ),
    "SPD": ("speed-20261029/report.json", "scripts.inference_speed", "measured 2026-10-29"),
    "DIV": ("it3-20261009/corpus-diversity.json", "scripts.corpus_diversity", "stored"),
    "P0E": ("phase0-20261003/eval-phase0.json", "scripts.phase0_eval", "stored run"),
    "CASS": ("it3-20261009/cass-summary.json", "scripts.it3_data cass", "stored"),
    "IT1T": ("proof-20261001/model/proof-train.json", "scripts.proof_train", "it1 manifest"),
}
# Values with no machine-readable source in results/: aggregates read by hand from the
# private data root on 2026-11-03 (no text, no identifiers). label | value | provenance
PRIVATE = {
    "A0-train/real rows": (5104, "student-v1-amended-v2/manifest.json :: train/sources/real"),
    "A0-train/synthetic rows": (
        2591,
        "student-v1-amended-v2/manifest.json :: train/sources/synthetic",
    ),
    "CF pairs/total": (741, "it3-20261009/model/it3-train.json :: pairs"),
    "CF pairs/negation": (207, "it3-train.json :: pairs_by_kind/negated"),
    "CF pairs/hypothetical": (145, "it3-train.json :: pairs_by_kind/hypothetical"),
    "CF pairs/deceased subject": (27, "it3-train.json :: pairs_by_kind/deceased"),
    "CF pairs/company subject": (13, "it3-train.json :: pairs_by_kind/company"),
    "CF pairs/category swap": (349, "it3-train.json :: pairs_by_kind/fam_to_* (127+128+94)"),
    "Cass-train/training rows (windows)": (615, "it3-train.json :: cass_rows"),
    "IPW sample/certain harmless": (
        1937,
        "final-20261003/labels-deletion.json :: not uncertain, no head (6 uncertain, 57 pos.)",
    ),
    "Held-out/positives/round 1 cue-mined": (
        85,
        "final-20261003/positive-private.json x labels-positives.json :: round 1, origin cue_*",
    ),
    "Held-out/positives/round 1 score tranche": (
        125,
        "same :: round 1, origin score_tranche (150 candidates)",
    ),
    "Held-out/positives/round 2 score tranche": (
        132,
        "same :: round 2 (extend), origin score_tranche (150 candidates)",
    ),
    "Held-out/positives/round 2 famiglia cue sample": (
        30,
        "same :: round 2 (extend), origin cue_famiglia (150 candidates)",
    ),
    "it3m_c/encoder parameters": (
        278218752,
        "student-it3m-c-20261010/model.safetensors header, excl. pooler and classifier",
    ),
    "it3m_c/all parameters": (
        278813189,
        "student-it3m-c-20261010/model.safetensors header, incl. pooler and 5-way classifier",
    ),
    "GLiNER-PII/parameters": (
        288949504,
        "cached urchade/gliner_multi_pii-v1 pytorch_model.bin, sum of tensor sizes",
    ),
}
# Table 3 (main results): label | held-out | IPW | audit100 | doc leak | Cassazione | twins
T3 = {
    "A0": [
        ("CE", f"A0@saved/{p}")
        for p in (
            "instrument/held_out_recall",
            "instrument/ipw_harmless_deleted",
            "audit100/sensitive_removed",
            "instrument/held_out_doc_leak",
            "sentenze/decision_leak/all_heads",
            "counterfactuals/negated/twin_kept (edited harmless)",
        )
    ],
    "it1": [
        ("FIN", "it1/instrument/held_out_recall"),
        ("FIN", "it1/instrument/ipw_harmless_deleted"),
        ("FIN", "it1/audit100/sensitive_removed"),
        ("FIN", "it1/instrument/held_out_doc_leak"),
        ("IT3", "it1@final_global/sentenze/decision_leak/all_heads"),
        ("IT3", "it1@final_global/counterfactuals/negated/twin_kept (edited harmless)"),
    ],
    "it3": [
        ("IT3", f"it3@A0_fixed/{p}")
        for p in (
            "instrument/held_out_recall",
            "instrument/ipw_harmless_deleted",
            "audit100/sensitive_removed",
            "instrument/held_out_doc_leak",
            "sentenze/decision_leak/all_heads",
            "counterfactuals/negated/twin_kept (edited harmless)",
        )
    ],
    "it3m": [
        ("HYB", f"it3m@A0_fixed/{p}")
        for p in (
            "instrument/held_out_recall",
            "instrument/ipw_harmless_deleted",
            "audit100/sensitive_removed",
            "instrument/held_out_doc_leak",
            "sentenze/decision_leak/all_heads",
            "counterfactuals/negated/twin_kept (edited harmless)",
        )
    ],
    "it3m_c": [
        ("CE", f"it3m_c@saved/{p}")
        for p in (
            "instrument/held_out_recall",
            "instrument/ipw_harmless_deleted",
            "audit100/sensitive_removed",
            "instrument/held_out_doc_leak",
            "sentenze/decision_leak/all_heads",
            "counterfactuals/negated/twin_kept (edited harmless)",
        )
    ],
    "Presidio, any entity": [
        ("NER", f"presidio_any/{p}")
        for p in (
            "final/held_out_recall",
            "final/ipw_harmless_deleted",
            "audit100/sensitive_removed",
            "final/held_out_doc_leak",
            "sentenze/decision_leak/all_heads",
        )
    ],
    "GLiNER-PII": [
        ("GLC", f"calibrated/headline/{p}")
        for p in (
            "held_out_recall",
            "ipw_harmless_deleted",
            "audit100_recall",
            "held_out_doc_leak",
            "cassazione_decision_leak",
        )
    ],
    "Gemma 4 12B, composed": [
        ("LLMF", "llm_composed/held_out_recall"),
        ("LLMF", "llm_composed/ipw_harmless_deleted"),
        None,
        ("LLMF", "llm_composed/held_out_doc_leak"),
        ("LLMS", "llm_composed/decision_leak/all_heads"),
    ],
}
T3_METRICS = (
    "Held-out/{m}/recall (/372)",
    "IPW sample/{m}/harmless deletion",
    "audit100/{m}/recall (/158)",
    "Held-out/{m}/doc leak (/116)",
    "Cassazione/{m}/decision leak (/307)",
    "Twins/{m}/negated kept (/54)",
)


def kn(s: str) -> tuple[int, int]:
    m = re.match(r"\s*(\d+)/(\d+)", s)
    assert m, s
    return int(m[1]), int(m[2])


def pc(k: int | float, n: int | float, d: int = 1) -> str:
    q = Decimal(1).scaleb(-d)
    return str((Decimal(k) * 100 / Decimal(n)).quantize(q, ROUND_HALF_UP))


def cnt(s: str) -> str:
    """'371/372 = ...' or '1/116' -> '371 (99.7%)' (Table 3 cell)."""
    k, n = kn(s)
    return f"{k} ({pc(k, n)}%)"


def frac(s: str) -> str:
    return "{}/{}".format(*kn(s))


def lead_pct(s: str) -> str:
    """First percentage of a value string: '7.25% [..]' -> '7.25%', '371/372 = 99.7%' -> '99.7%'."""
    m = re.search(r"([\d.]+)%", s)
    assert m, s
    return f"{m[1]}%"


def rng(s: str) -> str:
    m = re.search(r"\[([\d.]+)%, ([\d.]+)%\]", s)
    assert m, s
    lo, hi = (x[:-2] if x.endswith(".0") and x.startswith("100") else x for x in m.groups())
    return f"{lo}--{hi}%"


def gemma_speed(ups: list[float]) -> str:
    """Units per second range -> 'sent./s, ms/sent.' as Table 7 prints it."""
    lo, hi = min(ups), max(ups)
    return f"{f3(lo, 2)}--{f3(hi, 1)}, {1 / hi:.1f}k--{1 / lo:.1f}k"


def th(x: int) -> str:
    return f"{x:,}"


def f3(x: float, d: int) -> str:
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP))


def joined(*parts: str) -> str:
    return ", ".join(parts)


def report_specs() -> list[tuple[str, str, Any, list[tuple[str, str] | None]]]:
    """(label, where in the report, formatter over the looked-up values, refs)."""
    specs_: list[tuple[str, str, Any, list[tuple[str, str] | None]]] = []

    def add(label: str, where: str, fn: Any, *refs: tuple[str, str] | None) -> None:
        specs_.append((label, where, fn, list(refs)))

    # Table 3 and the sentences that repeat it (abstract, intro, results, conclusion).
    for m, refs in T3.items():
        fmts = (cnt, lead_pct, cnt, cnt, cnt, cnt)
        for metric, fn, ref in zip(T3_METRICS, fmts, refs + [None] * (6 - len(refs)), strict=True):
            if ref is not None:
                add(metric.format(m=m), "Table 3", lambda v, fn=fn: fn(v[0]), ref)
    for m, alias, p in (("A0", "CE", "A0@saved"), ("it3m_c", "CE", "it3m_c@saved")):
        add(
            f"Held-out/{m}/recall 95%",
            "Table 3 caption",
            lambda v: rng(v[0]),
            (alias, f"{p}/instrument/held_out_recall"),
        )
        add(
            f"IPW sample/{m}/harmless deletion 95%",
            "Table 3 caption",
            lambda v: rng(v[0]),
            (alias, f"{p}/instrument/ipw_harmless_deleted"),
        )
        add(
            f"Cassazione/{m}/decision leak 95%",
            "Table 3 caption",
            lambda v: rng(v[0]),
            (alias, f"{p}/sentenze/decision_leak/all_heads"),
        )
    add(
        "it1/global threshold",
        "Table 3 caption",
        lambda v: f3(v[0][0], 3),
        ("FIN", "it1/thresholds"),
    )
    add(
        "IPW sample/A0/harmless deletion, 600-sentence subsample",
        "Table 3 caption",
        lambda v: lead_pct(v[0]),
        ("LLMF", "a0_production/ipw_harmless_deleted"),
    )
    add(
        "Held-out/Gemma 4 12B, raw/recall",
        "Experiments, Results",
        lambda v: lead_pct(v[0]),
        ("LLMF", "llm_raw/held_out_recall"),
    )
    add(
        "Held-out/A0 vs it3m_c/doc leak McNemar p",
        "Experiments, Results",
        lambda v: f"p={f3(v[0], 1)}",
        ("CE", "it3m_c@saved/leak_stats/instrument/vs_A0/p_exact"),
    )
    add(
        "Cassazione/A0 vs it3m_c/decision leak McNemar p",
        "Experiments, Results",
        lambda v: f"p={v[0]}",
        ("CE", "it3m_c@saved/leak_stats/sentenze/vs_A0/p_exact"),
    )
    add(
        "Held-out/A0 vs it3m/doc leak McNemar p",
        "not in the final report",
        lambda v: f"p={f3(v[0], 2)}",
        ("HYB", "it3m@A0_fixed/leak_stats/instrument/vs_A0/p_exact"),
    )
    # Keyword subset: the 115 held-out positives found by keywords only.
    for m, ref in (
        ("A0", ("CE", "A0@saved/instrument/held_out_recall_cue_mined_only")),
        ("it3m_c", ("CE", "it3m_c@saved/instrument/held_out_recall_cue_mined_only")),
    ):
        add(f"Keyword subset/{m}/recall", "Experiments, Results", lambda v: frac(v[0]), ref)
    add(
        "Keyword subset/Presidio, any entity/recall",
        "Experiments, Results",
        lambda v: lead_pct(v[0]),
        ("NER", "presidio_any/final/held_out_recall_cue_mined_only"),
    )
    add(
        "Keyword subset/GLiNER-PII/recall",
        "Experiments, Results",
        lambda v: lead_pct(v[0]),
        ("GLC", "calibrated/headline/held_out_recall_cue_mined_only"),
    )
    add(
        "Keyword subset/positives",
        "Table 1 caption; App. F",
        lambda v: str(kn(v[0])[1]),
        ("CE", "A0@saved/instrument/held_out_recall_cue_mined_only"),
    )
    add(
        "Held-out/positives from the score tranches",
        "Table 1 caption; App. F (372 - 115)",
        lambda v: str(kn(v[0])[1] - kn(v[1])[1]),
        ("CE", "A0@saved/instrument/held_out_recall"),
        ("CE", "A0@saved/instrument/held_out_recall_cue_mined_only"),
    )
    add(
        "Held-out/score-tranche candidates, round 1",
        "Table 1 caption; App. F",
        lambda v: str(v[0]),
        ("SUM", "positive_candidates/score_tranche"),
    )
    # Corpus, splits, sizes (Data section, Table 1).
    add(
        "Corpus/source files",
        "Data, Corpus and splits",
        lambda v: th(v[0]),
        ("CE", "frames/corpus/source_files"),
    )
    add(
        "Corpus/documents",
        "Intro; Data",
        lambda v: th(v[0]),
        ("CE", "frames/corpus/unique_documents"),
    )
    add(
        "Corpus/near-duplicate groups",
        "Data",
        lambda v: th(v[0]),
        ("CE", "frames/corpus/near_duplicate_groups"),
    )
    add(
        "Corpus/documents train, validation, test",
        "Data",
        lambda v: f"{th(v[0])}, {th(v[1])} and {th(v[2])}",
        ("CE", "frames/corpus/documents_by_split/train"),
        ("CE", "frames/corpus/documents_by_split/validation"),
        ("CE", "frames/corpus/documents_by_split/test"),
    )
    add(
        "Corpus/median sentences per document",
        "Data (train 180, val. 189, test 182)",
        lambda v: f"about {int(v[0])}",
        ("DIV", "splits/train/sentences_per_doc/median"),
        ("DIV", "splits/validation/sentences_per_doc/median"),
        ("DIV", "splits/test/sentences_per_doc/median"),
    )
    add(
        "Corpus/sentences (windows)",
        "Data, Pre-processing (exact 920,752)",
        lambda v: f"about {th(round(sum(v), -4))}",
        ("CE", "frames/corpus/windows/train"),
        ("CE", "frames/corpus/windows/validation"),
        ("CE", "frames/corpus/windows/test"),
    )
    add(
        "audit100/documents",
        "Table 1",
        lambda v: str(v[0]),
        ("CE", "frames/corpus/audit_documents"),
    )
    add(
        "audit100/sentences",
        "Data; Table 1",
        lambda v: th(v[0]),
        ("CE", "fresh_vs_cached_scores/A0/audit100/sentences_fresh"),
    )
    add(
        "audit100/positives",
        "Data; Table 1",
        lambda v: str(kn(v[0])[1]),
        ("CE", "A0@saved/audit100/sensitive_removed"),
    )
    add(
        "audit100/positive rate",
        "Data, Evaluation sets (exact 1 in 236.2)",
        lambda v: f"about one in {round(v[0] / kn(v[1])[1], -1):.0f}",
        ("CE", "fresh_vs_cached_scores/A0/audit100/sentences_fresh"),
        ("CE", "A0@saved/audit100/sensitive_removed"),
    )
    add(
        "Test sets/picked sentences (IPW + Held-out + Twins)",
        "Data, Pre-processing (exact 2,426)",
        lambda v: f"about {th(round(v[0] + v[1] + v[2], -2))}",
        ("SUM", "deletion_units"),
        ("CE", "frames/held_out_positives"),
        ("CF", "pairs_kept/negated"),
    )
    add(
        "Held-out/documents",
        "Table 1",
        lambda v: str(v[0]),
        ("CE", "frames/held_out_positive_documents"),
    )
    add("Held-out/positives", "Table 1", lambda v: str(v[0]), ("CE", "frames/held_out_positives"))
    add(
        "Held-out/positives per head",
        "Table 1 caption",
        lambda v: "famiglia {}, minori {}, salute {}, penale {}".format(*(kn(x)[1] for x in v)),
        *(
            ("CE", f"A0@saved/instrument/held_out_recall_by_head/{h}")
            for h in ("famiglia", "minori", "salute", "penale")
        ),
    )
    add(
        "Held-out/unseen documents (pool)",
        "App. F",
        lambda v: str(v[0]),
        ("SUM", "unseen_documents"),
    )
    add(
        "Held-out/positives-side documents drawn",
        "App. F",
        lambda v: str(v[0]),
        ("SUM", "positives_side_documents"),
    )
    add(
        "IPW sample/documents",
        "Table 1; Metrics",
        lambda v: str(v[0]),
        ("SUM", "deletion_documents"),
    )
    add("IPW sample/sentences", "Table 1; Metrics", lambda v: th(v[0]), ("SUM", "deletion_units"))
    add(
        "IPW sample/population sentences",
        "Table 1 caption; App. E",
        lambda v: th(v[0]),
        ("SUM", "population_sentences"),
    )
    add(
        "IPW sample/band sizes",
        "App. E",
        lambda v: ", ".join(th(b[0]) for b in v[0].values()),
        ("SUM", "bins"),
    )
    add(
        "IPW sample/drawn per band",
        "App. E (285 or 286)",
        lambda v: f"about {max(b[1] for b in v[0].values())}",
        ("SUM", "bins"),
    )
    add(
        "IPW sample/weight of the 36,252 band",
        "App. E (exact 126.8)",
        lambda v: f"about {round(list(v[0].values())[1][0] / list(v[0].values())[1][1])}",
        ("SUM", "bins"),
    )
    add("Cassazione/decisions", "Table 1", lambda v: str(v[0]), ("DIV", "cassazione/decisions"))
    add("Cassazione/sentences", "Table 1", lambda v: th(v[0]), ("CE", "A0@saved/sentenze/units"))
    add(
        "Cassazione/years",
        "Table 1 caption; App. F",
        lambda v: "{} to {}".format(*v[0]),
        ("DIV", "cassazione/year_range"),
    )
    add(
        "Cassazione/decisions with positives",
        "Table 1 caption",
        lambda v: str(kn(v[0])[1]),
        ("CE", "A0@saved/sentenze/decision_leak/all_heads"),
    )
    add(
        "Cassazione/positives per head",
        "Table 1 caption",
        lambda v: "salute {}, penale {}, minori {} and famiglia {}".format(*v),
        *(
            ("CE", f"A0@saved/sentenze/{h}/positives")
            for h in ("salute", "penale", "minori", "famiglia")
        ),
    )
    add(
        "Cassazione/max sentences per decision",
        "App. F",
        lambda v: str(v[0]),
        ("CE", "A0@saved/sentenze/salute/max_per_decision"),
    )
    add(
        "Cassazione/teacher pass agreement",
        "App. F",
        lambda v: f"{lead_pct(v[0])}, kappa={v[1]}",
        ("CE", "cassazione_teacher_agreement/pass_agreement/remove_any"),
        ("CE", "cassazione_teacher_agreement/pass_agreement/remove_any_kappa"),
    )
    add(
        "Twins/negated pairs, source groups",
        "Table 1",
        lambda v: f"{v[0]} pairs, {v[1]} groups",
        ("CF", "pairs_kept/negated"),
        ("CF", "pair_source_groups/negated"),
    )
    add(
        "Twins/deceased, company, category swap",
        "Table 1 caption",
        lambda v: f"{v[0]} deceased, {v[1]} company-subject and {v[2] + v[3] + v[4]} category-swap",
        ("CF", "pairs_kept/deceased"),
        ("CF", "pairs_kept/company"),
        ("CF", "pairs_kept/fam_to_salute"),
        ("CF", "pairs_kept/fam_to_penale"),
        ("CF", "pairs_kept/fam_to_minori"),
    )
    add(
        "Twins/realism AUC",
        "not in the final report",
        lambda v: str(v[0]),
        ("CF", "realism_auc/all"),
    )
    # Training sets.
    add("A0-train/rows", "Abstract; Data; Experiments", lambda v: th(v[0]), ("IT1T", "replay_rows"))
    add(
        "A0-train/synthetic share",
        "Data, Labels (private counts 2,591 / 7,695)",
        lambda v: f"{pc(PRIVATE['A0-train/synthetic rows'][0], v[0])}%",
        ("IT1T", "replay_rows"),
    )
    add("Teacher rows/rows", "Experiments", lambda v: th(v[0]), ("IT1T", "sol_rows"))
    add("Edge suite/rows", "Experiments", lambda v: th(v[0]), ("IT1T", "edge_rows"))
    add("Cass-train/sentences", "Experiments", lambda v: str(v[0]), ("CASS", "units"))
    add(
        "Training total (A0-train + Teacher rows + Edge suite + 2 x CF pairs + Cass-train)",
        "Data, Pre-processing (exact 11,966; CF pairs 741 is private)",
        lambda v: (
            f"about {th(round(v[0] + v[1] + v[2] + 2 * PRIVATE['CF pairs/total'][0] + v[3], -3))}"
        ),
        ("IT1T", "replay_rows"),
        ("IT1T", "sol_rows"),
        ("IT1T", "edge_rows"),
        ("CASS", "units"),
    )
    add(
        "it1/updates, learning rate",
        "Experiments",
        lambda v: f"{v[0]}, {v[1]}",
        ("IT1T", "updates"),
        ("IT1T", "lr"),
    )
    add(
        "Probe/two-pass agreement",
        "Data, Labels",
        lambda v: f"{frac(v[0])} ({lead_pct(v[0])}), kappa={v[1]}",
        ("PROBE", "label_ceiling_A_vs_B/remove_any"),
        ("PROBE", "label_ceiling_A_vs_B/remove_any_kappa"),
    )
    # Thresholds and validation (Experiments, App. B).
    add(
        "A0/thresholds (3 decimals)",
        "Experiments, Threshold calibration",
        lambda v: "({})".format(", ".join(f3(x, 3) for x in v[0])),
        ("CE", "canonical/A0_thresholds"),
    )
    add(
        "A0/thresholds",
        "App. B",
        lambda v: ", ".join(f3(x, 5).rstrip("0") for x in v[0]),
        ("CE", "canonical/A0_thresholds"),
    )
    add(
        "it3m_c/penale threshold",
        "Experiments; App. B",
        lambda v: f"{f3(v[0], 4)} ({f3(v[0], 7)})",
        ("CE", "penale_selection/saved_threshold"),
    )
    add(
        "Validation/all-negative rows",
        "Experiments; App. B",
        lambda v: str(v[0]),
        ("CE", "penale_selection/all_negative_rows"),
    )
    add(
        "Validation/it3m_c penale fired, recall",
        "Experiments, Threshold calibration",
        lambda v: f"{v[0]} rows, {v[1]}",
        ("CE", "penale_selection/saved_threshold_negatives_fired"),
        ("CE", "penale_selection/saved_threshold_recall"),
    )
    add(
        "Validation/firing budget",
        "Experiments, Threshold calibration; App. C",
        lambda v: f"{pc(v[0], 1, 0)}%",
        ("GLC", "config/negative_firing_budget"),
    )
    add(
        "Validation/it3m_c penale budget",
        "Experiments, Threshold calibration",
        lambda v: "4.8%" if v[0] == v[1] and v[2] else "rule does not give the saved value",
        ("CE", "penale_selection/budget_sweep/0.048/in_sample_threshold"),
        ("CE", "penale_selection/saved_threshold"),
        ("CE", "penale_selection/saved_equals_in_sample_rule_at_0.048"),
    )
    add(
        "Validation/recall targets",
        "Experiments, Threshold calibration",
        lambda v: "salute {}, penale {}, minori {}, famiglia {}".format(*v[0].values()),
        ("GLC", "config/recall_targets"),
    )
    for m in ("A0", "it3m_c"):
        for h in ("salute", "penale", "minori", "famiglia"):
            add(
                f"Validation/{m}/{h} fired, recall",
                "App. B, Table 6",
                lambda v: f"{kn(v[0])[0]}, {v[1]}",
                ("CE", f"validation/{m}/{h}/negatives_fired"),
                ("CE", f"validation/{m}/{h}/recall"),
            )
    # Appendix C.
    for m, p in (("A0", "A0@saved"), ("it3m_c", "it3m_c@saved")):
        add(
            f"Held-out/{m}/recall salute, penale, minori",
            "App. C",
            lambda v: ", ".join(frac(x) for x in v),
            *(
                ("CE", f"{p}/instrument/held_out_recall_by_head/{h}")
                for h in ("salute", "penale", "minori")
            ),
        )
        add(
            f"Cassazione/{m}/recall per head",
            "App. C",
            lambda v: ", ".join(frac(x) for x in v),
            *(
                ("CE", f"{p}/sentenze/{h}/recall_removed")
                for h in ("salute", "penale", "minori", "famiglia")
            ),
        )
    add(
        "GLiNER-PII/thresholds",
        "App. C",
        lambda v: ", ".join(f3(x, 3) for x in v[0]),
        ("GLC", "calibrated/thresholds"),
    )
    add(
        "Validation/GLiNER-PII/recall per head",
        "App. C; Discussion",
        lambda v: ", ".join(v),
        *(
            ("GLC", f"validation/en/{h}/chosen/recall")
            for h in ("salute", "penale", "minori", "famiglia")
        ),
    )
    add(
        "Validation/GLiNER-PII/labels recalled, English vs Italian",
        "App. C",
        lambda v: f"{v[0].replace('/', ' of ')}, Italian {kn(v[1])[0]}",
        ("GLC", "label_choice/en"),
        ("GLC", "label_choice/it"),
    )
    add(
        "Held-out/GLiNER-PII/global t 0.05, 0.1, 0.2, 0.3, 0.5 recall",
        "Results; App. C",
        lambda v: "/".join(lead_pct(x)[:-1] for x in v) + "%",
        *(
            ("GLC", f"global_curve/{t}/held_out_recall")
            for t in ("0.05", "0.1", "0.2", "0.3", "0.5")
        ),
    )
    add(
        "IPW sample/GLiNER-PII/global t 0.05, 0.1, 0.2, 0.3, 0.5 deletion",
        "Results; App. C",
        lambda v: "/".join(lead_pct(x)[:-1] for x in v) + "%",
        *(
            ("GLC", f"global_curve/{t}/ipw_harmless_deleted")
            for t in ("0.05", "0.1", "0.2", "0.3", "0.5")
        ),
    )
    add(
        "Held-out/GLiNER-PII/minori, salute recall",
        "Discussion",
        lambda v: f"{frac(v[0])}, {frac(v[1])}",
        ("GLC", "calibrated/final/held_out_recall_by_head/minori"),
        ("GLC", "calibrated/final/held_out_recall_by_head/salute"),
    )
    add(
        "A0 global threshold/threshold, Held-out recall, IPW deletion",
        "App. C",
        lambda v: f"{f3(v[0][0], 3)}, {frac(v[1])}, {lead_pct(v[2])}",
        ("FIN", "A0/thresholds"),
        ("FIN", "A0/instrument/held_out_recall"),
        ("FIN", "A0/instrument/ipw_harmless_deleted"),
    )
    add(
        "hybrid, hybrid_b/IPW deletion, held-out doc leak",
        "App. C",
        lambda v: f"{lead_pct(v[0])} and {lead_pct(v[1])}, {v[2]} and {v[3]}",
        ("HYB", "hybrid@A0_fixed/instrument/ipw_harmless_deleted"),
        ("HYB", "hybrid_b@A0_fixed/instrument/ipw_harmless_deleted"),
        ("HYB", "hybrid@A0_fixed/instrument/held_out_doc_leak"),
        ("HYB", "hybrid_b@A0_fixed/instrument/held_out_doc_leak"),
    )
    # Discussion.
    for m, p in (("A0", "A0@saved"), ("it3m_c", "it3m_c@saved")):
        add(
            f"Clean frame/{m}/harmless deleted with salute, minori, penale cues vs none",
            "Discussion",
            lambda v: ", ".join(lead_pct(x) for x in v[:3]) + f" vs {lead_pct(v[3])}",
            *(
                ("CE", f"{p}/clean/harmless_cue_deleted/{h}")
                for h in ("salute", "minori", "penale", "non_cue")
            ),
        )
        add(
            f"Cassazione/{m}/harmless deleted, penale cue and no cue",
            "Discussion, Error analysis",
            lambda v: f"{lead_pct(v[0])}, {lead_pct(v[1])}",
            ("CE", f"{p}/sentenze/harmless_deleted/penale"),
            ("CE", f"{p}/sentenze/harmless_deleted/non_cue"),
        )
        add(
            f"Held-out/{m}/harmless neighbours deleted",
            "Discussion, Error analysis",
            lambda v: v[0],
            ("CE", f"{p}/instrument/neighbour_bleed"),
        )
        add(
            f"Cassazione/{m}/decision leak outside famiglia",
            "Discussion, Error analysis",
            lambda v: frac(v[0]),
            ("CE", f"{p}/sentenze/decision_leak/salute_penale_minori"),
        )
    add(
        "Twins/A0/negated originals deleted",
        "Intro; Discussion; Table 4",
        lambda v: frac(v[0]),
        ("CE", "A0@saved/counterfactuals/negated/target_removed (original positive)"),
    )
    for m, ref in (
        ("A0", ("CE", "A0@saved")),
        ("it1", ("IT3", "it1@final_global")),
        ("it3", ("IT3", "it3@A0_fixed")),
        ("it3m", ("HYB", "it3m@A0_fixed")),
        ("it3m_c", ("CE", "it3m_c@saved")),
    ):
        add(
            f"Twins/{m}/deceased kept",
            "Intro; Discussion; Table 4",
            lambda v: frac(v[0]),
            (ref[0], f"{ref[1]}/counterfactuals/deceased/twin_kept (edited harmless)"),
        )
    add(
        "Phase 0 real-only students/audit deletion at 156/158",
        "Discussion (25%, 50%, 100% real)",
        lambda v: "{} to {}%".format(
            *(f(float(x.split()[-1][:-1]) for x in v) for f in (min, max))
        ),
        *(("P0E", f"student-{k}/audit100/harmless_deleted") for k in (25, 50, 100)),
    )
    # Human check (Limitations, Table 5).
    add(
        "Human check/items",
        "Abstract; Limitations; Conclusion",
        lambda v: str(v[0]),
        ("HUM", "answered_items"),
    )
    add(
        "Human check/identical-set items",
        "Table 5; Limitations (1)",
        lambda v: str(v[0]),
        ("HUM", "comparisons/sol_reference_items/n_items"),
    )
    for name, key in (
        ("Human vs teacher", "human_vs_sol"),
        ("Human vs A0's local annotators", "human_vs_small"),
        ("Teacher vs A0's local annotators", "small_vs_sol"),
    ):
        add(
            f"Human check/119/{name}/kappa, exact match",
            "Table 5; Limitations (1)",
            lambda v: f"{f3(v[0], 2)}, {pc(v[1], 1)}%",
            ("HUM", f"comparisons/sol_reference_items/{key}/pooled_item_x_head/kappa"),
            ("HUM", f"comparisons/sol_reference_items/{key}/exact_set_agreement"),
        )
    add(
        "Human check/v1 items",
        "Table 5; Limitations (1)",
        lambda v: str(v[0]),
        ("HUM", "comparisons/v1_items/n_items"),
    )
    add(
        "Human check/49/kappa, human vs teacher and vs A0's local annotators",
        "Table 5; Limitations (1)",
        lambda v: f"{f3(v[0], 2)} vs {f3(v[1], 2)}",
        ("HUM", "comparisons/v1_items/human_vs_sol/pooled_item_x_head/kappa"),
        ("HUM", "comparisons/v1_items/human_vs_small/pooled_item_x_head/kappa"),
    )
    add(
        "Human check/49/exact match, human vs teacher and vs A0's local annotators",
        "Table 5; Limitations (1)",
        lambda v: f"{pc(v[0], 1)}% vs {pc(v[1], 1)}%",
        ("HUM", "comparisons/v1_items/human_vs_sol/exact_set_agreement"),
        ("HUM", "comparisons/v1_items/human_vs_small/exact_set_agreement"),
    )
    add(
        "Human check/186/salute positives, LLM reference vs human",
        "not in the final report",
        lambda v: f"{v[0]} vs {v[1]}",
        ("HUM", "reproduce_existing/all_answered/per_head/salute/teacher_positives"),
        ("HUM", "reproduce_existing/all_answered/per_head/salute/human_positives"),
    )
    add(
        "Human check/186/human-positive sentences",
        "not in the final report",
        lambda v: str(v[0]["tp"] + v[0]["fn"]),
        ("HUM", "reproduce_existing/all_answered/sentence_any_category/confusion"),
    )
    # Speed and size (App. D, Table 7).
    add(
        "Speed/documents, sentences",
        "App. D; Table 7",
        lambda v: f"{v[0]} documents ({th(v[1])} sentences)",
        ("SPD", "documents/n"),
        ("SPD", "documents/sentences_total"),
    )
    for name, key, d in (
        ("it3m_c", "it3m_c", 1),
        ("GLiNER-PII", "gliner_pii@0.3", 1),
        ("Presidio", "presidio_any", 1),
    ):
        add(
            f"Speed/{name}/sentences per second, ms per sentence",
            "App. D; Table 7",
            lambda v, d=d: f"{f3(v[0], d)}, {round(1000 / v[0])}",
            ("SPD", f"methods/{key}/median_of_passes/sentences_per_s"),
        )
    add(
        "Speed/it3m_c/seconds for a median document",
        "App. D",
        lambda v: f"{f3(v[0], 1)}",
        ("SPD", "methods/it3m_c/median_of_passes/s_per_doc_median"),
    )
    add(
        "Speed/Gemma 12B/sentences per second, ms per sentence",
        "Table 7",
        lambda v: gemma_speed(list(v[0].values())),
        ("SPD", "gemma_12b_reference/units_per_second_by_instrument"),
    )
    add(
        "Speed/Gemma 12B/times slower than it3m_c",
        "Results; App. D; Conclusion",
        lambda v: "{} to {}".format(*v[0]),
        ("SPD", "gemma_12b_reference/slowdown_vs_it3m_c_range"),
    )
    add("Speed/hardware", "Approach; App. D", lambda v: v[0], ("SPD", "hardware"))
    add(
        "Speed/Presidio version",
        "Approach, Baselines",
        lambda v: ".".join(v[0].split(".")[:2]),
        ("SPD", "versions/presidio-analyzer"),
    )
    for name, key, where in (
        ("it3m_c", "it3m_c/all parameters", "Table 7; Conclusion"),
        ("GLiNER-PII", "GLiNER-PII/parameters", "Table 7"),
    ):
        add(
            f"Size/{name}/parameters",
            f"{where} (exact {th(PRIVATE[key][0])})",
            lambda v, key=key: f"{round(PRIVATE[key][0] / 1e6)}M",
        )
    add(
        "Size/mDeBERTa-v3-base encoder/parameters",
        "Background; Approach (exact 278,218,752)",
        lambda v: f"{round(PRIVATE['it3m_c/encoder parameters'][0] / 1e6)}M",
    )
    return specs_


def report_rows(load: Any) -> list[dict[str, Any]]:
    out = []
    files = FILES | FILES_REPORT
    for label, where, fn, refs in report_specs():
        vals = [lookup(load(files[a][0]), p) for a, p in refs]
        out.append(
            {
                "id": label,
                "value": fn(vals),
                "raw": vals,
                "where": where,
                "source": "; ".join(f"results/{files[a][0]}" for a, _ in refs)
                or "private aggregate (table below)",
                "path": "; ".join(p for _, p in refs),
                "command": "; ".join(sorted({f"{ENV} {files[a][1]}" for a, _ in refs})),
                "status": "report row",
                "headline_before": None,
                "changed": False,
            }
        )
    for label, (value, where) in PRIVATE.items():
        out.append(
            {
                "id": label,
                "value": th(value),
                "raw": [value],
                "where": "private aggregate",
                "source": f"$PRIVACY_DATA_ROOT/{where.split(' :: ')[0]}",
                "path": where,
                "command": "read by hand 2026-11-03 (no results/ JSON)",
                "status": "typed private aggregate",
                "headline_before": None,
                "changed": False,
            }
        )
    return out


def report_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Report numbers (labels and formatting as printed in the report)",
        "",
        "Appended 2026-11-03. One row per number the report states, labelled with the report's "
        "names and formatted as the report prints it (`--` is the LaTeX range dash; 'about' "
        "rows give the exact value in the location column). Extra sources:",
        "",
        "| source | command | status |",
        "|---|---|---|",
    ]
    lines += [f"| `{f}` | `{c}` | {s} |" for f, c, s in FILES_REPORT.values()]
    lines += ["", "| label | report value | where | source :: path |", "|---|---|---|---|"]
    for r in rows:
        if r["status"] != "report row":
            continue
        src = "; ".join(s.removeprefix("results/") for s in r["source"].split("; "))
        lines.append(f"| {r['id']} | {r['value']} | {r['where']} | `{src}` :: `{r['path']}` |")
    lines += [
        "",
        "## Not in results/: private aggregates typed by hand",
        "",
        "No aggregate JSON in `results/` carries these values; each was read on 2026-11-03 from "
        "the private data root (`$PRIVACY_DATA_ROOT`), counts only. The 372 held-out positives by "
        "origin: 85 + 30 = 115 in the keyword subset, 125 + 132 = 257 from the two "
        "score tranches of 150 candidates each.",
        "",
        "| label | value | provenance |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['value']} | `{r['path']}` |" for r in rows if r["status"] != "report row"
    ]
    return lines


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cache: dict[str, Any] = {}
    out = []
    for nid, alias, path, old in specs():
        file, command, status = FILES[alias]
        data = cache.setdefault(file, json.loads((R / file).read_text()))
        value = lookup(data, path)
        out.append(
            {
                "id": nid,
                "value": value,
                "source": f"results/{file}",
                "path": path,
                "command": f"{ENV} {command}",
                "status": status,
                "headline_before": old or None,
                "changed": changed(old, value),
            }
        )
    extra = report_rows(lambda f: cache.setdefault(f, json.loads((R / f).read_text())))
    (R / "numbers.json").write_text(json.dumps(out + extra, indent=1, ensure_ascii=False) + "\n")
    lines = [
        "# Numbers: one value, one source",
        "",
        "Generated by `python -m scripts.numbers`; do not edit by hand. `old` is the value "
        "`HEADLINE.md` printed before the canonical re-run of 2026-10-30; **CHANGED** marks a "
        "number whose figures differ. Commands are run from the repository root, prefixed "
        f"with `{ENV}`.",
        "",
        "## Definitions that resolve earlier conflicts",
        RESOLVED,
        "| source | command | status |",
        "|---|---|---|",
    ]
    lines += [f"| `{f}` | `{c}` | {s} |" for f, c, s in FILES.values()]
    lines += ["", "| id | value | old (HEADLINE) | source :: path |", "|---|---|---|---|"]
    for r in out:
        value = r["value"] if isinstance(r["value"], str) else json.dumps(r["value"])
        old = r["headline_before"] or ""
        if r["changed"]:
            old = f"**CHANGED** {old}"
        lines.append(f"| {r['id']} | {value} | {old} | `{r['source'][8:]}` :: `{r['path']}` |")
    lines += report_lines(extra)
    (R / "NUMBERS.md").write_text("\n".join(lines) + "\n")
    print(sum(r["changed"] for r in out), "changed of", len(out))
    print(len(extra), "report rows appended")


if __name__ == "__main__":
    main()
