"""Build results/numbers.json and results/NUMBERS.md: one value and one source per number.

Every value is read from an aggregate JSON in ``results/`` (never typed by hand),
paired with the command that regenerates that JSON, and compared with the value
``results/HEADLINE.md`` printed before the canonical re-run (2026-10-30). A
number is marked changed when its leading figures differ from the old value.
"""

from __future__ import annotations

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


def main() -> None:
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
    (R / "numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
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
    (R / "NUMBERS.md").write_text("\n".join(lines) + "\n")
    print(sum(r["changed"] for r in out), "changed of", len(out))


if __name__ == "__main__":
    main()
