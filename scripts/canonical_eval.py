"""Canonical re-run: every reported number from one configuration, scored fresh.

Canonical configuration: the shipped it3m_c checkpoint as saved
(``student-it3m-c-20261010``: weights, tokenizer and its ``thresholds.json``,
penale 0.0098325), A0 = ``student-v1-amended-v2`` at its own saved thresholds,
and the frozen test sets exactly as stored (final instrument, audit100, clean
frame, Cassazione units, counterfactual pairs, real validation rows).

* ``score``: both models re-score every test sentence on MPS, ignoring every
  cached research score; per-sentence scores go to the private directory
  ``$PRIVACY_DATA_ROOT/canonical-20261029/scores``.
* ``report``: the frozen tests' own metric functions at each model's saved
  thresholds (and it3m_c at the research point 0.009772, for the old -> new
  diff), paired leak tests, the penale threshold re-selection on validation,
  a comparison with the cached research scores, the frame document counts, and
  the baselines recomputed from their stored predictions. Writes aggregates
  only, to ``results/canonical-20261029/eval-canonical.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts import baselines_llm, baselines_ner, hybrid_eval, it3_eval
from scripts import proof_score as ps
from scripts.clean_mining import C, wilson
from scripts.clean_mining import summarise as clean_summary
from scripts.final_eval import F, instrument_keys
from scripts.final_instrument import score_v1
from scripts.proof_eval import audit_frames
from scripts.proof_score import HEADS, OUT, ROOT, V1, load_model, private_dir, score_texts

K = ROOT / "canonical-20261029"
SCORES = K / "scores"
CANON = ROOT / "student-it3m-c-20261010"
MODELS = {"A0": V1, "it3m_c": CANON}
VALIDATION = ROOT / "development-preparation-v3/training-inputs-v3/validation.jsonl"
RESEARCH_PENALE = 0.009772  # hybrid-penale-threshold.json, the dev firing-count match
REPORT = Path("results/canonical-20261029/eval-canonical.json")
BUILDERS = {
    "instrument": it3_eval.instrument_requests,
    "clean": it3_eval.clean_requests,
    "sentenze": it3_eval.sentenze_requests,
    "counterfactuals": it3_eval.cf_requests,
}
TESTS = (*BUILDERS, "audit100")
PENALE = HEADS.index("penale")
Scores = dict[tuple[Any, ...], list[float]]


def saved_thresholds(model: str) -> list[float]:
    return json.loads((MODELS[model] / "thresholds.json").read_text())["thresholds"][:4]


def save(path: Path, sc: Scores) -> None:
    ps.write_json(path, [[list(k), v] for k, v in sc.items()])


def read(path: Path) -> Scores:
    return {tuple(k): v for k, v in json.loads(path.read_text())}


def validation_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in VALIDATION.read_text().split("\n") if line]


# --- score -------------------------------------------------------------------------


def score(model: str) -> None:
    private_dir(SCORES)
    it3_eval.MODELS[model] = ("v1", MODELS[model])
    for test, build in BUILDERS.items():
        path = SCORES / f"{model}-{test}.json"
        if not path.exists():
            save(path, it3_eval.score_ranges(model, *build()))
            print(model, test, "scored", flush=True)
    path = SCORES / f"{model}-audit100.json"
    if not path.exists():
        items = [
            ((f["source_doc_id"], f["sentence_start"], f["sentence_end"]), f["text"])
            for f in audit_frames()
        ]
        save(path, score_v1({model: MODELS[model]}, items)[model])
        print(model, "audit100 scored", flush=True)
    path = SCORES / f"{model}-validation.json"
    if not path.exists():
        rows = validation_rows()
        torch, tok, net = load_model(MODELS[model])
        s = score_texts(torch, tok, net, [r["text"] for r in rows])
        ps.write_json(path, {r["id"]: [float(x) for x in v] for r, v in zip(rows, s, strict=True)})
        print(model, "validation scored", flush=True)


# --- report ------------------------------------------------------------------------


def scores(model: str, test: str) -> Scores:
    sc = read(SCORES / f"{model}-{test}.json")
    if test == "instrument":
        for k in {k for part in instrument_keys().values() for k in part}:
            sc.setdefault(k, [0.0] * 4)
    return sc


def operating_point(model: str, thr: list[float]) -> dict[str, Any]:
    row: dict[str, Any] = {"thresholds": thr}
    for test in TESTS:
        sc = scores(model, test)
        if test == "clean":
            private = json.loads((C / "units-private.json").read_text())
            for p in private:
                sc.setdefault((p["source_doc_id"], p["ss"], p["se"]), [0.0] * 4)
            labels = json.loads((C / "labels.json").read_text())
            row[test] = clean_summary(private, labels, sc, thr)
        else:
            row[test] = hybrid_eval.report(test, sc, thr)
    return row


def leaks(model: str, thr: list[float]) -> dict[str, dict[str, bool]]:
    return {
        "instrument": hybrid_eval.instrument_leaks(scores(model, "instrument"), thr),
        "audit100": hybrid_eval.audit_leaks(scores(model, "audit100"), thr),
        "sentenze": hybrid_eval.sentenze_leaks(scores(model, "sentenze"), thr),
    }


def lowest_within(s: np.ndarray, negative: np.ndarray, budget: float) -> float:
    """Lowest observed score whose standalone firing rate on the negatives is <= budget."""
    negs = np.sort(s[negative])
    rate = (len(negs) - np.searchsorted(negs, np.unique(s), side="left")) / len(negs)
    return float(np.unique(s)[np.argmax(rate <= budget)])


def penale_selection(model: str) -> dict[str, Any]:
    """Amendment-v2 penale rule on validation: budget sweep, in sample and out of fold.

    Out of fold: 5 folds grouped by doc_id, seeds 1134/20260916/7 (the amendment-v2
    design); the threshold is re-selected on each fold's training rows and applied
    to the held-out rows. The original it3m_c out-of-fold run was ad hoc and unsaved;
    this is a reconstruction of its documented design.
    """
    rows = validation_rows()
    sv = json.loads((SCORES / f"{model}-validation.json").read_text())
    s = np.array([sv[r["id"]][PENALE] for r in rows])
    negative = np.array([not r["labels"] for r in rows])
    positive = np.array(["penale" in r["labels"] for r in rows])
    groups = np.array([r["doc_id"] for r in rows])
    unique = np.unique(groups)
    out: dict[str, Any] = {
        "all_negative_rows": int(negative.sum()),
        "penale_positives": int(positive.sum()),
    }
    sweep = {}
    for budget in np.round(np.arange(0.050, 0.0449, -0.001), 3):
        rates, recalls = [], []
        for seed in (1134, 20260916, 7):
            order = np.random.default_rng(seed).permutation(len(unique))
            fold = dict(zip(unique[order], np.arange(len(unique)) % 5, strict=True))
            member = np.array([fold[g] for g in groups])
            fired = np.zeros(len(s), dtype=bool)
            for f in range(5):
                held = member == f
                fired[held] = s[held] >= lowest_within(s[~held], negative[~held], budget)
            rates.append(float(fired[negative].mean()))
            recalls.append(float(fired[positive].mean()))
        t = lowest_within(s, negative, budget)
        sweep[f"{budget:.3f}"] = {
            "in_sample_threshold": t,
            "in_sample_negatives_fired": int((s[negative] >= t).sum()),
            "oof_standalone_removal_rate": float(np.mean(rates)),
            "oof_rate_per_seed": rates,
            "oof_recall": float(np.mean(recalls)),
        }
    chosen = next(b for b, v in sweep.items() if v["oof_standalone_removal_rate"] <= 0.05)
    out["budget_sweep"] = sweep
    out["rule_choice_this_reconstruction"] = {"budget": chosen, **sweep[chosen]}
    saved = saved_thresholds(model)[PENALE]
    out["saved_threshold"] = saved
    out["saved_threshold_negatives_fired"] = int((s[negative] >= saved).sum())
    out["saved_threshold_recall"] = f"{int((s[positive] >= saved).sum())}/{int(positive.sum())}"
    out["research_threshold_negatives_fired"] = int((s[negative] >= RESEARCH_PENALE).sum())
    out["saved_equals_in_sample_rule_at_0.048"] = sweep["0.048"]["in_sample_threshold"] == saved
    return out


def validation_gate(model: str) -> dict[str, Any]:
    rows = validation_rows()
    sv = json.loads((SCORES / f"{model}-validation.json").read_text())
    s = np.array([sv[r["id"]][:4] for r in rows])
    negative = np.array([not r["labels"] for r in rows])
    out = {}
    for i, h in enumerate(HEADS):
        t = saved_thresholds(model)[i]
        pos = np.array([h in r["labels"] for r in rows])
        out[h] = {
            "negatives_fired": f"{int((s[negative, i] >= t).sum())}/{int(negative.sum())}",
            "recall": f"{int((s[pos, i] >= t).sum())}/{int(pos.sum())}",
        }
    return out


def cached_research(model: str, test: str) -> Scores | None:
    """The research-time per-sentence scores, for the fresh-vs-cached comparison."""
    if model == "it3m_c":
        return read(it3_eval.I3 / "scores" / f"it3m-{test}.json")
    if test == "instrument":
        return {tuple(k): v for k, v in json.loads((F / "eval-scores.json").read_text())["A0"]}
    if test == "audit100":
        cur: Scores = {}
        for f, row in zip(audit_frames(), np.load(OUT / "eval" / "A0-audit.npy"), strict=True):
            key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
            cur[key] = [
                max(a, float(b)) for a, b in zip(cur.get(key, [0.0] * 4), row[:4], strict=True)
            ]
        return cur
    return read(it3_eval.I3 / "scores" / f"A0-{test}.json")


def compare_cached(model: str, thr: list[float]) -> dict[str, Any]:
    out = {}
    for test in TESTS:
        fresh, old = read(SCORES / f"{model}-{test}.json"), cached_research(model, test)
        common = fresh.keys() & old.keys()
        diff = max(abs(a - b) for k in common for a, b in zip(fresh[k], old[k], strict=True))
        flips = sum(
            hybrid_eval.removed(fresh[k], thr) != hybrid_eval.removed(old[k], thr) for k in common
        )
        out[test] = {
            "sentences_fresh": len(fresh),
            "sentences_cached": len(old),
            "common": len(common),
            "max_abs_score_diff": diff,
            "decision_flips_at_saved_thresholds": flips,
        }
    return out


def frames() -> dict[str, Any]:
    def docs(path: Path, key: str = "source_doc_id") -> set[str]:
        return {p[key] for p in json.loads(path.read_text())}

    summary = json.loads((F / "instrument-summary.json").read_text())
    dele, pos, nb = (docs(F / f"{p}-private.json") for p in ("deletion", "positive", "neighbour"))
    lp = json.loads((F / "labels-positives.json").read_text())
    certain = [
        p
        for p in json.loads((F / "positive-private.json").read_text())
        if not lp[p["unit_id"]]["uncertain"] and lp[p["unit_id"]]["remove_heads"]
    ]
    frame = json.loads((C / "frame-private.json").read_text())
    units = docs(C / "units-private.json")
    corpus = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    manifest = json.loads((ROOT / "corpus-v1/manifest.json").read_text())
    group_sizes = Counter(d["doc_id"] for d in corpus)
    return {
        "corpus": {
            "source_files": manifest["source_files"],
            "unique_documents": len(corpus),
            "near_duplicate_groups": len(group_sizes),
            "largest_group": max(group_sizes.values()),
            "documents_by_split": dict(Counter(d["split"] for d in corpus)),
            "audit_documents_by_split": dict(Counter(d["split"] for d in corpus if d["audit"])),
            "audit_documents": sum(d["audit"] for d in corpus),
            "audit_groups": len({d["doc_id"] for d in corpus if d["audit"]}),
            "windows": {
                s: manifest["counts"][f"windows_{s}"] for s in ("train", "validation", "test")
            },
        },
        "instrument_documents_drawn": {
            "deletion": summary["deletion_documents"],
            "positives_side": summary["positives_side_documents"],
            "total": summary["deletion_documents"] + summary["positives_side_documents"],
        },
        "instrument_documents_with_units": {
            "deletion_sample": len(dele),
            "positive_candidates": len(pos),
            "neighbours": len(nb),
            "union": len(dele | pos | nb),
        },
        "held_out_positives": len(certain),
        "held_out_positive_documents": len({p["source_doc_id"] for p in certain}),
        "clean_frame_documents": len(frame),
        "clean_frame_documents_with_units": len(units & set(frame)),
    }


def baselines() -> dict[str, Any]:
    """Stored baseline predictions: same canonical sentences, and the reports re-derive."""
    out: dict[str, Any] = {}
    stored_keys = json.loads((baselines_ner.B / "keys-private.json").read_text())
    stored_report = json.loads((baselines_ner.B / "report.json").read_text())
    original_b, original_write = baselines_ner.B, ps.write_json
    redirected: dict[str, Any] = {}
    try:
        baselines_ner.B = private_dir(K / "ner-collect")
        baselines_ner.collect()  # fresh sentence keys and text ids from the frozen sets
        fresh_keys = json.loads((baselines_ner.B / "keys-private.json").read_text())
        baselines_ner.B = original_b
        ps.write_json = lambda path, value: redirected.__setitem__(Path(path).name, value)
        baselines_ner.report()
    finally:
        baselines_ner.B, ps.write_json = original_b, original_write
    out["ner_sentence_keys_identical"] = {
        k: sorted(map(tuple, fresh_keys[k])) == sorted(map(tuple, stored_keys[k]))
        for k in stored_keys
    }
    out["ner_report_identical"] = redirected["report.json"] == stored_report
    llm: dict[str, Any] = {}
    for step, call in (
        ("final", lambda: baselines_llm.final(600)),
        ("sentenze", baselines_llm.sentenze),
    ):
        stored = json.loads((baselines_llm.L / f"report-{step}.json").read_text())
        fresh = json.loads(json.dumps(call()))
        llm[step] = all(stored[k] == v for k, v in fresh.items() if k != "throughput")
    out["llm_report_identical_from_cached_predictions"] = llm
    return out


def diversity() -> dict[str, Any]:
    """Re-run the corpus statistics; the tribunal/region fields need the absent DB export."""
    from scripts import corpus_diversity as cd

    stored = json.loads((it3_eval.I3 / "corpus-diversity.json").read_text())
    captured: dict[str, Any] = {}
    original = cd.write_json
    cd.write_json = lambda path, value: captured.__setitem__("report", value)
    try:
        cd.main()
    finally:
        cd.write_json = original
    fresh = captured["report"]
    fields = (
        "documents",
        "attachment_type",
        "perizia_year_extracted",
        "sentences_per_doc",
        "ocr_proxies",
        "near_duplicate_groups",
        "group_size_distribution",
    )
    return {
        "metadata_export_available": fresh["db_available"],
        "identical_without_metadata": {
            split: {f: fresh["splits"][split][f] == v[f] for f in fields}
            for split, v in stored["splits"].items()
        },
        "cassazione_identical": fresh["cassazione"] == stored["cassazione"],
    }


def cassazione_agreement() -> dict[str, Any]:
    """Two blind teacher passes on the Cassazione units, recomputed from the stored passes."""
    from scripts.phase0_label import agreement
    from scripts.sentenze import S

    passes = json.loads((S / "labels-passes.json").read_text())
    both = sorted(set(passes["L1"]) & set(passes["L2"]))
    labels = json.loads((S / "labels.json").read_text())
    return {
        "labelled_both": len(both),
        "uncertain": sum(v["uncertain"] for v in labels.values()),
        "pass_agreement": agreement(
            {i: passes["L1"][i] for i in both}, {i: passes["L2"][i] for i in both}
        ),
    }


def report() -> None:
    a0, c = saved_thresholds("A0"), saved_thresholds("it3m_c")
    research = c[:PENALE] + [RESEARCH_PENALE] + c[PENALE + 1 :]
    points = {
        "A0@saved": ("A0", a0),
        "it3m_c@saved": ("it3m_c", c),
        "it3m_c@0.009772": ("it3m_c", research),
    }
    out: dict[str, Any] = {
        "canonical": {"checkpoint": CANON.name, "thresholds": c, "A0": V1.name, "A0_thresholds": a0}
    }
    leak = {}
    for name, (model, thr) in points.items():
        out[name] = operating_point(model, thr)
        leak[name] = leaks(model, thr)
        print(name, "done", flush=True)
    for name in points:
        out[name]["leak_stats"] = {
            part: {
                "wilson": wilson(sum(v.values()), len(v)),
                "vs_A0": hybrid_eval.mcnemar(leak["A0@saved"][part], v),
            }
            for part, v in leak[name].items()
        }
    out["ship_rule"] = [
        it3_eval.verdict({"A0@production": out["A0@saved"], n: out[n]}, n)
        for n in points
        if n != "A0@saved"
    ]
    out["validation"] = {m: validation_gate(m) for m in MODELS}
    out["penale_selection"] = penale_selection("it3m_c")
    out["fresh_vs_cached_scores"] = {m: compare_cached(m, saved_thresholds(m)) for m in MODELS}
    out["frames"] = frames()
    out["baselines"] = baselines()
    out["corpus_diversity"] = diversity()
    out["cassazione_teacher_agreement"] = cassazione_agreement()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(it3_eval.strip(out), indent=1, default=str) + "\n")
    print(REPORT.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("score", "report"))
    args = ap.parse_args()
    if args.step == "score":
        for model in MODELS:
            score(model)
    else:
        report()


if __name__ == "__main__":
    main()
