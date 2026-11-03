"""GLiNER-PII baseline calibrated on validation the way A0 was, then re-tested.

Same input as ``baselines_ner``: the masked target sentence alone, chunked at
200 words, per-head score = max span score of that head's label over the
chunks. What changes: the label phrasing (English set of ``baselines_ner`` or
one Italian set) and the per-head thresholds are chosen on the real validation
rows only, never on a test instrument.

* ``collect`` (project env): the validation targets, deduplicated, to a private
  file; the test sentences are the ones ``baselines_ner collect`` already wrote.
* ``score --labels en|it --part validation|test`` (gliner venv): scores with an
  inference floor of ``FLOOR`` (spans below it are not returned, score 0.0),
  appended to a private JSONL so an interrupted run resumes.
* ``report`` (project env):
  1. sanity: English labels at t=0.3 from the fresh scores must reproduce the
     stored ``gliner_pii@0.3`` report;
  2. label set: higher validation recall (head positives recalled, summed over
     the four heads) at the maximum-recall threshold within the 5% budget;
  3. thresholds per head, A0's rule: candidates are the realisable tied-score
     boundaries; feasible = observed recall >= target and standalone firing on
     the all-negative validation rows <= 5%; among feasible, maximum precision,
     then recall, then threshold (``evaluation.select_validation_thresholds``);
     if none is feasible, maximum recall within the budget, then precision, then
     threshold, recorded as infeasible;
  4. the chosen configuration on every instrument with the instruments' own
     metric functions, plus a global-threshold curve.
Aggregates only go to ``results/gliner-calibrated-20261101``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.baselines_ner import (
    GLINER_MODELS,
    HEADS,
    ROOT,
    append,
    chunks,
    keyed,
    sid,
    strip,
)
from scripts.baselines_ner import B as TEST_DIR

G = ROOT / "gliner-calibrated-20261101"
VALIDATION = ROOT / "development-preparation-v3/training-inputs-v3/validation.jsonl"
OUT = Path("results/gliner-calibrated-20261101")
MODEL = "gliner_pii"
LABEL_SETS = {
    "en": {
        "salute": "health condition / disease / disability of a person",
        "penale": "criminal offence, conviction, criminal proceeding",
        "minori": "minor child",
        "famiglia": "family relationship, marriage, separation, divorce",
    },
    "it": {
        "salute": "condizione di salute, malattia o disabilità di una persona",
        "penale": "reato, condanna o procedimento penale",
        "minori": "minore, figlio minorenne",
        "famiglia": "relazione familiare, matrimonio, separazione o divorzio",
    },
}
FLOOR = 0.001  # inference threshold: every calibrated threshold must lie above it
TARGETS = {"salute": 0.95, "penale": 0.85, "minori": 0.98, "famiglia": 0.98}
BUDGET = 0.05
CURVE = (0.05, 0.1, 0.2, 0.3, 0.5)


def scores_path(labels: str) -> Path:
    return G / f"scores-{MODEL}-{labels}.jsonl"


# --- collect (project env) ---------------------------------------------------


def validation_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in VALIDATION.read_text().split("\n") if line]


def collect() -> None:
    from perizia_privacy.training import target_text
    from scripts.phase0_eval import doc_text
    from scripts.proof_score import private_dir, write_json

    private_dir(G)
    text_of, rows, mismatch = {}, [], 0
    for r in validation_rows():
        text = doc_text(r["source_doc_id"])[r["start"] : r["end"]]
        mismatch += text.strip() != target_text(r["text"]).strip()
        i = sid(text)
        text_of[i] = text
        rows.append([i, r["labels"]])
    write_json(G / "validation-private.json", {"text_of": text_of, "rows": rows})
    print(json.dumps({"rows": len(rows), "sentences": len(text_of), "target_mismatch": mismatch}))


# --- score (gliner venv) -----------------------------------------------------


def score(labels: str, part: str, batch: int = 64) -> None:
    import torch
    from gliner import GLiNER

    if part == "validation":
        text_of = json.loads((G / "validation-private.json").read_text())["text_of"]
    else:
        text_of = json.loads((TEST_DIR / "sentences-private.json").read_text())
    path = scores_path(labels)
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
    todo = {i: t for i, t in sorted(text_of.items(), key=lambda x: len(x[1])) if i not in done}
    print(json.dumps({"labels": labels, "part": part, "done": len(done), "todo": len(todo)}))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = GLiNER.from_pretrained(GLINER_MODELS[MODEL]).to(device)
    net.eval()
    names = [LABEL_SETS[labels][h] for h in HEADS]
    ids = list(todo)
    for start in range(0, len(ids), batch):
        part_ids = ids[start : start + batch]
        flat = [(i, c) for i in part_ids for c in chunks(todo[i])]
        with torch.no_grad():
            preds = net.inference(
                [c for _, c in flat],
                names,
                threshold=FLOOR,
                flat_ner=False,
                multi_label=True,
                batch_size=16,
            )
        best = {i: [0.0] * len(HEADS) for i in part_ids}
        for (i, _), spans in zip(flat, preds, strict=True):
            for span in spans:
                h = names.index(span["label"])
                best[i][h] = max(best[i][h], float(span["score"]))
        append(path, [{"id": i, "s": best[i]} for i in part_ids])
        print(json.dumps({"scored": start + len(part_ids), "of": len(ids)}), flush=True)


# --- report (project env) ----------------------------------------------------


def load(labels: str) -> dict[str, list[float]]:
    out = {}
    for line in scores_path(labels).read_text().splitlines():
        if line:
            r = json.loads(line)
            out[r["id"]] = r["s"]
    return out


def candidates(s: list[float], pos: list[bool], neg: list[bool]) -> list[dict[str, Any]]:
    """Every realisable ``>= t`` boundary, as in ``select_validation_thresholds``."""
    order = sorted(range(len(s)), key=lambda i: -s[i])
    support, negatives = sum(pos), sum(neg)
    out, tp, fp, fired = [], 0, 0, 0
    for rank, i in enumerate(order):
        tp, fp, fired = tp + pos[i], fp + (not pos[i]), fired + neg[i]
        if rank + 1 < len(order) and s[order[rank + 1]] == s[i]:
            continue
        out.append(
            {
                "threshold": s[i],
                "recall": tp / support,
                "precision": tp / (tp + fp),
                "tp": tp,
                "fired": fired,
                "fire_rate": fired / negatives,
            }
        )
    return out


def select(s: list[float], pos: list[bool], neg: list[bool], target: float) -> dict[str, Any]:
    cands = candidates(s, pos, neg)
    within = [c for c in cands if c["fire_rate"] <= BUDGET and c["threshold"] > 0]
    feasible = [c for c in within if c["recall"] >= target]
    best = max(within, key=lambda c: (c["recall"], c["precision"], c["threshold"]))
    if feasible:
        chosen = max(feasible, key=lambda c: (c["precision"], c["recall"], c["threshold"]))
    else:
        chosen = best
    # The floor binds when even the lowest returned score stays within the budget.
    floor_binding = min(c["threshold"] for c in cands if c["threshold"] > 0) == best["threshold"]

    def view(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "threshold": c["threshold"],
            "recall": f"{c['tp']}/{sum(pos)}",
            "negatives_fired": f"{c['fired']}/{sum(neg)}",
            "precision": round(c["precision"], 4),
        }

    return {
        "target": target,
        "feasible": bool(feasible),
        "chosen": view(chosen),
        "max_recall_within_budget": view(best),
        "floor_binding": floor_binding,
        "_thr": chosen["threshold"],
        "_best_thr": best["threshold"],
        "_best_tp": best["tp"],
    }


def validation(labels: str) -> dict[str, Any]:
    v = json.loads((G / "validation-private.json").read_text())
    sc = load(labels)
    neg = [not lab for _, lab in v["rows"]]
    out: dict[str, Any] = {}
    for k, h in enumerate(HEADS):
        s = [sc[i][k] for i, _ in v["rows"]]
        out[h] = select(s, [h in lab for _, lab in v["rows"]], neg, TARGETS[h])
    out["head_positives_recalled_within_budget"] = sum(out[h]["_best_tp"] for h in HEADS)
    out["head_positives"] = sum(h in lab for _, lab in v["rows"] for h in HEADS)
    fired = [
        any(sc[i][k] >= out[h]["_thr"] for k, h in enumerate(HEADS))
        for i, lab in v["rows"]
        if not lab
    ]
    out["or_negatives_fired_at_chosen"] = f"{sum(fired)}/{len(fired)}"
    return out


def evaluate(scores: dict[str, list[float]], thr: list[float]) -> dict[str, Any]:
    """The body of ``baselines_ner.report`` for one operating point."""
    from scripts.baselines_ner import cf_metrics
    from scripts.clean_mining import summarise as clean_summary
    from scripts.final_eval import audit_metrics, instrument_metrics
    from scripts.sentenze import S
    from scripts.sentenze import summarise as sentenze_summary

    clean_units = json.loads((ROOT / "clean-20261008/units-private.json").read_text())
    clean_labels = json.loads((ROOT / "clean-20261008/labels.json").read_text())
    sent_units = json.loads((S / "units-private.json").read_text())
    sent_labels = json.loads((S / "labels.json").read_text())
    res: dict[str, Any] = {"thresholds": thr}
    res["final"] = instrument_metrics(keyed("final", scores, -1), thr)
    res["clean"] = clean_summary(clean_units, clean_labels, keyed("clean", scores, -1), thr)
    res["cf"] = cf_metrics(keyed("cf", scores, -1), thr)
    res["sentenze"] = sentenze_summary(sent_units, sent_labels, keyed("sentenze", scores, -1), thr)
    res["audit100"] = audit_metrics(keyed("audit100", scores, -1), thr)
    return res


def headline(res: dict[str, Any]) -> dict[str, Any]:
    f, cf = res["final"], res["cf"]["negated"]
    return {
        "held_out_recall": f["held_out_recall"],
        "held_out_recall_cue_mined_only": f["held_out_recall_cue_mined_only"],
        "held_out_doc_leak": f["held_out_doc_leak"],
        "ipw_harmless_deleted": f["ipw_harmless_deleted"],
        "audit100_recall": res["audit100"]["sensitive_removed"],
        "cassazione_decision_leak": res["sentenze"]["decision_leak"]["all_heads"],
        "negated_twin_kept": cf["twin_kept"],
        "negated_original_removed": cf["target_removed"],
    }


def check(value: Any, path: str = "") -> None:
    """Data rules: short strings only, no codici fiscali, no identifiers."""
    if isinstance(value, dict):
        for k, v in value.items():
            check(k, path)
            check(v, f"{path}/{k}")
    elif isinstance(value, list):
        for v in value:
            check(v, path)
    elif isinstance(value, str):
        bad = (
            len(value) > 120
            or re.search(r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]", value)
            or re.search(r"[0-9a-f]{16,}", value)
        )
        if bad:
            sys.exit(f"aggregate check failed at {path}")


def report() -> None:
    stored = json.loads((TEST_DIR / "report.json").read_text())[f"{MODEL}@0.3"]
    en = load("en")
    sanity = strip(evaluate(en, [0.3] * len(HEADS)))
    sanity.pop("thresholds")
    expected = {k: v for k, v in stored.items() if k != "threshold"}
    old = {}
    for line in (TEST_DIR / f"scores-{MODEL}.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            old[r["id"]] = r["s"]
    clipped = {i: [x if x >= 0.3 else 0.0 for x in en[i]] for i in old}
    out: dict[str, Any] = {
        "config": {
            "model": GLINER_MODELS[MODEL],
            "input": "masked target sentence alone, 200-word chunks, max span score per head",
            "inference_floor": FLOOR,
            "recall_targets": TARGETS,
            "negative_firing_budget": BUDGET,
            "label_sets": LABEL_SETS,
        },
        "sanity_en_t0.3": {
            "report_identical_to_stored": sanity == expected,
            "max_abs_score_diff_vs_stored": max(
                abs(a - b) for i in old for a, b in zip(clipped[i], old[i], strict=True)
            ),
            "decision_flips_at_0.3": sum(
                (a >= 0.3) != (b >= 0.3) for i in old for a, b in zip(en[i], old[i], strict=True)
            ),
            **headline(sanity),
        },
    }
    if out["sanity_en_t0.3"]["report_identical_to_stored"] is not True:
        print(json.dumps(out, indent=1))
        sys.exit("sanity reproduction failed")
    val = {labels: validation(labels) for labels in LABEL_SETS}
    choice = max(
        LABEL_SETS, key=lambda k: (val[k]["head_positives_recalled_within_budget"], k == "en")
    )
    out["validation"] = val
    out["label_choice"] = {
        "chosen": choice,
        "criterion": "head positives recalled within the 5% budget, summed over heads",
        **{
            k: f"{val[k]['head_positives_recalled_within_budget']}/{val[k]['head_positives']}"
            for k in LABEL_SETS
        },
    }
    scores = en if choice == "en" else load(choice)
    thr = [val[choice][h]["_thr"] for h in HEADS]
    best = [val[choice][h]["_best_thr"] for h in HEADS]
    calibrated = evaluate(scores, thr)
    out["calibrated"] = {"headline": headline(calibrated), **calibrated}
    if best != thr:
        alt = evaluate(scores, best)
        out["max_recall_within_budget"] = {"headline": headline(alt), **alt}
    out["global_curve"] = {}
    from scripts.final_eval import instrument_metrics

    for t in CURVE:
        m = instrument_metrics(keyed("final", scores, -1), [t] * len(HEADS))
        out["global_curve"][str(t)] = {
            "held_out_recall": m["held_out_recall"],
            "ipw_harmless_deleted": m["ipw_harmless_deleted"],
        }
    out = strip(out)
    check(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "score", "report"))
    ap.add_argument("--labels", choices=tuple(LABEL_SETS), default="en")
    ap.add_argument("--part", choices=("validation", "test"), default="validation")
    args = ap.parse_args()
    if args.step == "collect":
        collect()
    elif args.step == "score":
        score(args.labels, args.part)
    else:
        report()


if __name__ == "__main__":
    main()
