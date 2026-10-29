"""Final step: A0, it1 and Run A on the frozen instrument, then the rule.

Operating point per model: one global threshold on the max remove score, the
highest that keeps audit100 recall >= 156/158 (A0 also at its production
per-head thresholds). At that point: audit100 deletion, bleed and document
leak; IPW harmless deletion on the deletion sample; bleed on the neighbours of
held-out positives; held-out positive recall per head (Clopper-Pearson 95%)
and the held-out document leak. The decision rule is applied as ratified.
Prints aggregates only.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from perizia_privacy import factored as fs
from perizia_privacy.corpus import sentence_spans
from perizia_privacy.evaluation import clopper_pearson
from scripts.final_instrument import IT1, score_v1, sentence_windows
from scripts.phase0_eval import (
    audit_labels,
    doc_text,
    sentence_scores,
    threshold_for,
)
from scripts.proof_eval import audit_frames
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    V1,
    private_dir,
    v1_thresholds,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

F = ROOT / "final-20261003"
RUN_A = F / "run-a"
Key = tuple[str, int, int]
Scores = dict[Key, list[float]]


def instrument_keys() -> dict[str, list[Key]]:
    out = {}
    for part in ("deletion", "positive", "neighbour"):
        rows = json.loads((F / f"{part}-private.json").read_text())
        out[part] = [(p["source_doc_id"], p["ss"], p["se"]) for p in rows]
    return out


def v1_instrument(name: str, path: Path, keys: dict[str, list[Key]]) -> Scores:
    """A0/it1 per-head sentence scores on every instrument sentence."""
    out: Scores = {}
    pop = json.loads((F / "population-scores.json").read_text())
    col = 1 if name == "A0" else 2
    for row in pop:
        out[tuple(row[0])] = row[col]
    wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for part in ("positive", "neighbour"):
        for src, s0, s1 in keys[part]:
            wanted[src].add((s0, s1))
    items = [
        ((src, *span), e)
        for src, spans in sorted(wanted.items())
        for span, examples in sentence_windows(doc_text(src)).items()
        if span in spans
        for e in examples
    ]
    out |= score_v1({name: path}, items)[name]
    return out


def factored_instrument(keys: dict[str, list[Key]]) -> Scores:
    torch, tok, model = fs.load(RUN_A, "mps")
    wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for part in keys.values():
        for src, s0, s1 in part:
            wanted[src].add((s0, s1))
    out: Scores = {}
    for src, spans_wanted in sorted(wanted.items()):
        text = doc_text(src)
        spans = sentence_spans(text)
        targets = [t for t in fs.document_targets(text, tok) if spans[t[2]] in spans_wanted]
        for (_, _, i), row in zip(
            targets, fs.score_targets(torch, tok, model, text, targets, "mps", 32), strict=True
        ):
            key = (src, *spans[i])
            prev = out.get(key, [0.0] * 4)
            out[key] = [max(a, b) for a, b in zip(prev, row[fs.REMOVE], strict=True)]
    return out


def audit_scores() -> dict[str, Scores]:
    out: dict[str, Scores] = {}
    for name, cache in (("A0", "A0"), ("it1", "model")):
        s = np.load(OUT / "eval" / f"{cache}-audit.npy")
        cur: Scores = {}
        for f, row in zip(audit_frames(), s, strict=True):
            key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
            prev = cur.get(key, [0.0] * 4)
            cur[key] = [max(a, float(b)) for a, b in zip(prev, row[:4], strict=True)]
        out[name] = cur
    parts = sentence_scores(RUN_A, "final-run-a")
    out["RunA"] = {k: list(v[fs.REMOVE]) for k, v in parts["audit"].items()}
    return out


def removed(s: list[float] | None, thr: list[float]) -> bool:
    return s is not None and any(x >= t for x, t in zip(s, thr, strict=True))


def ci(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    if lo is None or hi is None:
        return f"{k}/{n}"
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def audit_metrics(sc: Scores, thr: list[float]) -> dict[str, Any]:
    sens = hit = harm = harm_hit = adj = adj_hit = docs = leaked = 0
    for doc, rows in audit_labels().items():
        pos = [i for i, r in enumerate(rows) if set(r["labels"]) & set(HEADS)]
        leak = False
        for i, r in enumerate(rows):
            h = removed(sc.get((doc, r["start"], r["end"])), thr)
            if set(r["labels"]) & set(HEADS):
                sens, hit, leak = sens + 1, hit + h, leak or not h
            elif not r["labels"]:
                harm, harm_hit = harm + 1, harm_hit + h
                if any(abs(i - j) == 1 for j in pos):
                    adj, adj_hit = adj + 1, adj_hit + h
        if pos:
            docs, leaked = docs + 1, leaked + leak
    return {
        "sensitive_removed": f"{hit}/{sens}",
        "harmless_deleted": f"{harm_hit / harm:.2%}",
        "bleed": f"{adj_hit}/{adj}",
        "doc_leak": f"{leaked}/{docs}",
        "_recall_hits": hit,
    }


def instrument_metrics(sc: Scores, thr: list[float]) -> dict[str, Any]:
    dl = json.loads((F / "labels-deletion.json").read_text())
    dp = json.loads((F / "deletion-private.json").read_text())
    harmless = [
        (float(p["weight"]), removed(sc[(p["source_doc_id"], p["ss"], p["se"])], thr))
        for p in dp
        if not dl[p["unit_id"]]["uncertain"] and not dl[p["unit_id"]]["remove_heads"]
    ]
    w = np.array([x for x, _ in harmless])
    r = np.array([y for _, y in harmless], dtype=float)
    rate = float((w * r).sum() / w.sum())
    rng = random.Random(1134)
    boot = []
    for _ in range(1000):
        idx = [rng.randrange(len(w)) for _ in range(len(w))]
        boot.append(float((w[idx] * r[idx]).sum() / w[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    lp = json.loads((F / "labels-positives.json").read_text())
    pp = json.loads((F / "positive-private.json").read_text())
    per_head = {h: [0, 0] for h in HEADS}
    total = [0, 0]
    cue = [0, 0]
    docs: dict[str, bool] = {}
    for p in pp:
        lab = lp[p["unit_id"]]
        if lab["uncertain"] or not lab["remove_heads"]:
            continue
        h = removed(sc[(p["source_doc_id"], p["ss"], p["se"])], thr)
        total[0], total[1] = total[0] + 1, total[1] + h
        if p["origin"] != "score_tranche":
            cue[0], cue[1] = cue[0] + 1, cue[1] + h
        for head in lab["remove_heads"]:
            per_head[head][0] += 1
            per_head[head][1] += h
        docs[p["source_doc_id"]] = docs.get(p["source_doc_id"], False) or not h
    ln = json.loads((F / "labels-neighbours.json").read_text())
    neigh = [
        removed(sc[(p["source_doc_id"], p["ss"], p["se"])], thr)
        for p in json.loads((F / "neighbour-private.json").read_text())
        if not ln[p["unit_id"]]["uncertain"] and not ln[p["unit_id"]]["remove_heads"]
    ]
    return {
        "ipw_harmless_deleted": f"{rate:.2%} [{lo:.2%}, {hi:.2%}]",
        "_ipw": rate,
        "neighbour_bleed": f"{sum(neigh)}/{len(neigh)}",
        "held_out_recall": ci(total[1], total[0]),
        "_recall": total[1] / max(total[0], 1),
        "held_out_recall_cue_mined_only": ci(cue[1], cue[0]),
        "held_out_recall_by_head": {h: ci(k, n) for h, (n, k) in per_head.items()},
        "held_out_doc_leak": f"{sum(docs.values())}/{len(docs)}",
    }


def main() -> None:
    private_dir(F)
    cache = F / "eval-scores.json"
    keys = instrument_keys()
    if cache.exists():
        raw = json.loads(cache.read_text())
        inst = {m: {tuple(k): v for k, v in rows} for m, rows in raw.items()}
    else:
        inst = {
            "A0": v1_instrument("A0", V1, keys),
            "it1": v1_instrument("it1", IT1, keys),
            "RunA": factored_instrument(keys),
        }
        cache.write_text(
            json.dumps({m: [[list(k), v] for k, v in s.items()] for m, s in inst.items()})
        )
    audit = audit_scores()
    points = {}
    for m in ("A0", "it1", "RunA"):
        top = {k: max(v) for k, v in audit[m].items()}
        points[m] = [threshold_for(top)] * 4
    points["A0_production"] = v1_thresholds()
    results: dict[str, Any] = {}
    for name, thr in points.items():
        m = "A0" if name.startswith("A0") else name
        results[name] = {
            "thresholds": [round(t, 5) for t in thr],
            "audit100": audit_metrics(audit[m], thr),
            "instrument": instrument_metrics(inst[m], thr),
        }
    a0_ipw = min(results[n]["instrument"]["_ipw"] for n in ("A0", "A0_production"))

    def passes(name: str) -> bool:
        return (
            results[name]["audit100"]["_recall_hits"] >= 156
            and results[name]["instrument"]["_recall"] >= 0.98
        )

    if passes("RunA") and results["RunA"]["instrument"]["_ipw"] <= 0.015:
        winner = "RunA"
    elif passes("it1") and results["it1"]["instrument"]["_ipw"] < a0_ipw:
        winner = "it1"
    else:
        winner = "A0_production"
    results["winner"] = {"model": winner, "thresholds": results[winner]["thresholds"]}
    write_json(F / "eval-final.json", results)
    print(
        json.dumps(
            {
                k: v
                if not isinstance(v, dict)
                else {
                    a: (
                        {c: d for c, d in b.items() if not c.startswith("_")}
                        if isinstance(b, dict)
                        else b
                    )
                    for a, b in v.items()
                }
                for k, v in results.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
