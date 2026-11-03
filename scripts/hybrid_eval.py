"""Hybrid it3m + A0 heads: OR the two models' decisions, every frozen test.

A sentence is removed when it3m removes it at A0's production thresholds, OR
an A0 head named by the variant clears its own production threshold:
  hybrid    = it3m OR A0 penale;
  hybrid_b  = it3m OR A0 penale OR A0 salute;
  it3m_c    = it3m alone, penale threshold lowered on the train-side dev set
              until its dev penale recall equals A0's (see ``penale_threshold``).
Because both models use the same per-head thresholds, an OR is exact as a
per-head max of the two score vectors, so the existing report functions apply
unchanged. Leak counts get Wilson intervals and an exact McNemar test against
A0 on the same documents. Reads cached scores only; prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any

from scripts import counterfactuals as cf
from scripts.clean_mining import wilson
from scripts.final_eval import (
    F,
    audit_metrics,
    instrument_metrics,
    removed,
)
from scripts.it3_eval import (
    I3,
    audit_scores_for,
    cached,
    cf_requests,
    instrument_scores,
    sentenze_requests,
    verdict,
)
from scripts.phase0_eval import audit_labels
from scripts.proof_score import HEADS, v1_thresholds, write_json
from scripts.sentenze import S
from scripts.sentenze import summarise as sentenze_summary

Scores = dict[tuple[str, ...], list[float]]
TESTS = ("instrument", "audit100", "sentenze", "counterfactuals")
PENALE = HEADS.index("penale")
SALUTE = HEADS.index("salute")


def combine(base: Scores, extra: Scores, heads: tuple[int, ...]) -> Scores:
    """Per-head max of ``base`` and, on ``heads`` only, ``extra``."""
    out: Scores = {}
    for k in base.keys() | extra.keys():
        b = list(base.get(k, [0.0] * 4))
        e = extra.get(k, [0.0] * 4)
        for h in heads:
            b[h] = max(b[h], e[h])
        out[k] = b
    return out


def penale_threshold(
    a0_rows: list[dict[str, Any]], rows: list[dict[str, Any]], thr: float
) -> dict[str, Any]:
    """Highest penale threshold <= ``thr`` whose dev penale recall reaches A0's.

    Recall is counted on the penale head alone, on dev positives that carry
    penale. With no penale positive on dev the recall rule is vacuous; the
    fallback matches A0's penale-head firing count on all certain dev rows.
    """

    def fires(rs: list[dict[str, Any]], t: float, pos_only: bool) -> int:
        return sum(r["s"][PENALE] >= t for r in rs if not pos_only or "penale" in r["heads"])

    positives = sum("penale" in r["heads"] for r in rows)
    pos_only = positives > 0
    target = fires(a0_rows, thr, pos_only)
    grid = sorted({round(10 ** (-4 + 4 * i / 400), 6) for i in range(401)} | {thr}, reverse=True)
    chosen = next((t for t in grid if t <= thr and fires(rows, t, pos_only) >= target), grid[-1])
    return {
        "dev_penale_positives": positives,
        "rule": "penale recall on dev positives" if pos_only else "penale-head firing count on dev",
        "A0_count": target,
        "it3m_count_at_A0_threshold": fires(rows, thr, pos_only),
        "threshold": chosen,
        "it3m_count_at_chosen": fires(rows, chosen, pos_only),
    }


# --- leaks per unit, for paired tests ---------------------------------------------


def instrument_leaks(sc: Scores, thr: list[float]) -> dict[str, bool]:
    lp = json.loads((F / "labels-positives.json").read_text())
    docs: dict[str, bool] = {}
    for p in json.loads((F / "positive-private.json").read_text()):
        lab = lp[p["unit_id"]]
        if lab["uncertain"] or not lab["remove_heads"]:
            continue
        hit = removed(sc[(p["source_doc_id"], p["ss"], p["se"])], thr)
        docs[p["source_doc_id"]] = docs.get(p["source_doc_id"], False) or not hit
    return docs


def audit_leaks(sc: Scores, thr: list[float]) -> dict[str, bool]:
    docs: dict[str, bool] = {}
    for doc, rows in audit_labels().items():
        for r in rows:
            if set(r["labels"]) & set(HEADS):
                hit = removed(sc.get((doc, r["start"], r["end"])), thr)
                docs[doc] = docs.get(doc, False) or not hit
    return docs


def sentenze_leaks(sc: Scores, thr: list[float]) -> dict[str, bool]:
    labels = json.loads((S / "labels.json").read_text())
    out: dict[str, bool] = {}
    for p in json.loads((S / "units-private.json").read_text()):
        lab = labels[p["unit_id"]]
        if lab["uncertain"] or not lab["remove_heads"]:
            continue
        hit = removed(sc.get((p["decision"], p["ss"], p["se"]), [0.0] * 4), thr)
        out[p["decision"]] = out.get(p["decision"], False) or not hit
    return out


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    """Exact two-sided McNemar on the discordant units of two paired leak vectors."""
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n if n else 1.0
    return {"only_A0": only_a, "only_variant": only_b, "p_exact": round(min(1.0, 2 * tail), 4)}


# --- reports -----------------------------------------------------------------------


def report(test: str, sc: Scores, thr: list[float]) -> dict[str, Any]:
    if test == "instrument":
        return instrument_metrics(sc, thr)
    if test == "audit100":
        return audit_metrics(sc, thr)
    if test == "sentenze":
        private = json.loads((S / "units-private.json").read_text())
        for p in private:
            sc.setdefault((p["decision"], p["ss"], p["se"]), [0.0] * 4)
        return sentenze_summary(private, json.loads((S / "labels.json").read_text()), sc, thr)
    return cf.pair_metrics(cf.kept_pairs()[0], sc, thr)


def model_scores(model: str, test: str) -> Scores:
    if test == "instrument":
        return instrument_scores(model)
    if test == "audit100":
        return audit_scores_for(model)
    return cached(model, test, sentenze_requests if test == "sentenze" else cf_requests)


def variants() -> dict[str, tuple[dict[str, Scores], list[float]]]:
    from scripts.it3_train import dev_rows

    thr = v1_thresholds()
    a0 = {t: model_scores("A0", t) for t in TESTS}
    m = {t: model_scores("it3m", t) for t in TESTS}
    c = penale_threshold(dev_rows("A0"), dev_rows("it3m"), thr[PENALE])
    write_json(I3 / "hybrid-penale-threshold.json", c)
    thr_c = thr[:PENALE] + [c["threshold"]] + thr[PENALE + 1 :]
    return {
        "A0@production": (a0, thr),
        "it3m@A0_fixed": (m, thr),
        "hybrid@A0_fixed": ({t: combine(m[t], a0[t], (PENALE,)) for t in TESTS}, thr),
        "hybrid_b@A0_fixed": ({t: combine(m[t], a0[t], (PENALE, SALUTE)) for t in TESTS}, thr),
        "it3m_c@dev_penale": (m, thr_c),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    out: dict[str, Any] = {}
    leaks: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    for name, (sc, thr) in variants().items():
        row = out.setdefault(name, {"thresholds": [round(t, 5) for t in thr]})
        for t in TESTS:
            row[t] = report(t, dict(sc[t]), thr)
        leaks[name] = {
            "instrument": instrument_leaks(sc["instrument"], thr),
            "audit100": audit_leaks(sc["audit100"], thr),
            "sentenze": sentenze_leaks(sc["sentenze"], thr),
        }
    for name in out:
        out[name]["leak_stats"] = {
            part: {
                "wilson": wilson(sum(v.values()), len(v)),
                "vs_A0": mcnemar(leaks["A0@production"][part], v),
            }
            for part, v in leaks[name].items()
        }
    out["verdicts"] = [verdict(out, k) for k in out if k != "A0@production"]
    write_json(I3 / "eval-hybrid.json", out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
