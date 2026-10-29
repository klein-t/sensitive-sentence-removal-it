"""it3: score models on every frozen test and apply the ship rule.

Tests: the clean rare-class frame, the real counterfactual pairs, the
Cassazione decisions (each scored with the report functions that built it),
the final instrument (held-out recall, IPW deletion, document leak) and
audit100. Models: A0 and it1 (v1 encoding), Run A (factored student), it3 and
its margin variant (v1 encoding). Operating points: A0 production per-head
thresholds; it1 and Run A at their final-evaluation global thresholds; it3 at
A0's fixed thresholds and at the per-head thresholds chosen on the train-side
dev set. Scores are cached per model and test. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from perizia_privacy import factored as fs
from perizia_privacy.corpus import example_text, windows
from scripts import counterfactuals as cf
from scripts.clean_mining import C
from scripts.clean_mining import summarise as clean_summary
from scripts.final_eval import (
    RUN_A,
    F,
    audit_metrics,
    audit_scores,
    instrument_keys,
    instrument_metrics,
)
from scripts.final_instrument import IT1, score_v1
from scripts.phase0_eval import doc_text
from scripts.proof_eval import audit_frames
from scripts.proof_score import (
    ROOT,
    V1,
    private_dir,
    v1_thresholds,
    write_json,
)
from scripts.sentenze import S, decisions
from scripts.sentenze import summarise as sentenze_summary

if TYPE_CHECKING:
    from pathlib import Path

I3 = ROOT / "it3-20261009"
MODELS: dict[str, tuple[str, Path]] = {
    "A0": ("v1", V1),
    "it1": ("v1", IT1),
    "RunA": ("factored", RUN_A),
    "it3": ("v1", I3 / "model"),
    "it3m": ("v1", I3 / "model-margin"),
}
TESTS = ("clean", "counterfactuals", "sentenze", "instrument", "audit100")
Request = tuple[tuple[str, ...], str, int, int]  # key, text id, start, end
Scores = dict[tuple[str, ...], list[float]]


def operating_points(model: str) -> dict[str, list[float]]:
    final = json.loads((F / "eval-final.json").read_text())
    if model == "A0":
        return {"production": v1_thresholds()}
    if model in ("it1", "RunA"):
        return {"final_global": final[model]["thresholds"], "A0_fixed": v1_thresholds()}
    points = {"A0_fixed": v1_thresholds()}
    chosen = I3 / f"thresholds-{model}.json"
    if chosen.exists():
        points["dev_chosen"] = json.loads(chosen.read_text())["thresholds"]
    return points


# --- scoring -----------------------------------------------------------------


def score_ranges(model: str, requests: list[Request], texts: dict[str, str]) -> Scores:
    """Per-request per-head max over every window or target overlapping [start, end)."""
    kind, path = MODELS[model]
    by_text: dict[str, list[Request]] = defaultdict(list)
    for r in requests:
        by_text[r[1]].append(r)
    if kind == "v1":
        items = []
        for tid, reqs in by_text.items():
            text = texts[tid]
            for a, b, _, _ in windows(text):
                for key, _, s, e in reqs:
                    if a < e and b > s:
                        items.append((key, example_text(text, a, b)))
        return score_v1({model: path}, items)[model] if items else {}
    torch, tok, net = fs.load(path, "mps")
    out: Scores = {}
    for tid, reqs in sorted(by_text.items()):
        text = texts[tid]
        targets = [
            t
            for t in fs.document_targets(text, tok)
            if any(t[0] < e and t[1] > s for _, _, s, e in reqs)
        ]
        probs = fs.score_targets(torch, tok, net, text, targets, "mps", 32)
        for (a, b, _), row in zip(targets, probs, strict=True):
            for key, _, s, e in reqs:
                if a < e and b > s:
                    prev = out.get(key, [0.0] * 4)
                    out[key] = [max(x, float(y)) for x, y in zip(prev, row[fs.REMOVE], strict=True)]
    del net
    torch.mps.empty_cache()
    return out


def cached(model: str, test: str, build) -> Scores:
    path = private_dir(I3 / "scores") / f"{model}-{test}.json"
    if path.exists():
        return {tuple(k): v for k, v in json.loads(path.read_text())}
    requests, texts = build()
    sc = score_ranges(model, requests, texts)
    write_json(path, [[list(k), v] for k, v in sc.items()])
    return sc


def corpus_texts(srcs: set[str]) -> dict[str, str]:
    return {s: doc_text(s) for s in srcs}


def clean_requests() -> tuple[list[Request], dict[str, str]]:
    private = json.loads((C / "units-private.json").read_text())
    reqs = [
        ((p["source_doc_id"], p["ss"], p["se"]), p["source_doc_id"], p["ss"], p["se"])
        for p in private
    ]
    return reqs, corpus_texts({p["source_doc_id"] for p in private})


def sentenze_requests() -> tuple[list[Request], dict[str, str]]:
    private = json.loads((S / "units-private.json").read_text())
    wanted = {p["decision"] for p in private}
    texts = {d["id"]: d["text"] for d in decisions() if d["id"] in wanted}
    reqs = [((p["decision"], p["ss"], p["se"]), p["decision"], p["ss"], p["se"]) for p in private]
    return reqs, texts


def cf_requests() -> tuple[list[Request], dict[str, str]]:
    pairs, _, frame = cf.kept_pairs()
    reqs: list[Request] = []
    texts: dict[str, str] = {}
    for p in pairs:
        text = frame[p["source_doc_id"]]
        texts[p["source_doc_id"]] = text
        texts[p["uid"]] = text[: p["ss"]] + p["edited"] + text[p["se"] :]
        reqs.append(((p["uid"], "original"), p["source_doc_id"], p["ss"], p["se"]))
        reqs.append(((p["uid"], "edited"), p["uid"], p["ss"], p["ss"] + len(p["edited"])))
    return reqs, texts


def instrument_requests() -> tuple[list[Request], dict[str, str]]:
    keys = {k for part in instrument_keys().values() for k in part}
    return [(k, k[0], k[1], k[2]) for k in keys], corpus_texts({k[0] for k in keys})


def instrument_scores(model: str) -> Scores:
    if model in ("A0", "it1", "RunA"):  # the frozen final-evaluation cache
        raw = json.loads((F / "eval-scores.json").read_text())[model]
        sc = {tuple(k): v for k, v in raw}
    else:
        sc = cached(model, "instrument", instrument_requests)
    for k in {k for part in instrument_keys().values() for k in part}:
        sc.setdefault(k, [0.0] * 4)
    return sc


def audit_scores_for(model: str) -> Scores:
    if model in ("A0", "it1", "RunA"):
        return audit_scores()[model]
    path = private_dir(I3 / "scores") / f"{model}-audit100.json"
    if path.exists():
        return {tuple(k): v for k, v in json.loads(path.read_text())}
    items = [
        ((f["source_doc_id"], f["sentence_start"], f["sentence_end"]), f["text"])
        for f in audit_frames()
    ]
    sc = score_v1({model: MODELS[model][1]}, items)[model]
    write_json(path, [[list(k), v] for k, v in sc.items()])
    return sc


# --- reports -----------------------------------------------------------------


def test_report(model: str, test: str, thr: list[float]) -> dict[str, Any]:
    if test == "clean":
        sc = cached(model, test, clean_requests)
        private = json.loads((C / "units-private.json").read_text())
        for p in private:
            sc.setdefault((p["source_doc_id"], p["ss"], p["se"]), [0.0] * 4)
        return clean_summary(private, json.loads((C / "labels.json").read_text()), sc, thr)
    if test == "sentenze":
        sc = cached(model, test, sentenze_requests)
        private = json.loads((S / "units-private.json").read_text())
        for p in private:
            sc.setdefault((p["decision"], p["ss"], p["se"]), [0.0] * 4)
        return sentenze_summary(private, json.loads((S / "labels.json").read_text()), sc, thr)
    if test == "counterfactuals":
        sc = cached(model, test, cf_requests)
        return cf.pair_metrics(cf.kept_pairs()[0], sc, thr)
    if test == "instrument":
        return instrument_metrics(instrument_scores(model), thr)
    if test == "audit100":
        return audit_metrics(audit_scores_for(model), thr)
    raise ValueError(test)


def evaluate(models: list[str], tests: list[str], out_name: str) -> dict[str, Any]:
    out_path = I3 / out_name
    report: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in models:
        for point, thr in operating_points(m).items():
            row = report.setdefault(f"{m}@{point}", {"thresholds": [round(t, 5) for t in thr]})
            row["thresholds"] = [round(t, 5) for t in thr]
            for t in tests:
                row[t] = test_report(m, t, thr)
                write_json(out_path, report)
                print(m, point, t, "done", flush=True)
    return report


def verdict(report: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Ship rule: held-out recall >= 99%, audit >= 156/158, leak <= A0, IPW deletion < A0."""
    a0 = report["A0@production"]
    c = report[candidate]
    leak = lambda r: int(r["instrument"]["held_out_doc_leak"].split("/")[0])  # noqa: E731
    checks = {
        "held_out_recall_ge_99": c["instrument"]["_recall"] >= 0.99,
        "audit_ge_156": c["audit100"]["_recall_hits"] >= 156,
        "leak_le_A0": leak(c) <= leak(a0),
        "ipw_deletion_lt_A0": c["instrument"]["_ipw"] < a0["instrument"]["_ipw"],
    }
    return {"candidate": candidate, "checks": checks, "ship_candidate": all(checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("step0", "final"))
    args = ap.parse_args()
    private_dir(I3)
    if args.step == "step0":
        report = evaluate(
            ["A0", "it1", "RunA"], ["clean", "counterfactuals", "sentenze"], "step0.json"
        )
    else:
        models = ["A0", "it1"] + [m for m in ("it3", "it3m") if MODELS[m][1].exists()]
        report = evaluate(models, list(TESTS), "eval-it3.json")
        report["verdicts"] = [
            verdict(report, k) for k in report if k.startswith("it3") and "@" in k
        ]
        write_json(I3 / "eval-it3.json", report)
    print(json.dumps(strip(report), indent=1))


def strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not str(k).startswith("_")}
    return value


if __name__ == "__main__":
    main()
