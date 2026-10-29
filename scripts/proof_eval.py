"""One-day proof, step 6: A0 (v1) against the fine-tuned model.

Scores each model once on: the train-side dev slice (held-out Sol groups plus
v1's real validation rows), the sealed audit100 windows, the held-out half of
the edge suite and the frozen artificial-context probe. Thresholds per head
are recalibrated on the dev slice only (recall target), never on audit100.
Writes aggregate gates; prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from perizia_privacy.corpus import example_text, windows
from perizia_privacy.retraining import training as rt
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    V1,
    edge_rows,
    load_model,
    private_dir,
    score_ids,
    score_texts,
    v1_thresholds,
    write_json,
)
from scripts.proof_train import (
    edge_examples,
    edge_relabels,
    sol_examples,
)

RECALL = 0.98
PARTS = ("dev", "edge", "audit", "probe_original_ids", "probe_padded_ids")
PARTS2 = ("edge", "audit", "probe_original_ids", "probe_padded_ids", "cal", "pop")
CAL_RECALL = 0.97
DELETION_CAP = 0.03


def dev_slice() -> list[dict[str, Any]]:
    rows = sol_examples(dev=True)
    with (ROOT / "development-preparation-v3/training-inputs-v3/validation.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            rows.append({"text": r["text"], "labels": r["labels"], "kind": "v1_validation"})
    return rows


def cal_rows() -> list[dict[str, Any]]:
    """Iteration 2 calibration units: groups that no training run has seen."""
    from scripts.proof_iter2 import calibration_groups

    groups = calibration_groups()
    return sol_examples(None, groups=groups) + sol_examples(None, "iter2-", "cal")


def pop_rows() -> list[dict[str, Any]]:
    """Random sentences from the same unused groups, for the deletion cap."""
    paths = {
        d["source_doc_id"]: d["path"]
        for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }
    by_doc: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for p in json.loads((OUT / "iter2-population.json").read_text()):
        by_doc[p["source_doc_id"]].add((p["ss"], p["se"]))
    out = []
    for src, spans in sorted(by_doc.items()):
        text = Path(paths[src]).read_text(encoding="utf-8")
        for a, b, s0, s1 in windows(text):
            if (s0, s1) in spans:
                out.append({"text": example_text(text, a, b), "key": (src, s0, s1)})
    return out


def calibrate_capped(sc: dict[str, np.ndarray]) -> tuple[list[float], dict[str, Any]]:
    """~97% recall per head on calibration positives, then raise thresholds
    greedily (cheapest recall loss first) until population deletion <= 3%."""
    cal = cal_rows()
    units: dict[str, dict[str, Any]] = {}
    for r, row in zip(cal, sc["cal"], strict=True):
        u = units.setdefault(r["unit"], {"labels": r["labels"], "s": np.zeros(4)})
        u["s"] = np.maximum(u["s"], row[:4])
    pop: dict[tuple[str, int, int], np.ndarray] = {}
    for r, row in zip(pop_rows(), sc["pop"], strict=True):
        pop[r["key"]] = np.maximum(pop.get(r["key"], np.zeros(4)), row[:4])
    pop_s = np.array(list(pop.values()))
    pos = [
        np.sort([u["s"][h] for u in units.values() if head in u["labels"]])
        for h, head in enumerate(HEADS)
    ]
    thr = [
        float(p[int(np.floor((1 - CAL_RECALL) * len(p)))]) if len(p) else t
        for p, t in zip(pos, v1_thresholds(), strict=True)
    ]

    def deletion(t: list[float]) -> float:
        return float((np.array(t) <= pop_s).any(axis=1).mean())

    while deletion(thr) > DELETION_CAP:
        best = None
        for h in range(4):
            above = pop_s[:, h][pop_s[:, h] >= thr[h]]
            if not len(above):
                continue
            t_new = float(above.min()) + 1e-6
            trial = thr[:h] + [t_new] + thr[h + 1 :]
            lost = int(((pos[h] >= thr[h]) & (pos[h] < t_new)).sum())
            gain = deletion(thr) - deletion(trial)
            cost = (lost + 1e-3) / max(gain, 1e-9)
            if best is None or cost < best[0]:
                best = (cost, h, t_new)
        if best is None:
            break
        thr[best[1]] = best[2]
    info = {
        "calibration_positives": {h: len(p) for h, p in zip(HEADS, pos, strict=True)},
        "calibration_recall": {
            h: f"{int((p >= t).sum())}/{len(p)}" for h, p, t in zip(HEADS, pos, thr, strict=True)
        },
        "population_sentences": len(pop_s),
        "population_deletion": round(deletion(thr), 4),
    }
    return thr, info


def audit_frames() -> list[dict[str, Any]]:
    with (ROOT / "final-document-audit-adaptive-v1/frames.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def scores_for(name: str, path: Path, parts: tuple[str, ...] = PARTS) -> dict[str, np.ndarray]:
    """Score each evaluation set once per model; each part is cached separately."""
    cache = private_dir(OUT / "eval")
    out = {
        p: np.load(cache / f"{name}-{p}.npy") for p in parts if (cache / f"{name}-{p}.npy").exists()
    }
    if len(out) == len(parts):
        return out
    torch, tok, model = load_model(path)
    probe = json.loads(
        (ROOT / "retraining-20260929-context-probe-v1/paired-inputs.json").read_text()
    )
    for part in parts:
        if part in out:
            continue
        if part.startswith("probe_"):
            rows = [r[part[len("probe_") :]] for r in probe["rows"]]
            s = np.zeros((len(rows), 5), dtype=np.float32)
            for idx, sc in score_ids(torch, model, tok.pad_token_id, rows):
                s[idx] = sc
        else:
            texts = {
                "dev": dev_slice,
                "edge": lambda: edge_examples(train=False),
                "audit": audit_frames,
                "cal": cal_rows,
                "pop": pop_rows,
            }[part]()
            s = score_texts(torch, tok, model, [r["text"] for r in texts])
        np.save(cache / f"{name}-{part}.npy", s)
        out[part] = s
    return out


def calibrate(dev: list[dict[str, Any]], s: np.ndarray) -> list[float]:
    """Highest threshold per head with dev recall >= RECALL."""
    thr = []
    for h, head in enumerate(HEADS):
        pos = np.sort(np.array([s[i, h] for i, r in enumerate(dev) if head in r["labels"]]))
        k = int(np.floor((1 - RECALL) * len(pos)))
        thr.append(float(pos[k]) if len(pos) else 0.5)
    return thr


def audit_gates(frames: list[dict[str, Any]], s: np.ndarray, thr: list[float]) -> dict[str, Any]:
    removed: dict[tuple[str, int, int], bool] = defaultdict(bool)
    for f, row in zip(frames, s, strict=True):
        key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
        removed[key] |= any(row[h] >= thr[h] for h in range(4))
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (ROOT / "audit100-relabel-v1/labels-merged-qa.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            labels[r["source_doc_id"]].append(r)
    sens = sens_hit = harm = harm_hit = adj = adj_hit = 0
    for doc, rows in labels.items():
        rows.sort(key=lambda r: r["index"])
        positive = [i for i, r in enumerate(rows) if set(r["labels"]) & set(HEADS)]
        for i, r in enumerate(rows):
            hit = removed[(doc, r["start"], r["end"])]
            if set(r["labels"]) & set(HEADS):
                sens += 1
                sens_hit += hit
            elif not r["labels"]:
                harm += 1
                harm_hit += hit
                if any(abs(i - j) == 1 for j in positive):
                    adj += 1
                    adj_hit += hit
    return {
        "sensitive_removed": f"{sens_hit}/{sens}",
        "harmless_deleted": f"{harm_hit}/{harm} = {harm_hit / harm:.2%}",
        "adjacent_neighbour_removed": f"{adj_hit}/{adj}",
    }


def edge_gates(edge: list[dict[str, Any]], s: np.ndarray, thr: list[float]) -> dict[str, Any]:
    per_row: dict[str, dict[str, Any]] = {}
    for r, row in zip(edge, s, strict=True):
        cur = per_row.setdefault(r["row"], {"pol": r["polarity"], "t": [], "n": []})
        fired = any(row[h] >= thr[h] for h in range(4))
        (cur["t"] if r["is_target"] else cur["n"]).append(fired)
    rem = [all(v["t"]) for v in per_row.values() if v["pol"] == "remove"]
    keep = [not any(v["t"]) for v in per_row.values() if v["pol"] == "keep"]
    neigh = [x for v in per_row.values() if v["pol"] == "remove" for x in v["n"]]
    return {
        "sensitive_removed": f"{sum(rem)}/{len(rem)} = {sum(rem) / len(rem):.1%}",
        "harmless_twin_kept": f"{sum(keep)}/{len(keep)} = {sum(keep) / len(keep):.1%}",
        "neighbour_of_sensitive_removed": f"{sum(neigh)}/{len(neigh)}",
    }


def probe_gates(sc: dict[str, np.ndarray], thr: list[float]) -> dict[str, Any]:
    rows, _ = rt.load_rows(
        ROOT / "retraining-20260929-exploratory-data-v1/train-mixed.jsonl", "train"
    )
    probe = json.loads(
        (ROOT / "retraining-20260929-context-probe-v1/paired-inputs.json").read_text()
    )
    by_id = {r.id: r for r in rows}
    out = {}
    for key in ("original_ids", "padded_ids"):
        tp = pos = fp = neg = 0
        for k, p in enumerate(probe["rows"]):
            r = by_id[p["id"]]
            for h in range(len(HEADS)):
                fired = sc["probe_" + key][k, h] >= thr[h]
                if not r.loss_mask[h]:
                    continue
                if r.labels[h]:
                    pos += 1
                    tp += fired
                else:
                    neg += 1
                    fp += fired
        out[key.split("_")[0]] = f"TP {tp}/{pos}, FP {fp}/{neg}"
    return out


def report(name: str, sc: dict[str, np.ndarray], thr: list[float]) -> dict[str, Any]:
    edge = edge_examples(train=False)
    out = {
        "thresholds": [round(t, 4) for t in thr],
        "audit100": audit_gates(audit_frames(), sc["audit"], thr),
        "edge_heldout": edge_gates(edge, sc["edge"], thr),
        "context_probe": probe_gates(sc, thr),
    }
    if (OUT / "iter2-labels.json").exists():
        relabeled = edge_examples(train=False, relabel=edge_relabels())
        out["edge_heldout_relabeled"] = edge_gates(relabeled, sc["edge"], thr)
    return out


def main_iter2(models: list[str]) -> None:
    """Every gate for each model at v1's fixed thresholds and at capped calibration."""
    results = {}
    for name in models:
        sc = scores_for(name, V1 if name == "A0" else OUT / name, PARTS2)
        thr, info = calibrate_capped(sc)
        results[name] = {
            "fixed_v1_thresholds": report(name, sc, v1_thresholds()),
            "calibrated": report(name, sc, thr) | {"calibration": info},
        }
    write_json(OUT / "eval-iter2.json", results)
    print(json.dumps(results, indent=1))


def edge_by_head(s: np.ndarray, thr: list[float]) -> dict[str, str]:
    """Held-out edge removal and twin keeping per head (a twin takes its pair's heads)."""
    rows = {r["id"]: r for r in edge_rows()}
    heads_of = {
        r["pair_id"]: r["expected_labels"] for r in rows.values() if r["polarity"] == "remove"
    }
    fired: dict[str, list[bool]] = defaultdict(list)
    for e, x in zip(edge_examples(train=False), s, strict=True):
        if e["is_target"]:
            fired[e["row"]].append(any(x[h] >= thr[h] for h in range(4)))
    out = {}
    for head in HEADS:
        rem = [
            all(f)
            for i, f in fired.items()
            if rows[i]["polarity"] == "remove" and head in rows[i]["expected_labels"]
        ]
        keep = [
            not any(f)
            for i, f in fired.items()
            if rows[i]["polarity"] == "keep" and head in heads_of.get(rows[i]["pair_id"], [])
        ]
        out[head] = f"removed {sum(rem)}/{len(rem)}, twins kept {sum(keep)}/{len(keep)}"
    return out


def main_hybrid() -> None:
    """Per-head hybrids of cached A0 and iteration 1 scores at v1's fixed thresholds."""
    thr = v1_thresholds()
    a0 = scores_for("A0", V1, PARTS2)
    it1 = scores_for("model", OUT / "model", PARTS2)
    results = {
        "A0_by_head": edge_by_head(a0["edge"], thr),
        "it1_by_head": edge_by_head(it1["edge"], thr),
    }
    for name, from_it1 in (
        ("hybrid_famiglia_from_it1", {3}),
        ("hybrid_famiglia_from_A0_rest_it1", {0, 1, 2}),
    ):
        sc = {}
        for part in PARTS2:
            mix = a0[part].copy()
            for h in from_it1:
                mix[:, h] = it1[part][:, h]
            sc[part] = mix
        results[name] = report(name, sc, thr) | {"by_head": edge_by_head(sc["edge"], thr)}
    write_json(OUT / "eval-hybrid.json", results)
    print(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--iter2", default="", help="comma-separated models, e.g. A0,model,model2")
    ap.add_argument("--hybrid", action="store_true", help="per-head A0/iteration 1 hybrids")
    args = ap.parse_args()
    if args.hybrid:
        main_hybrid()
        return
    if args.iter2:
        main_iter2(args.iter2.split(","))
        return
    dev = dev_slice()
    results = {}
    a0 = scores_for("A0", V1)
    new = scores_for(args.model, OUT / args.model)
    results["A0_production_thresholds"] = report("A0", a0, v1_thresholds())
    results["A0_recalibrated"] = report("A0", a0, calibrate(dev, a0["dev"]))
    results[f"{args.model}_recalibrated"] = report(args.model, new, calibrate(dev, new["dev"]))
    # The dev recall rule above collapses thresholds (few, noisy dev positives);
    # v1's ratified thresholds were never tuned on audit100, so they are the
    # honest fixed operating point for the comparison.
    results[f"{args.model}_v1_thresholds"] = report(args.model, new, v1_thresholds())
    sol = [r for r in dev if r.get("kind") != "v1_validation"]
    results[f"{args.model}_sol_dev_recalibrated"] = report(
        args.model, new, calibrate(sol, new["dev"][: len(sol)])
    )
    results["dev_slice"] = {
        "rows": len(dev),
        "positives_per_head": {h: sum(h in r["labels"] for r in dev) for h in HEADS},
        "sol_rows": sum(r.get("kind") != "v1_validation" for r in dev),
        "recall_target": RECALL,
    }
    write_json(OUT / f"eval-{args.model}.json", results)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
