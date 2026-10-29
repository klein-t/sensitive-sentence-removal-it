"""Phase 0: learning-curve evaluation against A0 (v1) and iteration 1.

Every model is scored through the production encode path
(`factored.document_targets` + `score_targets`). One global threshold on the
max remove probability is chosen per model as the highest that keeps audit100
recall >= 156/158; A0 and iteration 1 get the same rule from their cached
scores. At that point: audit100 deletion and bleed, gate-keep accuracy on the
held-out probe units (Sol A/B consensus), and the held-out edge suite as a
diagnostic. A redactor smoke check confirms `FactoredRedactor` removes exactly
the sentences the evaluation counts. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from perizia_privacy import factored as fs
from perizia_privacy.corpus import sentence_spans
from scripts.proof_eval import audit_frames
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    edge_rows,
    private_dir,
    v1_thresholds,
    write_json,
)
from scripts.proof_train import (
    edge_examples,
    edge_family_train,
    target_sentences,
)

P0 = ROOT / "phase0-20261003"
RECALL_FLOOR = 156

Key = tuple[str, int, int]


def audit_labels() -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (ROOT / "audit100-relabel-v1/labels-merged-qa.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            labels[r["source_doc_id"]].append(r)
    for rows in labels.values():
        rows.sort(key=lambda r: r["index"])
    return labels


def doc_text(src: str, _paths: dict[str, str] = {}) -> str:  # noqa: B006 - tiny cache
    if not _paths:
        docs = json.loads((ROOT / "corpus-v1/documents.json").read_text())
        _paths.update({d["source_doc_id"]: d["path"] for d in docs})
    return Path(_paths[src]).read_text(encoding="utf-8")


def sentence_scores(model_dir: Path, name: str) -> dict[str, dict[Key, np.ndarray]]:
    """Per-sentence max over targets of every output, per part; cached per model."""
    cache = private_dir(P0 / "eval") / f"{name}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {part: {tuple(k): np.array(v) for k, v in rows} for part, rows in raw.items()}
    torch, tok, model = fs.load(model_dir, "mps")
    parts: dict[str, dict[Key, np.ndarray]] = {"audit": {}, "probe": {}, "edge": {}}

    def score(part: str, src: str, text: str, wanted: set[tuple[int, int]] | None) -> None:
        spans = sentence_spans(text)
        targets = [
            t for t in fs.document_targets(text, tok) if wanted is None or spans[t[2]] in wanted
        ]
        probs = fs.score_targets(torch, tok, model, text, targets, "mps", 32)
        for (_, _, i), row in zip(targets, probs, strict=True):
            key = (src, *spans[i])
            cur = parts[part].get(key)
            parts[part][key] = np.array(row) if cur is None else np.maximum(cur, row)

    for src in audit_labels():
        score("audit", src, doc_text(src), None)
    by_doc: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for p in json.loads((P0 / "probe-private.json").read_text()):
        by_doc[p["source_doc_id"]].add((p["ss"], p["se"]))
    for src, wanted in by_doc.items():
        score("probe", src, doc_text(src), wanted)
    for row in edge_rows():
        if not edge_family_train(row):
            score("edge", row["id"], row["paragraph"], None)
    cache.write_text(
        json.dumps(
            {p: [[list(k), v.tolist()] for k, v in rows.items()] for p, rows in parts.items()}
        )
    )
    return parts


def threshold_for(audit: dict[Key, float]) -> float:
    labels = audit_labels()
    sens = sorted(
        (
            audit.get((doc, r["start"], r["end"]), 0.0)
            for doc, rows in labels.items()
            for r in rows
            if set(r["labels"]) & set(HEADS)
        ),
        reverse=True,
    )
    return float(sens[RECALL_FLOOR - 1])


def audit_report(audit: dict[Key, float], t: float) -> dict[str, Any]:
    sens = sens_hit = harm = harm_hit = adj = adj_hit = 0
    per_head: dict[str, list[int]] = {h: [0, 0] for h in HEADS}
    for doc, rows in audit_labels().items():
        positive = [i for i, r in enumerate(rows) if set(r["labels"]) & set(HEADS)]
        for i, r in enumerate(rows):
            hit = audit.get((doc, r["start"], r["end"]), 0.0) >= t
            if set(r["labels"]) & set(HEADS):
                sens, sens_hit = sens + 1, sens_hit + hit
                for h in set(r["labels"]) & set(HEADS):
                    per_head[h][0] += 1
                    per_head[h][1] += hit
            elif not r["labels"]:
                harm, harm_hit = harm + 1, harm_hit + hit
                if any(abs(i - j) == 1 for j in positive):
                    adj, adj_hit = adj + 1, adj_hit + hit
    return {
        "threshold": round(t, 5),
        "sensitive_removed": f"{sens_hit}/{sens}",
        "harmless_deleted": f"{harm_hit}/{harm} = {harm_hit / harm:.2%}",
        "adjacent_neighbour_removed": f"{adj_hit}/{adj}",
        "sensitive_removed_by_head": {h: f"{k}/{n}" for h, (n, k) in per_head.items()},
    }


def at_budget(audit: dict[Key, float], budget: float = 597 / 37162) -> dict[str, Any]:
    """Recall at iteration 1's v1-threshold deletion rate (1.61%)."""
    harmless = sorted(
        (
            audit.get((doc, r["start"], r["end"]), 0.0)
            for doc, rows in audit_labels().items()
            for r in rows
            if not r["labels"]
        ),
        reverse=True,
    )
    return audit_report(audit, harmless[int(budget * len(harmless))] + 1e-9)


def edge_report(fired: dict[tuple[str, bool], list[bool]]) -> dict[str, Any]:
    rows = {r["id"]: r for r in edge_rows() if not edge_family_train(r)}
    rem = [all(fired[(i, True)]) for i, r in rows.items() if r["polarity"] == "remove"]
    keep = [not any(fired[(i, True)]) for i, r in rows.items() if r["polarity"] == "keep"]
    neigh = [x for i, r in rows.items() if r["polarity"] == "remove" for x in fired[(i, False)]]
    return {
        "sensitive_removed": f"{sum(rem)}/{len(rem)} = {sum(rem) / len(rem):.1%}",
        "harmless_twin_kept": f"{sum(keep)}/{len(keep)} = {sum(keep) / len(keep):.1%}",
        "neighbour_of_sensitive_removed": f"{sum(neigh)}/{len(neigh)}",
    }


def edge_new(edge: dict[Key, float], t: float) -> dict[str, Any]:
    fired: dict[tuple[str, bool], list[bool]] = defaultdict(list)
    for row in edge_rows():
        if edge_family_train(row):
            continue
        text = row["paragraph"]
        start = text.index(row["target"])
        targets = target_sentences(text, start, start + len(row["target"]))
        for a, b in sentence_spans(text):
            fired[(row["id"], (a, b) in targets)].append(edge.get((row["id"], a, b), 0.0) >= t)
    return edge_report(fired)


def probe_report(probe: dict[Key, float], t: float) -> dict[str, Any]:
    private = json.loads((P0 / "probe-private.json").read_text())
    labs = {tag: P0 / f"labels-{tag}.json" for tag in ("A", "B")}
    labs = {tag: json.loads(p.read_text()) for tag, p in labs.items() if p.exists()}
    if not labs:
        return {"note": "no Sol probe labels yet"}
    traps = kept = pos = removed = 0
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in private:
        votes = [labs[tag].get(p["unit_id"]) for tag in labs]
        if any(v is None or v["uncertain"] for v in votes):
            continue
        remove = {tuple(sorted(v["remove_heads"])) for v in votes}
        if len(remove) != 1:
            continue
        hit = probe.get((p["source_doc_id"], p["ss"], p["se"]), 0.0) >= t
        if remove == {()}:
            if any(v["topics"] for v in votes):
                traps, kept = traps + 1, kept + (not hit)
                by[p["stratum"]][0] += 1
                by[p["stratum"]][1] += not hit
        else:
            pos, removed = pos + 1, removed + hit
    return {
        "labels_from": "+".join(labs),
        "gate_keep_accuracy": f"{kept}/{traps} = {kept / max(traps, 1):.1%}",
        "gate_keep_by_stratum": {s: f"{k}/{n}" for s, (n, k) in by.items()},
        "positives_removed": f"{removed}/{pos}",
    }


def baseline(name: str) -> dict[str, Any]:
    """A0 / iteration 1 from cached frame scores, same recall rule."""
    sc = {p: np.load(OUT / "eval" / f"{name}-{p}.npy") for p in ("audit", "edge")}
    audit: dict[Key, float] = defaultdict(float)
    for f, row in zip(audit_frames(), sc["audit"], strict=True):
        key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
        audit[key] = max(audit[key], float(row[:4].max()))
    t = threshold_for(audit)
    fired: dict[tuple[str, bool], list[bool]] = defaultdict(list)
    for e, row in zip(edge_examples(train=False), sc["edge"], strict=True):
        fired[(e["row"], e["is_target"])].append(float(row[:4].max()) >= t)
    v1_audit_fire = defaultdict(bool)
    thr = v1_thresholds()
    for f, row in zip(audit_frames(), sc["audit"], strict=True):
        key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
        v1_audit_fire[key] |= any(row[h] >= thr[h] for h in range(4))
    return {
        "matched_recall": {"audit100": audit_report(audit, t), "edge_heldout": edge_report(fired)},
        "audit100_at_it1_deletion": at_budget(audit),
        "v1_thresholds": {
            "audit100": audit_report({k: float(v) for k, v in v1_audit_fire.items()}, 1.0)
        },
    }


def redactor_check(model_dir: Path, audit: dict[Key, float], t: float, docs: int = 3) -> str:
    """FactoredRedactor at threshold t removes exactly the sentences counted here."""
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["thresholds"] = [t] * 4
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    redactor = fs.FactoredRedactor(model_dir, device="mps")
    same = 0
    sources = sorted(audit_labels())[:docs]
    for src in sources:
        text = doc_text(src)
        result = redactor.redact(text)
        removed = set()
        for start, end in sentence_spans(text):
            original = text[start:end]
            lo = start + len(original) - len(original.lstrip())
            if any(s.start <= lo < s.end for s in result.spans):
                removed.add((start, end))
        expected = {
            (a, b) for (d, a, b), v in audit.items() if d == src and v >= t and text[a:b].strip()
        }
        same += removed == expected
    return f"{same}/{len(sources)} documents identical"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="student-25,student-50,student-100")
    args = ap.parse_args()
    results: dict[str, Any] = {"A0": baseline("A0"), "it1": baseline("model")}
    for name in args.models.split(","):
        model_dir = P0 / name
        if not (model_dir / "manifest.json").exists():
            continue
        parts = sentence_scores(model_dir, name)
        top = {
            p: {k: float(v[fs.REMOVE].max()) for k, v in rows.items()} for p, rows in parts.items()
        }
        t = threshold_for(top["audit"])
        results[name] = {
            "audit100": audit_report(top["audit"], t),
            "audit100_at_it1_deletion": at_budget(top["audit"]),
            "probe_heldout": probe_report(top["probe"], t),
            "edge_heldout": edge_new(top["edge"], t),
            "redactor_check": redactor_check(model_dir, top["audit"], t),
        }
        write_json(P0 / "eval-phase0.json", results)
    write_json(P0 / "eval-phase0.json", results)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
