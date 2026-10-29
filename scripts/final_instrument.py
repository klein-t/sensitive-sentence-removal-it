"""Final step: freeze the evaluation instrument on never-used documents.

Unseen = appraisal documents (any corpus split, never audit100) whose group
appears in no A0/it1/Phase 0 training or calibration input and shares no
12-gram with the fictional edge suite. Groups are split by a fixed hash:

* deletion side: 200 documents, every sentence scored with A0 and it1, binned
  on max(A0, it1) and sampled (2,000) with known inclusion rates, so the
  deletion estimate is inverse-probability weighted to the whole population;
* positives side: cue-mined sentences (salute cue x person cue, the lexical
  miner for the rest) sampled at random per head, plus a tranche of the top
  A0/it1 scores, labelled to find held-out real positives; then the adjacent
  sentences of the confirmed positives, for bleed.

Labels use the frozen Phase 0 decision-tree schema. Deletion units get two
passes (disagreement = uncertain); positive candidates get two (pass 2 on
pass-1 positives/uncertain); neighbours get one. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import MINING, example_text, sentence_spans, windows
from scripts.phase0_label import run_pass
from scripts.proof_iter2 import HEALTH, PERSON
from scripts.proof_package import unit
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    TESTO,
    V1,
    edge_rows,
    grams,
    load_model,
    opaque,
    private_dir,
    score_texts,
    write_json,
)

F = ROOT / "final-20261003"
Key = tuple[str, int, int]
P0 = ROOT / "phase0-20261003"
IT1 = OUT / "model"
CAP = 3_000_000
DELETION_DOCS = 200
DELETION_UNITS = 2000
BINS = (0.0, 0.005, 0.02, 0.1, 0.3, 0.6, 0.9, 1.01)
CUE_SAMPLE = {"salute": 250, "salute_broad": 150, "penale": 150, "minori": 120, "famiglia": 200}
SCORE_TRANCHE = 150
CUE_SCORED = 2000
MAX_CHARS = 1500
TRAINING_INPUTS = (
    "development-preparation-v3/training-inputs-v3/train.jsonl",
    "development-preparation-v3/training-inputs-v3/validation.jsonl",
    "development-preparation-v3/training-inputs-v3/calibration.jsonl",
    "training-inputs-v2/train.jsonl",
    "training-inputs-v2/validation.jsonl",
    "training-inputs-v2/calibration.jsonl",
)


def used_groups() -> set[str]:
    used: set[str] = set()
    for name in TRAINING_INPUTS:
        with (ROOT / name).open() as stream:
            used |= {json.loads(line)["doc_id"] for line in stream if line.strip()}
    group_of = {
        d["source_doc_id"]: d["doc_id"]
        for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }
    for name in ("units-private.json", "iter2-units-private.json", "iter2-population.json"):
        for u in json.loads((OUT / name).read_text()):
            used.add(u.get("group") or group_of[u["source_doc_id"]])
    used |= {p["group"] for p in json.loads((P0 / "probe-private.json").read_text())}
    return used


def unseen_documents() -> tuple[list[dict[str, Any]], dict[str, int]]:
    used = used_groups()
    suite = set().union(*(grams(r["paragraph"]) for r in edge_rows()))
    out, stats = [], Counter()
    for d in json.loads((ROOT / "corpus-v1/documents.json").read_text()):
        if d["audit"] or d["doc_id"] in used:
            continue
        meta = json.loads((TESTO / (Path(d["path"]).stem + ".json")).read_text())
        if meta.get("tipo") not in ("PERIZ", "STIMA"):
            continue
        if grams(Path(d["path"]).read_text(encoding="utf-8")) & suite:
            stats["edge_gram_overlap_dropped"] += 1
            continue
        stats[f"split_{d['split']}"] += 1
        out.append(d)
    return out, dict(stats)


def side(group: str) -> str:
    return "deletion" if int(opaque("final-side", group), 16) % 5 < 3 else "positives"


def sentence_windows(text: str) -> dict[tuple[int, int], list[str]]:
    """v1-format examples per sentence (the A0/it1 encoding)."""
    out: dict[tuple[int, int], list[str]] = defaultdict(list)
    for a, b, s0, s1 in windows(text):
        out[(s0, s1)].append(example_text(text, a, b))
    return out


def score_v1(
    models: dict[str, Path], items: list[tuple[tuple[str, int, int], str]]
) -> dict[str, dict[tuple[str, int, int], list[float]]]:
    """Per-sentence per-head max of A0 and it1 over the sentence's windows."""
    out: dict[str, dict[tuple[str, int, int], list[float]]] = {}
    for name, path in models.items():
        torch, tok, model = load_model(path)
        s = score_texts(torch, tok, model, [t for _, t in items])
        cur: dict[tuple[str, int, int], list[float]] = {}
        for (key, _), row in zip(items, s, strict=True):
            prev = cur.get(key, [0.0] * 4)
            cur[key] = [max(a, float(b)) for a, b in zip(prev, row[:4], strict=True)]
        out[name] = cur
        del model
        torch.mps.empty_cache()
    return out


def build() -> None:
    private_dir(F)
    rng = random.Random(20261004)
    docs, stats = unseen_documents()
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_side[side(d["doc_id"])].append(d)
    rng.shuffle(by_side["deletion"])
    seen_groups: set[str] = set()
    deletion_docs = []
    for d in by_side["deletion"]:
        if d["doc_id"] not in seen_groups and len(deletion_docs) < DELETION_DOCS:
            seen_groups.add(d["doc_id"])
            deletion_docs.append(d)
    texts = {d["source_doc_id"]: Path(d["path"]).read_text(encoding="utf-8") for d in docs}
    groups = {d["source_doc_id"]: d["doc_id"] for d in docs}

    # Deletion population: every sentence of the sampled documents.
    items, population, long_skipped = [], [], 0
    for d in deletion_docs:
        src = d["source_doc_id"]
        for (s0, s1), examples in sentence_windows(texts[src]).items():
            if not texts[src][s0:s1].strip():
                continue
            if s1 - s0 > MAX_CHARS:
                long_skipped += 1
                continue
            population.append((src, s0, s1))
            items.extend(((src, s0, s1), e) for e in examples)
    if (F / "population-scores.json").exists():  # scoring is the slow part; reuse it
        saved = json.loads((F / "population-scores.json").read_text())
        scores = {m: {tuple(r[0]): r[c] for r in saved} for m, c in (("A0", 1), ("it1", 2))}
    else:
        scores = score_v1({"A0": V1, "it1": IT1}, items)
        write_json(
            F / "population-scores.json",
            [[list(k), scores["A0"][k], scores["it1"][k]] for k in population],
        )
    bins: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for k in population:
        m = max(*scores["A0"][k], *scores["it1"][k])
        bins[next(i for i in range(len(BINS) - 1) if m < BINS[i + 1])].append(k)
    alloc, left = {}, DELETION_UNITS
    for i in sorted(bins, key=lambda b: len(bins[b])):
        alloc[i] = min(len(bins[i]), left // (len(bins) - len(alloc)))
        left -= alloc[i]
    sample = []
    for i, keys in bins.items():
        for k in rng.sample(keys, alloc[i]):
            sample.append({"key": k, "bin": i, "weight": len(keys) / alloc[i]})

    # Positives side: cue-mined candidates, plus the top A0/it1 tranche.
    cue_hits: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for d in by_side["positives"]:
        src, text = d["source_doc_id"], texts[d["source_doc_id"]]
        for s0, s1 in sentence_spans(text):
            target = text[s0:s1]
            if s1 - s0 > MAX_CHARS or not target.strip():
                continue
            if HEALTH.search(target) and PERSON.search(target):
                cue_hits["salute"].append((src, s0, s1))
            elif MINING["salute"].search(target):
                cue_hits["salute_broad"].append((src, s0, s1))
            for h in ("penale", "minori", "famiglia"):
                if MINING[h].search(target):
                    cue_hits[h].append((src, s0, s1))
    chosen: dict[tuple[str, int, int], str] = {}
    scored_keys: set[tuple[str, int, int]] = set()
    for h in CUE_SAMPLE:
        rng.shuffle(cue_hits[h])
        for k in cue_hits[h][: CUE_SAMPLE[h]]:
            chosen.setdefault(k, f"cue_{h}")
        scored_keys |= set(cue_hits[h][:CUE_SCORED])
    cand_items = []
    by_src: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for src, s0, s1 in scored_keys:
        by_src[src].add((s0, s1))
    for src, wanted in sorted(by_src.items()):
        for span, examples in sentence_windows(texts[src]).items():
            if span in wanted:
                cand_items.extend(((src, *span), e) for e in examples)
    cs = score_v1({"A0": V1, "it1": IT1}, cand_items)
    ranked = sorted(scored_keys - set(chosen), key=lambda k: -max(*cs["A0"][k], *cs["it1"][k]))
    for k in ranked[:SCORE_TRANCHE]:
        chosen[k] = "score_tranche"

    def to_unit(key: tuple[str, int, int], prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
        src, s0, s1 = key
        spans = sentence_spans(texts[src])
        uid = opaque(prefix, src, s0, s1)
        return {"unit_id": uid, **unit(texts[src], spans, spans.index((s0, s1)))}, {
            "unit_id": uid,
            "source_doc_id": src,
            "group": groups[src],
            "ss": s0,
            "se": s1,
        }

    del_units, del_private = [], []
    for s in sample:
        u, p = to_unit(tuple(s["key"]), "final-del")
        del_units.append(u)
        del_private.append(p | {"bin": s["bin"], "weight": s["weight"]})
    pos_units, pos_private = [], []
    for k, origin in chosen.items():
        u, p = to_unit(k, "final-pos")
        pos_units.append(u)
        pos_private.append(p | {"origin": origin})
    write_json(F / "deletion-units.json", del_units)
    write_json(F / "deletion-private.json", del_private)
    write_json(F / "positive-units.json", pos_units)
    write_json(F / "positive-private.json", pos_private)
    summary = stats | {
        "unseen_documents": len(docs),
        "deletion_documents": len(deletion_docs),
        "positives_side_documents": len(by_side["positives"]),
        "population_sentences": len(population),
        "population_long_sentences_skipped": long_skipped,
        "bins": {f"[{BINS[i]},{BINS[i + 1]})": [len(bins[i]), alloc[i]] for i in sorted(bins)},
        "deletion_units": len(del_units),
        "cue_hits": {h: len(v) for h, v in cue_hits.items()},
        "positive_candidates": Counter(chosen.values()),
    }
    write_json(F / "instrument-summary.json", summary)
    print(json.dumps(summary, indent=1))


def consensus(a: dict[str, Any], b: dict[str, Any] | None) -> dict[str, Any]:
    """Two passes agree on remove heads and neither is uncertain, else uncertain."""
    agree = (
        b is not None
        and not a["uncertain"]
        and not b["uncertain"]
        and sorted(a["remove_heads"]) == sorted(b["remove_heads"])
    )
    return {"remove_heads": sorted(a["remove_heads"]), "uncertain": not agree}


def neighbours_of(
    positives: dict[str, dict[str, Any]], private: list[dict[str, Any]], exclude: set[Key]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Units for the sentences adjacent to confirmed positives (not candidates)."""
    paths = {
        d["source_doc_id"]: d["path"]
        for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }
    by_id = {p["unit_id"]: p for p in private}
    taken = exclude | {(p["source_doc_id"], p["ss"], p["se"]) for p in private}
    units, rows = [], []
    for uid, v in positives.items():
        if v["uncertain"] or not v["remove_heads"] or uid not in by_id:
            continue
        p = by_id[uid]
        text = Path(paths[p["source_doc_id"]]).read_text(encoding="utf-8")
        spans = sentence_spans(text)
        i = spans.index((p["ss"], p["se"]))
        for k in (i - 1, i + 1):
            if not 0 <= k < len(spans):
                continue
            s0, s1 = spans[k]
            key = (p["source_doc_id"], s0, s1)
            if key in taken or s1 - s0 > MAX_CHARS or not text[s0:s1].strip():
                continue
            taken.add(key)
            nid = opaque("final-neigh", *key)
            units.append({"unit_id": nid, **unit(text, spans, k)})
            rows.append({"unit_id": nid, "source_doc_id": key[0], "ss": s0, "se": s1})
    return units, rows


def extend(size: int) -> None:
    """Second mining round on the positives side when the first found < 300."""
    ledger = json.loads((F / "sol-ledger.json").read_text())
    rng = random.Random(20261005)
    docs, _ = unseen_documents()
    pos_docs = [d for d in docs if side(d["doc_id"]) == "positives"]
    texts = {d["source_doc_id"]: Path(d["path"]).read_text(encoding="utf-8") for d in pos_docs}
    groups = {d["source_doc_id"]: d["doc_id"] for d in pos_docs}
    private = json.loads((F / "positive-private.json").read_text())
    taken = {(p["source_doc_id"], p["ss"], p["se"]) for p in private}
    taken |= {
        (p["source_doc_id"], p["ss"], p["se"])
        for p in json.loads((F / "neighbour-private.json").read_text())
    }
    hits: dict[str, list[Key]] = defaultdict(list)
    for src, text in texts.items():
        for s0, s1 in sentence_spans(text):
            target = text[s0:s1]
            if s1 - s0 > MAX_CHARS or not target.strip() or (src, s0, s1) in taken:
                continue
            for h in ("salute", "penale", "minori", "famiglia"):
                if MINING[h].search(target):
                    hits[h].append((src, s0, s1))
    rng.shuffle(hits["minori"])
    scored = set(hits["salute"] + hits["penale"] + hits["famiglia"] + hits["minori"][:CUE_SCORED])
    by_src: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for src, s0, s1 in scored:
        by_src[src].add((s0, s1))
    items = [
        ((src, *span), e)
        for src, wanted in sorted(by_src.items())
        for span, examples in sentence_windows(texts[src]).items()
        if span in wanted
        for e in examples
    ]
    cs = score_v1({"A0": V1, "it1": IT1}, items)
    ranked = sorted(scored, key=lambda k: -max(*cs["A0"][k], *cs["it1"][k]))
    chosen = dict.fromkeys(ranked[:SCORE_TRANCHE], "score_tranche")
    fam = [k for k in hits["famiglia"] if k not in chosen]
    rng.shuffle(fam)
    chosen |= dict.fromkeys(fam[:150], "cue_famiglia")
    new_units, new_private = [], []
    for (src, s0, s1), origin in chosen.items():
        spans = sentence_spans(texts[src])
        uid = opaque("final-pos", src, s0, s1)
        new_units.append({"unit_id": uid, **unit(texts[src], spans, spans.index((s0, s1)))})
        new_private.append(
            {
                "unit_id": uid,
                "source_doc_id": src,
                "group": groups[src],
                "ss": s0,
                "se": s1,
                "origin": origin,
                "round": 2,
            }
        )
    p1 = run_pass("P1b", new_units, size, ledger, F, CAP)
    flagged = [
        u for u in new_units if p1[u["unit_id"]]["remove_heads"] or p1[u["unit_id"]]["uncertain"]
    ]
    p2 = run_pass("P2b", flagged, size, ledger, F, CAP)
    found = {
        uid: consensus(v, p2.get(uid)) if (v["remove_heads"] or v["uncertain"]) else v
        for uid, v in p1.items()
    }
    neigh_units, neigh_private = neighbours_of(found, new_private, taken)
    n1 = run_pass("N1b", neigh_units, size, ledger, F, CAP)
    for name, extra in (
        ("positive-units.json", new_units),
        ("positive-private.json", new_private),
        ("neighbour-units.json", neigh_units),
        ("neighbour-private.json", neigh_private),
    ):
        write_json(F / name, json.loads((F / name).read_text()) + extra)
    for name, extra in (("labels-positives.json", found), ("labels-neighbours.json", n1)):
        write_json(F / name, json.loads((F / name).read_text()) | extra)
    conf = [
        v
        for v in json.loads((F / "labels-positives.json").read_text()).values()
        if not v["uncertain"] and v["remove_heads"]
    ]
    print(
        json.dumps(
            {
                "sol_tokens": ledger["tokens"],
                "round2_candidates": len(new_units),
                "round2_positives": sum(
                    bool(v["remove_heads"]) and not v["uncertain"] for v in found.values()
                ),
                "held_out_positives": len(conf),
                "per_head": {h: sum(h in v["remove_heads"] for v in conf) for h in HEADS},
                "neighbours_added": len(n1),
            }
        )
    )


def label(size: int) -> None:
    ledger_path = F / "sol-ledger.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"tokens": 0, "batches": []}
    )
    dels = json.loads((F / "deletion-units.json").read_text())
    pos = json.loads((F / "positive-units.json").read_text())
    p1 = run_pass("P1", pos, size, ledger, F, CAP)
    flagged = [u for u in pos if p1[u["unit_id"]]["remove_heads"] or p1[u["unit_id"]]["uncertain"]]
    p2 = run_pass("P2", flagged, size, ledger, F, CAP)
    positives = {
        uid: consensus(v, p2.get(uid)) if (v["remove_heads"] or v["uncertain"]) else v
        for uid, v in p1.items()
    }
    write_json(F / "labels-positives.json", positives)
    d1 = run_pass("D1", dels, size, ledger, F, CAP)
    d2 = run_pass("D2", dels, size, ledger, F, CAP)
    write_json(F / "labels-deletion.json", {u: consensus(v, d2.get(u)) for u, v in d1.items()})
    write_json(F / "labels-deletion-passes.json", {"D1": d1, "D2": d2})

    # Neighbours of confirmed positives, one pass.
    private = json.loads((F / "positive-private.json").read_text())
    neigh_units, neigh_private = neighbours_of(positives, private, set())
    write_json(F / "neighbour-units.json", neigh_units)
    write_json(F / "neighbour-private.json", neigh_private)
    n1 = run_pass("N1", neigh_units, size, ledger, F, CAP)
    write_json(F / "labels-neighbours.json", n1)
    conf = [v for v in positives.values() if not v["uncertain"] and v["remove_heads"]]
    dl = json.loads((F / "labels-deletion.json").read_text())
    print(
        json.dumps(
            {
                "sol_tokens": ledger["tokens"],
                "positive_candidates": len(pos),
                "held_out_positives": len(conf),
                "held_out_positives_per_head": {
                    h: sum(h in v["remove_heads"] for v in conf) for h in HEADS
                },
                "positive_candidates_uncertain": sum(v["uncertain"] for v in positives.values()),
                "deletion_units": len(dl),
                "deletion_uncertain": sum(v["uncertain"] for v in dl.values()),
                "deletion_positive": sum(
                    bool(v["remove_heads"]) and not v["uncertain"] for v in dl.values()
                ),
                "neighbours": len(n1),
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "label", "extend"))
    ap.add_argument("--size", type=int, default=150)
    args = ap.parse_args()
    {"build": build, "label": lambda: label(args.size), "extend": lambda: extend(args.size)}[
        args.step
    ]()


if __name__ == "__main__":
    main()
