"""One-day proof, step 3: package labeling units from v1 scores.

Per head: the top v1-scored sentences, at most 3 per group. Hard negatives:
v1 fires from varied groups across a few surface buckets, plus neighbours of
the positive candidates. A unit is the target sentence with two sentences of
context on each side under an opaque ID. Prints counts only.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import example_text, sentence_spans, windows
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    V1,
    load_model,
    opaque,
    score_texts,
    v1_thresholds,
    write_json,
)

PER_HEAD = 150
PER_GROUP = 3
HARD_QUOTAS = {"deceased": 110, "negation": 110, "notice": 150, "penale_vocab": 150, "other": 130}
NEIGHBOURS = 150
# Surface buckets are used only to diversify the hard-negative sample.
BUCKETS = {
    "deceased": re.compile(r"decedut|defunt|de cuius|success|eredit", re.I),
    "negation": re.compile(r"\bnon risult|\bnessun|\bassenz|\bnegativ|\bnon (?:vi )?sono", re.I),
    "notice": re.compile(r"offerent|aggiudicatar|avviso|partecipa|delegato|cauzione", re.I),
    "penale_vocab": re.compile(r"penal|sequestr|reat|procura|condann|arrest", re.I),
}
DEV_MOD = 5  # one group in five is held out as the threshold-calibration slice


def is_dev(group: str) -> bool:
    return int(opaque("dev", group), 16) % DEV_MOD == 0


def doc_paths() -> dict[str, str]:
    documents = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    return {d["source_doc_id"]: d["path"] for d in documents}


def sentences_scored() -> dict[tuple[str, int, int], dict[str, Any]]:
    meta = {r["id"]: r for r in json.loads((OUT / "candidates.json").read_text())}
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    with (OUT / "scores.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            m = meta.get(r["id"])
            if m is None:  # scored under an earlier, larger per-head cap
                continue
            key = (m["source_doc_id"], m["ss"], m["se"])
            cur = out.setdefault(key, {"group": m["doc_id"], "s": [0.0] * 4, "mined": set()})
            cur["s"] = [max(a, b) for a, b in zip(cur["s"], r["s"][:4], strict=True)]
            cur["mined"] |= set(m["mined"])
    return out


def unit(text: str, spans: list[tuple[int, int]], i: int) -> dict[str, Any]:
    def clean(k: int, cap: int) -> str:
        return " ".join(text[spans[k][0] : spans[k][1]].split())[:cap]

    return {
        "before": [clean(k, 400) for k in range(max(0, i - 2), i)],
        "target": clean(i, 1500),
        "after": [clean(k, 400) for k in range(i + 1, min(len(spans), i + 3))],
    }


def main() -> None:
    rng = random.Random(1134)
    thr = v1_thresholds()
    # Targets longer than the unit cap would be shown truncated; leave them out.
    sents = {k: v for k, v in sentences_scored().items() if k[2] - k[1] <= 1500}
    paths = doc_paths()
    chosen: dict[tuple[str, int, int], dict[str, Any]] = {}
    per_group: Counter[str] = Counter()

    for h, head in enumerate(HEADS):
        ranked = sorted(sents.items(), key=lambda kv: -kv[1]["s"][h])
        taken = 0
        for key, v in ranked:
            if taken == PER_HEAD:
                break
            if key in chosen or per_group[(head, v["group"])] >= PER_GROUP:
                continue
            per_group[(head, v["group"])] += 1
            chosen[key] = {"kind": "positive_candidate", "head": head, **v}
            taken += 1

    # Neighbours of positive candidates: score with v1, prefer the ones it fires on.
    torch, tok, model = load_model(V1)
    texts: dict[str, str] = {}
    neighbour_rows = []
    for (src, ss, se), v in list(chosen.items()):
        text = texts.setdefault(src, Path(paths[src]).read_text(encoding="utf-8"))
        spans = sentence_spans(text)
        i = spans.index((ss, se))
        for k in (i - 1, i + 1):
            if (
                0 <= k < len(spans)
                and (src, *spans[k]) not in chosen
                and spans[k][1] - spans[k][0] <= 1500
            ):
                neighbour_rows.append(((src, *spans[k]), v["group"]))
    neighbour_rows = list(dict(neighbour_rows).items())
    wins = []
    for n, ((src, ss, se), _) in enumerate(neighbour_rows):
        text = texts[src]
        for a, b, s0, s1 in windows(text):
            if (s0, s1) == (ss, se):
                wins.append((n, example_text(text, a, b)))
    scores = score_texts(torch, tok, model, [w for _, w in wins])
    nmax: dict[int, list[float]] = defaultdict(lambda: [0.0] * 4)
    for (n, _), s in zip(wins, scores, strict=True):
        nmax[n] = [max(a, float(b)) for a, b in zip(nmax[n], s[:4], strict=True)]
    neigh = [
        (key, group, nmax[n], any(s >= t for s, t in zip(nmax[n], thr, strict=True)))
        for n, (key, group) in enumerate(neighbour_rows)
    ]
    rng.shuffle(neigh)
    neigh.sort(key=lambda x: not x[3])
    for key, group, s, _fires in neigh[:NEIGHBOURS]:
        chosen[key] = {"kind": "neighbour", "head": None, "group": group, "s": s, "mined": set()}

    # Bucketed v1 fires from varied groups.
    fires = [
        (key, v)
        for key, v in sents.items()
        if key not in chosen and any(s >= t for s, t in zip(v["s"], thr, strict=True))
    ]
    rng.shuffle(fires)
    hard_group: Counter[str] = Counter()
    filled: Counter[str] = Counter()
    for key, v in fires:
        src, ss, se = key
        text = texts.setdefault(src, Path(paths[src]).read_text(encoding="utf-8"))
        target = text[ss:se]
        bucket = next((b for b, rx in BUCKETS.items() if rx.search(target)), "other")
        if filled[bucket] >= HARD_QUOTAS[bucket] or hard_group[v["group"]] >= 2:
            continue
        filled[bucket] += 1
        hard_group[v["group"]] += 1
        chosen[key] = {"kind": "hard_negative", "head": None, "bucket": bucket, **v}

    units, private = [], []
    for (src, ss, se), v in chosen.items():
        text = texts.setdefault(src, Path(paths[src]).read_text(encoding="utf-8"))
        spans = sentence_spans(text)
        uid = opaque("unit", src, ss, se)
        units.append({"unit_id": uid, **unit(text, spans, spans.index((ss, se)))})
        private.append(
            {
                "unit_id": uid,
                "source_doc_id": src,
                "group": v["group"],
                "ss": ss,
                "se": se,
                "kind": v["kind"],
                "head": v.get("head"),
                "bucket": v.get("bucket"),
                "v1": [round(x, 5) for x in v["s"]],
                "dev": is_dev(v["group"]),
            }
        )
    rng.shuffle(units)
    write_json(OUT / "units.json", units)
    write_json(OUT / "units-private.json", private)
    summary = {
        "units": len(units),
        "kinds": Counter(p["kind"] for p in private),
        "positive_candidates_per_head": Counter(p["head"] for p in private if p["head"]),
        "hard_negative_buckets": filled,
        "neighbours_v1_firing": sum(x[3] for x in neigh[:NEIGHBOURS]),
        "groups": len({p["group"] for p in private}),
        "dev_units": sum(p["dev"] for p in private),
    }
    write_json(OUT / "package-summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
