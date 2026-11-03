"""One-day proof, iteration 2: package the calibration, salute, rulings and edge units.

- Calibration slice: groups used by no training (not in v1's replay, not in
  iteration 1's Sol train units). v1 fires there, a sample of non-firing lexical
  hits (to catch misses), and a random population sample for the deletion cap.
- Training additions (other groups): salute cues x person cues across the whole
  frame, and rulings-style hard negatives (deceased, negated record, company
  seizure, hypothetical quesiti).
- The whole edge suite (fictional) for relabeling under the current policy.
Writes `iter2-units.json` for the labeler and private mappings; prints counts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import sentence_spans
from scripts.proof_package import doc_paths, sentences_scored, unit
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    V1,
    build_frame,
    edge_rows,
    encode,
    load_model,
    opaque,
    score_ids,
    v1_thresholds,
    write_json,
)

HEALTH = re.compile(
    r"malatti|invalid|disabil|handicap|patolog|ricover|terapi|diagnos|infermit|psichiatr|"
    r"tumor|oncolog|demenz|alzheimer|parkinson|autosufficien|l(?:egge|\.)\s*104|"
    r"accompagnament|degenz|ospedal|casa di cura|deambul|carrozzin|sedia a rotelle|"
    r"\bcure\b|\bmalat",
    re.I,
)
PERSON = re.compile(
    r"sig\.|signor|esecutat|debitor|propriet|occupant|conduttor|inquilin|figli|coniug|"
    r"moglie|marito|madre|padre|\[nome\]|omissis|anzian|persona|nat[oa] ",
    re.I,
)
RULINGS = {
    "deceased": re.compile(r"decedut|defunt|de cuius|eredi\b|successione", re.I),
    "negated_record": re.compile(
        r"(non risult|nessun|assenz|negativ)[^.]{0,80}(penal|carich|procediment|precedent)", re.I
    ),
    "company_seizure": re.compile(
        r"(sequestr|confisc)[^.]{0,120}(societ|s\.?r\.?l|s\.?p\.?a|ditta)", re.I
    ),
    "hypothetical": re.compile(r"quesito|offerent|eventual|ove (?:il|la)|qualora", re.I),
}
PER_RULINGS = 100
SALUTE_TRAIN = 250
CAL_FAMIGLIA_CAP = 250
CAL_MISS_SAMPLE = 150
POPULATION = 4000


def replay_groups() -> set[str]:
    path = ROOT / "development-preparation-v3/training-inputs-v3/train.jsonl"
    with path.open() as stream:
        return {json.loads(line)["doc_id"] for line in stream}


def target_of(text: str) -> str:
    return text[text.index("[TARGET]") + 8 : text.index("[/TARGET]")]


def edge_units() -> list[dict[str, Any]]:
    out = []
    for row in edge_rows():
        text = row["paragraph"]
        start = text.index(row["target"])
        spans = sentence_spans(text)
        idx = [i for i, (a, b) in enumerate(spans) if a < start + len(row["target"]) and b > start]
        clean = [" ".join(text[a:b].split()) for a, b in spans]
        out.append(
            {
                "unit_id": "edge-" + opaque("edge", row["id"]),
                "before": clean[max(0, idx[0] - 2) : idx[0]],
                "target": " ".join(row["target"].split()),
                "after": clean[idx[-1] + 1 : idx[-1] + 3],
                "_row": row["id"],
            }
        )
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rng = random.Random(20261002)
    thr = v1_thresholds()
    appraisal = build_frame()["appraisal"]
    iter1 = json.loads((OUT / "units-private.json").read_text())
    used = replay_groups() | {u["group"] for u in iter1 if not u["dev"]}
    free = set(appraisal.values()) - used
    done = set()
    with (OUT / "scores.jsonl").open() as stream:
        done = {json.loads(line)["id"] for line in stream}

    # Stream the pool once: lexical hits and population sample in free groups,
    # salute cue hits everywhere.
    rows: dict[str, dict[str, Any]] = {}
    population: list[str] = []
    seen_pop = 0
    salute_hits: set[str] = set()
    with (ROOT / "corpus-v1/train_pool.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["source_doc_id"] not in appraisal:
                continue
            target = target_of(r["text"])
            keep = {k: r[k] for k in ("id", "doc_id", "source_doc_id")}
            keep |= {"ss": r["sentence_start"], "se": r["sentence_end"], "text": r["text"]}
            keep["mined"] = [h for h in HEADS if h in r["mined_labels"]]
            is_salute = bool(HEALTH.search(target) and PERSON.search(r["text"]))
            if is_salute:
                salute_hits.add(r["id"])
            if r["doc_id"] in free:
                if any(h in r["mined_labels"] for h in HEADS) or is_salute:
                    rows[r["id"]] = keep
                seen_pop += 1
                if len(population) < POPULATION:
                    population.append(r["id"])
                    rows[r["id"]] = keep
                elif (j := rng.randrange(seen_pop)) < POPULATION:
                    old = rows[population[j]]
                    if not old["mined"] and population[j] not in salute_hits:
                        del rows[population[j]]
                    population[j] = r["id"]
                    rows[r["id"]] = keep
            elif is_salute:
                rows[r["id"]] = keep
    # Score what iteration 1 did not, with v1; append to the same score file.
    meta = {r["id"]: r for r in json.loads((OUT / "candidates.json").read_text())}
    todo = [r for i, r in rows.items() if i not in done]
    torch, tok, model = load_model(V1)
    encoded = [encode(tok, r["text"]) for r in todo]
    with (OUT / "scores.jsonl").open("a") as stream:
        for idx, s in score_ids(torch, model, tok.pad_token_id, encoded):
            for i, row in zip(idx, s, strict=True):
                stream.write(
                    json.dumps({"id": todo[i]["id"], "s": [round(float(v), 5) for v in row]}) + "\n"
                )
    for r in rows.values():
        meta.setdefault(
            r["id"],
            {k: r[k] for k in ("id", "doc_id", "source_doc_id", "ss", "se", "mined")},
        )
    write_json(OUT / "candidates.json", list(meta.values()))

    sents = sentences_scored()
    sent_of = {i: (r["source_doc_id"], r["ss"], r["se"]) for i, r in rows.items()}
    salute_sents = {sent_of[i] for i in salute_hits if i in sent_of}
    pop_sents = {sent_of[i] for i in population}
    iter1_keys = {(u["source_doc_id"], u["ss"], u["se"]) for u in iter1}
    paths = doc_paths()
    texts: dict[str, str] = {}

    def text_of(src: str) -> str:
        return texts.setdefault(src, Path(paths[src]).read_text(encoding="utf-8"))

    def fires(key: tuple[str, int, int]) -> bool:
        return any(s >= t for s, t in zip(sents[key]["s"], thr, strict=True))

    chosen: dict[tuple[str, int, int], dict[str, Any]] = {}
    ok = {k for k, v in sents.items() if k[2] - k[1] <= 1500 and k not in iter1_keys}

    # Calibration slice (free groups only).
    cal = [k for k in ok if sents[k]["group"] in free]
    fam = [k for k in cal if fires(k) and max(sents[k]["s"]) == sents[k]["s"][3]]
    rng.shuffle(fam)
    fam_drop = set(fam[CAL_FAMIGLIA_CAP:])
    for k in cal:
        if (fires(k) and k not in fam_drop) or k in salute_sents:
            chosen[k] = {"set": "cal", "kind": "cal_fire" if fires(k) else "cal_salute_cue"}
    misses = [k for k in cal if k not in chosen and sents[k]["mined"]]
    for k in rng.sample(misses, min(CAL_MISS_SAMPLE, len(misses))):
        chosen[k] = {"set": "cal", "kind": "cal_lexical_nonfire"}

    # Training additions (groups outside the calibration slice).
    train = [k for k in ok if sents[k]["group"] not in free and k not in chosen]
    sal = sorted((k for k in train if k in salute_sents), key=lambda k: -sents[k]["s"][0])
    per_group: Counter[str] = Counter()
    for k in sal:
        if sum(c["kind"] == "salute_cue" for c in chosen.values()) >= SALUTE_TRAIN:
            break
        if per_group[sents[k]["group"]] < 3:
            per_group[sents[k]["group"]] += 1
            chosen[k] = {"set": "train", "kind": "salute_cue"}
    rng.shuffle(train)
    train.sort(key=lambda k: not fires(k))
    filled: Counter[str] = Counter()
    hard_group: Counter[str] = Counter()
    for k in train:
        if k in chosen or hard_group[sents[k]["group"]] >= 2:
            continue
        target = text_of(k[0])[k[1] : k[2]]
        bucket = next((b for b, rx in RULINGS.items() if rx.search(target)), None)
        if bucket and filled[bucket] < PER_RULINGS:
            filled[bucket] += 1
            hard_group[sents[k]["group"]] += 1
            chosen[k] = {"set": "train", "kind": "rulings_" + bucket}

    units, private = [], []
    for (src, ss, se), c in chosen.items():
        spans = sentence_spans(text_of(src))
        uid = opaque("iter2", src, ss, se)
        units.append({"unit_id": uid, **unit(text_of(src), spans, spans.index((ss, se)))})
        private.append(
            {
                "unit_id": uid,
                "source_doc_id": src,
                "group": sents[(src, ss, se)]["group"],
                "ss": ss,
                "se": se,
                **c,
            }
        )
    edges = edge_units()
    edge_map = {u["unit_id"]: u.pop("_row") for u in edges}
    units += edges
    rng.shuffle(units)
    write_json(OUT / "iter2-units.json", units)
    write_json(OUT / "iter2-units-private.json", private)
    write_json(OUT / "iter2-edge-map.json", edge_map)
    write_json(
        OUT / "iter2-population.json",
        [{"source_doc_id": k[0], "ss": k[1], "se": k[2]} for k in pop_sents],
    )
    summary = {
        "free_groups": len(free),
        "units": len(units),
        "edge_units": len(edges),
        "kinds": Counter(p["kind"] for p in private),
        "population_sentences": len(pop_sents),
        "salute_cue_windows": len(salute_hits),
        "newly_scored_windows": len(todo),
        "groups_per_set": {
            s: len({p["group"] for p in private if p["set"] == s}) for s in ("cal", "train")
        },
    }
    write_json(OUT / "iter2-package-summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()


def calibration_groups() -> set[str]:
    """Groups held out of iteration 2 training for per-head calibration.

    The unused groups alone hold almost no penale/minori positives, so add
    iteration 1's dev groups and then whole iteration 1 train groups (seeded
    order) until penale and minori reach 30 Sol-confirmed positives each.
    Iteration 1 trained on the units in those added groups.
    """
    units1 = json.loads((OUT / "units-private.json").read_text())
    labels1 = json.loads((OUT / "labels.json").read_text())
    units2 = json.loads((OUT / "iter2-units-private.json").read_text())
    labels2 = json.loads((OUT / "iter2-labels.json").read_text())
    groups = {u["group"] for u in units2 if u["set"] == "cal"} | {
        u["group"] for u in units1 if u["dev"]
    }
    pos: dict[str, Counter[str]] = {}
    for units, labels in ((units1, labels1), (units2, labels2)):
        for u in units:
            lab = labels[u["unit_id"]]
            if not lab["uncertain"]:
                pos.setdefault(u["group"], Counter()).update(lab["heads"])

    def total(head: str) -> int:
        return sum(pos.get(g, Counter())[head] for g in groups)

    order = sorted({u["group"] for u in units1 if not u["dev"]})
    random.Random(1134).shuffle(order)
    for g in order:
        need = [h for h in ("penale", "minori") if total(h) < 30]
        if not need:
            break
        if any(pos.get(g, Counter())[h] for h in need):
            groups.add(g)
    return groups
