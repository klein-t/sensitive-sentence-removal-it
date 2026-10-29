"""Clean rare-class test set: cue-only mining on never-used documents.

The held-out positives of the final instrument are almost all famiglia and were
partly mined with model scores. This builds a test set for salute, penale and
minori without any model score:

* frame: validation and test appraisal documents whose group appears in no
  earlier artifact (v1/it1/it2/Phase 0/Run A training, calibration and
  validation slices, the final instrument, audit100/audit15, the v1 stratified
  evaluation, any other non-catalogue file), shares no 12-gram with the edge
  suite, and is not in the quarantined calibration file;
* mine: broad Italian cue lists per class, every hit (a uniform sample capped at
  3 sentences per group only above PER_CLASS), plus a uniform sample of
  sentences with no cue;
* label: two independent Sol passes with the Phase 0 schema on every unit;
  disagreement or either pass uncertain = uncertain;
* score: A0 at its production thresholds, only after labelling.

Prints aggregates only; unit text and Sol events stay in the private directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import sentence_spans
from scripts.final_instrument import MAX_CHARS, score_v1, sentence_windows
from scripts.phase0_label import run_pass
from scripts.proof_package import unit
from scripts.proof_score import (
    HEADS,
    ROOT,
    TESTO,
    V1,
    edge_rows,
    grams,
    opaque,
    private_dir,
    v1_thresholds,
    write_json,
)

C = ROOT / "clean-20261008"
CAP = 2_000_000
CLASSES = ("salute", "penale", "minori")
PER_CLASS = 400  # take every hit up to this; above it, sample with the group cap
PER_GROUP = 3
NON_CUE = 300
Key = tuple[str, int, int]

CUES = {
    "salute": re.compile(
        r"salut|sanitar|malatt|malat[oaie]\b|ammalat|infermit|invalid|inabil|disabil|handicap|"
        r"l(?:egge|\.)\s*(?:n\.?\s*)?104|non\s+autosufficien|patolog|diagnos|terapi|cur[ae]\b|"
        r"ricover|degenz|ospedal|clinic|casa\s+di\s+(?:cura|riposo)|\brsa\b|hospice|"
        r"psichiatr|psicolog|depress|demenz|alzheimer|parkinson|ictus|infart|cardiopat|tumor|"
        r"cancro|oncolog|chemioterap|dialisi|diabet|autis|tossicodipend|alcolis|sieroposit|"
        r"\bhiv\b|gravidanz|gravid|incinta|gestant|deambul|carrozzin|sedia\s+a\s+rotelle|"
        r"montascal|ausili|ciec[oa]|non\s+vedente|sord[oa]|ipoacus|accompagnament|badante|"
        r"assistenza\s+(?:domiciliar|sanitar|continuativ)|amministrazione\s+di\s+sostegno|"
        r"amministratore\s+di\s+sostegno|interdett|inabilitat|certificat\w*\s+medic|"
        r"medic[oi]\b|\basl\b|\basp\b|intervento\s+chirurg|operat[oa]\s+(?:al|di)|"
        r"decedut\w*\s+(?:a\s+causa|per)|caus[ae]\s+del\s+decesso|morte\s+per|"
        r"pension\w*\s+di\s+invalidit|convalescen|riabilitaz",
        re.I,
    ),
    "penale": re.compile(
        r"penal|\breat[oi]\b|delitt|crimin|confisc|sequestr\w*\s+(?:preventiv|penal|probator)|"
        r"antimafia|misur\w*\s+di\s+prevenzione|159\s*/\s*2011|\banbsc\b|condann|imputat|"
        r"indagat|rinvi\w*\s+a\s+giudizio|arrest|domiciliar|detenut|detenzion|carcer|"
        r"casa\s+circondarial|penitenziar|reclusion|procura\s+della\s+repubblica|"
        r"pubblico\s+ministero|\bp\.?\s?m\.?\b|\bg\.?i\.?p\.?\b|\bg\.?u\.?p\.?\b|casellario|"
        r"carichi\s+pendenti|c\.?\s?p\.?\s?p\.?|codice\s+di\s+procedura\s+penale|"
        r"art\.?\s*(?:321|633|640|644|648)|querel|denunci|carabinier|questura|"
        r"guardia\s+di\s+finanza|polizia|usur[ae]\b|estorsi|truff|riciclagg|bancarott|"
        r"stupefacent|spaccio|omicid|furt[oi]\b|rapin|maltrattament|stalking|violenz|"
        r"ordine\s+di\s+protezione|patteggiament|misur\w*\s+cautelar|latitan|evasion|"
        r"invasione\s+di\s+(?:terreni|edifici)|corruzion|frode|peculato|sorveglianza",
        re.I,
    ),
    "minori": re.compile(
        r"minor[ei]\b|minoren|figli|figlio|figlia|bambin|neonat|ragazz|adolescen|infanzi|"
        r"infant|prole|affid|tutela\s+(?:dei\s+)?minor|tutor[ei]|genitor|potest[aà]|"
        r"responsabilit[aà]\s+genitorial|tribunale\s+(?:per\s+i\s+)?minorenni|"
        r"servizi\s+sociali|scuol|scolastic|asilo|studente|alunn|nipot|"
        r"nat[oa]\s+(?:a\s+\S+\s+)?(?:il|nel)\s+[^.]{0,20}20[0-2]\d|"
        r"\b(?:[1-9]|1[0-7])\s+anni\b|\bet[aà]\b|collocament|casa\s+familiare",
        re.I,
    ),
}

# Files that list every corpus document (inventories, ledgers, splits) are
# catalogues, not uses; any other file that names a group counts as a use.
CATALOGUES = (
    "retraining-20260928-phase1-v1/access-ledger.json",
    "retraining-20260928-phase3-preparation-v1/split.json",
    "retraining-20260928-phase3-preparation-v2/split.json",
    "recovery-20260929-step1-v1/case-input-records.json",
    "recovery-20260929-original-admission-v1/case-input-records.json",
    "recovery-20260929-sources-v2/inventory.json",
    "recovery-20260929-sources-v3/inventory.json",
)
CATEGORIES = (
    (
        "training_v1_it1_it2_phase0_runA",
        (
            "training-inputs-v2",
            "development-preparation-v3*",
            "proof-20261001",
            "phase0-20261003",
            "train-*",
            "student-*",
            "v2-train-v1",
        ),
    ),
    ("final_instrument", ("final-20261003",)),
    ("audit100_audit15", ("*audit100*", "*audit15*", "*document-audit*", "audit*")),
    (
        "calibration_validation_slices",
        ("*validation*", "*calibration*", "references-*", "*-mining-*", "annotations-*"),
    ),
    ("v1_stratified_final_evaluation", ("final-*",)),
)
LEDGER_EVENTS = {
    "a0_validation": "calibration_validation_slices",
    "historical_final_reference": "v1_stratified_final_evaluation",
    "audit100_reference": "audit100_audit15",
}
HEX64 = re.compile(rb"[0-9a-f]{64}")


def category(top: str) -> str:
    for name, patterns in CATEGORIES:
        if any(fnmatch.fnmatch(top, p) for p in patterns):
            return name
    return "other_prior_use"


def wilson(k: int, n: int, z: float = 1.959964) -> str:
    if n == 0:
        return "0/0"
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.1%} [{max(0.0, centre - half):.1%}, {min(1.0, centre + half):.1%}]"


def cue_classes(text: str) -> list[str]:
    return [c for c in CLASSES if CUES[c].search(text)]


def sample_capped(
    keys: list[Key], n: int, group_of: dict[str, str], rng: random.Random
) -> list[Key]:
    """All keys when they fit in n, else a uniform sample with PER_GROUP per group."""
    if len(keys) <= n:
        return keys[:]
    keys = keys[:]
    rng.shuffle(keys)
    per: Counter[str] = Counter()
    out = []
    for k in keys:
        if len(out) == n:
            break
        if per[group_of[k[0]]] < PER_GROUP:
            per[group_of[k[0]]] += 1
            out.append(k)
    return out


def consensus(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Certain only when both passes are certain and agree on the removed heads."""
    agree = (
        a is not None
        and b is not None
        and not a["uncertain"]
        and not b["uncertain"]
        and sorted(a["remove_heads"]) == sorted(b["remove_heads"])
    )
    heads = sorted(a["remove_heads"]) if a is not None else []
    return {"remove_heads": heads, "uncertain": not agree}


# --- frame -------------------------------------------------------------------


def prior_use(documents: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Validation/test groups per prior-use category."""
    ev = [d for d in documents if d["split"] != "train"]
    ids = {d[k]: d["doc_id"] for d in ev for k in ("doc_id", "source_doc_id", "content_hash")}
    used: dict[str, set[str]] = defaultdict(set)
    for d in ev:
        if d["audit"]:
            used["audit100_audit15"].add(d["doc_id"])
    ledger = json.loads((ROOT / CATALOGUES[0]).read_text())
    for r in ledger:
        for e in r["exposure_events"]:
            if e in LEDGER_EVENTS and r["doc_id"] in {d["doc_id"] for d in ev}:
                used[LEDGER_EVENTS[e]].add(r["doc_id"])
    skip = {ROOT / c for c in CATALOGUES}
    for dirpath, _, files in os.walk(ROOT):
        here = Path(dirpath)
        if here.is_relative_to(ROOT / "corpus-v1") or here.is_relative_to(C):
            continue
        for f in files:
            p = here / f
            if not f.endswith((".json", ".jsonl")) or p in skip:
                continue
            cat = category(p.relative_to(ROOT).parts[0])
            if p.is_relative_to(ROOT / "final-20261003/run-a"):
                cat = "training_v1_it1_it2_phase0_runA"
            with p.open("rb") as stream:
                for chunk in iter(lambda s=stream: s.read(1 << 24), b""):
                    for m in set(HEX64.findall(chunk)):
                        g = ids.get(m.decode())
                        if g:
                            used[cat].add(g)
    excl = json.loads(
        (ROOT / "recovery-20260929-exposure-incident-v1/evaluation-exclusions.json").read_text()
    )
    quarantined = set(excl["excluded_unit_ids"]) | set(excl["excluded_source_ids"])
    used["quarantined_calibration_file"] = {ids[q] for q in quarantined if q in ids}
    return used


def frame() -> None:
    private_dir(C)
    documents = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    used = prior_use(documents)
    suite = set().union(*(grams(r["paragraph"]) for r in edge_rows()))
    stats: Counter[str] = Counter()
    keep, reasons = [], defaultdict(list)
    for g, cats in (
        (g, [c for c, s in used.items() if g in s]) for g in set().union(*used.values())
    ):
        reasons[g].extend(cats)
    for d in documents:
        if d["split"] == "train":
            continue
        stats[f"pool_documents_{d['split']}"] += 1
        meta = json.loads((TESTO / (Path(d["path"]).stem + ".json")).read_text())
        if meta.get("tipo") not in ("PERIZ", "STIMA"):
            stats["not_appraisal"] += 1
            continue
        stats["appraisal_documents"] += 1
        if grams(Path(d["path"]).read_text(encoding="utf-8")) & suite:
            reasons[d["doc_id"]].append("edge_suite_12gram")
        if reasons.get(d["doc_id"]):
            continue
        keep.append(d)
    pool_groups = {d["doc_id"] for d in documents if d["split"] != "train"}
    write_json(
        C / "exclusion-hashes.json",
        [
            {"group_sha256": hashlib.sha256(g.encode()).hexdigest(), "reasons": sorted(set(r))}
            for g, r in sorted(reasons.items())
            if g in pool_groups and r
        ],
    )
    write_json(C / "frame-private.json", [d["source_doc_id"] for d in keep])
    excluded = {g for g in reasons if g in pool_groups}
    summary = dict(stats) | {
        "pool_groups": len(pool_groups),
        "excluded_groups_any_reason": len(excluded),
        "excluded_groups_by_reason": {
            c: len({g for g in excluded if c in reasons[g]})
            for c in sorted({c for r in reasons.values() for c in r})
        },
        "frame_documents": len(keep),
        "frame_groups": len({d["doc_id"] for d in keep}),
        "frame_by_split": Counter(d["split"] for d in keep),
    }
    write_json(C / "frame-summary.json", summary)
    print(json.dumps(summary, indent=1))


# --- mining ------------------------------------------------------------------


def frame_docs() -> list[dict[str, Any]]:
    keep = set(json.loads((C / "frame-private.json").read_text()))
    docs = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    return [d for d in docs if d["source_doc_id"] in keep]


def candidate_sentences(text: str) -> list[tuple[int, int]]:
    """Sentences the A0 encoding scores, within the unit cap."""
    return [
        (s0, s1)
        for s0, s1 in sentence_windows(text)
        if s1 - s0 <= MAX_CHARS and text[s0:s1].strip()
    ]


def mine() -> None:
    rng = random.Random(20261008)
    docs = frame_docs()
    group_of = {d["source_doc_id"]: d["doc_id"] for d in docs}
    texts = {d["source_doc_id"]: Path(d["path"]).read_text(encoding="utf-8") for d in docs}
    hits: dict[str, list[Key]] = defaultdict(list)
    cues: dict[Key, list[str]] = {}
    none: list[Key] = []
    for src in sorted(texts):
        text = texts[src]
        for s0, s1 in candidate_sentences(text):
            found = cue_classes(text[s0:s1])
            cues[(src, s0, s1)] = found
            for c in found:
                hits[c].append((src, s0, s1))
            if not found:
                none.append((src, s0, s1))
    chosen: dict[Key, str] = {}
    for c in CLASSES:
        for k in sample_capped(hits[c], PER_CLASS, group_of, rng):
            chosen.setdefault(k, f"cue_{c}")
    for k in rng.sample(none, NON_CUE):
        chosen[k] = "non_cue"
    units, private = [], []
    spans_of: dict[str, list[tuple[int, int]]] = {}
    for (src, s0, s1), origin in chosen.items():
        spans = spans_of.setdefault(src, sentence_spans(texts[src]))
        uid = opaque("clean-unit", src, s0, s1)
        units.append({"unit_id": uid, **unit(texts[src], spans, spans.index((s0, s1)))})
        private.append(
            {
                "unit_id": uid,
                "source_doc_id": src,
                "group": group_of[src],
                "ss": s0,
                "se": s1,
                "origin": origin,
                "cues": cues[(src, s0, s1)],
            }
        )
    write_json(C / "units.json", units)
    write_json(C / "units-private.json", private)
    summary = {
        "frame_sentences": len(cues),
        "cue_hits": {c: len(hits[c]) for c in CLASSES},
        "cue_hit_groups": {c: len({group_of[k[0]] for k in hits[c]}) for c in CLASSES},
        "non_cue_sentences": len(none),
        "units": len(units),
        "units_by_origin": Counter(chosen.values()),
        "unit_chars_mean": round(
            sum(len(json.dumps(u, ensure_ascii=False)) for u in units) / len(units)
        ),
    }
    write_json(C / "mine-summary.json", summary)
    print(json.dumps(summary, indent=1))


# --- labelling ---------------------------------------------------------------


def load_ledger() -> dict[str, Any]:
    path = C / "sol-ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"tokens": 0, "batches": []}


def label(size: int) -> None:
    ledger = load_ledger()
    units = json.loads((C / "units.json").read_text())
    l1 = run_pass("L1", units, size, ledger, C, CAP)
    l2 = run_pass("L2", units, size, ledger, C, CAP)
    labels = {u["unit_id"]: consensus(l1.get(u["unit_id"]), l2.get(u["unit_id"])) for u in units}
    write_json(C / "labels.json", labels)
    write_json(C / "labels-passes.json", {"L1": l1, "L2": l2})
    print(
        json.dumps(
            {
                "sol_tokens": ledger["tokens"],
                "units": len(units),
                "labelled_both": sum(u["unit_id"] in l1 and u["unit_id"] in l2 for u in units),
                "uncertain": sum(v["uncertain"] for v in labels.values()),
            }
        )
    )


# --- scoring -----------------------------------------------------------------


def removed(s: list[float], thr: list[float]) -> bool:
    return any(x >= t for x, t in zip(s, thr, strict=True))


def score() -> None:
    private = json.loads((C / "units-private.json").read_text())
    labels = json.loads((C / "labels.json").read_text())
    cache = C / "a0-scores.json"
    if cache.exists():
        sc = {tuple(k): v for k, v in json.loads(cache.read_text())}
    else:
        paths = {
            d["source_doc_id"]: d["path"]
            for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
        }
        wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for p in private:
            wanted[p["source_doc_id"]].add((p["ss"], p["se"]))
        items = [
            ((src, *span), e)
            for src, spans in sorted(wanted.items())
            for span, examples in sentence_windows(
                Path(paths[src]).read_text(encoding="utf-8")
            ).items()
            if span in spans
            for e in examples
        ]
        sc = score_v1({"A0": V1}, items)["A0"]
        write_json(cache, [[list(k), v] for k, v in sc.items()])
    thr = v1_thresholds()
    report = summarise(private, labels, sc, thr)
    report["thresholds"] = [round(t, 5) for t in thr]
    report["sol_tokens"] = load_ledger()["tokens"]
    write_json(C / "report.json", report)
    print(json.dumps(report, indent=1))


def summarise(
    private: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    sc: dict[Key, list[float]],
    thr: list[float],
) -> dict[str, Any]:
    pos: dict[str, list[tuple[dict[str, Any], bool, bool]]] = defaultdict(list)
    harmless: dict[str, list[bool]] = defaultdict(list)
    docs: dict[str, bool] = {}
    uncertain: Counter[str] = Counter()
    for p in private:
        lab = labels[p["unit_id"]]
        s = sc[(p["source_doc_id"], p["ss"], p["se"])]
        hit = removed(s, thr)
        if lab["uncertain"]:
            uncertain[p["origin"]] += 1
            continue
        if not lab["remove_heads"]:
            for c in p["cues"]:
                harmless[c].append(hit)
            if p["origin"] == "non_cue":
                harmless["non_cue"].append(hit)
            continue
        for h in lab["remove_heads"]:
            same = s[HEADS.index(h)] >= thr[HEADS.index(h)]
            pos[h].append((p, hit, same))
        if set(lab["remove_heads"]) & set(CLASSES):
            docs[p["source_doc_id"]] = docs.get(p["source_doc_id"], False) or not hit
    out: dict[str, Any] = {"uncertain_by_origin": dict(uncertain)}
    for h in HEADS:
        rows = pos[h]
        out[h] = {
            "positives": len(rows),
            "groups": len({p["group"] for p, _, _ in rows}),
            "from_non_cue_sample": sum(p["origin"] == "non_cue" for p, _, _ in rows),
            "max_per_group": max(Counter(p["group"] for p, _, _ in rows).values(), default=0),
            "recall_removed": wilson(sum(hit for _, hit, _ in rows), len(rows)),
            "recall_same_head": wilson(sum(same for _, _, same in rows), len(rows)),
        }
    out["harmless_cue_deleted"] = {c: wilson(sum(v), len(v)) for c, v in harmless.items()}
    out["doc_leak_rare_classes"] = f"{sum(docs.values())}/{len(docs)}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("frame", "mine", "label", "score"))
    ap.add_argument("--size", type=int, default=120)
    args = ap.parse_args()
    {"frame": frame, "mine": mine, "label": lambda: label(args.size), "score": score}[args.step]()


if __name__ == "__main__":
    main()
