"""it3: train-side counterfactual, negation and dev data, with Sol labels.

Three data sets, all from TRAIN-split appraisal groups or non-test Cassazione
decisions, and never from a group of any test instrument:

* pairs: real positives of iterations 1 and 2 (train split, certain labels)
  edited by Sol with the counterfactual tooling: positive -> negated /
  deceased / company / hypothetical (should be kept) and famiglia -> salute /
  penale / minori (should be removed). Two blind Sol passes label each edited
  target; a pair is kept only when the consensus shows the intended flip.
* cass: negation-rich real sentences (a negation cue and a category cue) from
  Cassazione decisions outside the 847 test decisions, two blind passes.
* dev: a threshold-choice set from train groups that no earlier model used,
  every sentence scored by v1, binned on the v1 max score and sampled with
  known inclusion rates (higher bins oversampled), two blind passes.

Disagreement or either pass uncertain = uncertain (loss-masked in training).
`check` asserts zero group and 12-gram overlap with every test instrument.
Prints aggregates only; unit text and Sol events stay in the private directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import MINING, sentence_spans
from perizia_privacy.retraining.codex_cli import build_codex_argv
from scripts import counterfactuals as cf
from scripts import phase0_label as phase0
from scripts.clean_mining import C, candidate_sentences, consensus
from scripts.final_instrument import (
    BINS,
    TRAINING_INPUTS,
    F,
    score_v1,
    sentence_windows,
)
from scripts.phase0_eval import audit_labels, doc_text
from scripts.proof_label import usage
from scripts.proof_package import unit
from scripts.proof_score import (
    OUT,
    ROOT,
    TESTO,
    V1,
    edge_rows,
    grams,
    opaque,
    private_dir,
    write_json,
)
from scripts.sentenze import CUE, DOMAIN_NOTE, S, decisions, round_robin

I3 = ROOT / "it3-20261009"
CAP = 2_950_000  # the whole task may spend at most 3M Sol tokens
CF = C / "counterfactuals"
P0 = ROOT / "phase0-20261003"
MAX_TARGET = 600
EDITS = cf.EDITS | {
    "hypothetical": (
        "any",
        [],
        "Make the sensitive fact hypothetical or conditional so the target no longer "
        "states that it happened ('qualora', 'eventuale', 'nel caso in cui', 'se'). "
        "Keep the same person and topic.",
    ),
}
QUOTA = {
    "fam_to_salute": 140,
    "fam_to_penale": 140,
    "fam_to_minori": 140,
    "negated": 280,
    "deceased": 190,
    "company": 230,
    "hypothetical": 180,
}
EDIT_BATCH = 100
LABEL_BATCH = 150
CASS_PER_CLASS = 125
CASS_PER_DECISION = 3
DEV_DOCS = 120
DEV_ALLOC = (40, 40, 80, 100, 120, 120, 140)  # units per v1 score bin, high bins oversampled
DEV_POS_PER_HEAD = 60  # cue-mined positive supplement per head (recall only, never IPW)
NEGATION = re.compile(
    r"\bnon\s+(?:risult|sussist|emers|[eè]\s+(?:stat|provat|emers|dimostrat)|"
    r"(?:vi|v')\s*[eè]\s+prova|era\s+affett|[eè]\s+affett|ha\s+(?:precedent|riportat)|"
    r"aveva\s+precedent|present|ricorr)|\bescluso|\besclusa|\bescludendo|\bescluder|"
    r"\bassolt|\bassoluzion|\bnessun|\bincensurat|\binsussisten|\bprivo\s+di\s+precedent|"
    r"\bnegativ|\bin\s+assenza\s+di|\bassenza\s+di",
    re.I,
)
Key = tuple[str, int, int]


def ledger() -> dict[str, Any]:
    path = I3 / "sol-ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"tokens": 0, "batches": []}


def corpus() -> dict[str, dict[str, Any]]:
    return {
        d["source_doc_id"]: d for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }


# --- exclusions --------------------------------------------------------------


def test_instruments() -> dict[str, Any]:
    """Appraisal groups, Cassazione decisions and target 12-grams of every frozen test (cached)."""
    cache = I3 / "tests-private.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {
            "groups": {k: set(v) for k, v in raw["groups"].items()},
            "decisions": set(raw["decisions"]),
            "grams": set(raw["grams"]),
        }
    docs = corpus()
    group = {s: d["doc_id"] for s, d in docs.items()}
    groups: dict[str, set[str]] = {
        "audit100_audit15": {d["doc_id"] for d in docs.values() if d["audit"]},
        "ipw_deletion_sample": {
            group[p["source_doc_id"]] for p in json.loads((F / "deletion-private.json").read_text())
        },
        "held_out_positives": {
            group[p["source_doc_id"]] for p in json.loads((F / "positive-private.json").read_text())
        },
        "held_out_neighbours": {
            group[p["source_doc_id"]]
            for p in json.loads((F / "neighbour-private.json").read_text())
        },
        "clean_frame": {group[s] for s in json.loads((C / "frame-private.json").read_text())},
        "counterfactual_sources": {
            group[p["source_doc_id"]]
            for name in ("edit-sources-private.json",)
            for p in json.loads((CF / name).read_text()).values()
        }
        | {
            group[p["source_doc_id"]]
            for p in json.loads((CF / "supplement-private.json").read_text())
        },
    }
    cass = {p["decision"] for p in json.loads((S / "units-private.json").read_text())}
    targets = [
        u["target"]
        for path in (
            C / "units.json",
            CF / "supplement-units.json",
            S / "units.json",
            F / "positive-units.json",
            F / "neighbour-units.json",
        )
        for u in json.loads(path.read_text())
    ]
    targets += [
        e["edited_target"] for e in json.loads((CF / "edits-private.json").read_text()).values()
    ]
    for src, rows in audit_labels().items():
        text = doc_text(src)
        targets += [text[r["start"] : r["end"]] for r in rows if r["labels"]]
    out = {"groups": groups, "decisions": cass, "grams": set().union(*map(grams, targets))}
    write_json(
        cache,
        {
            "groups": {k: sorted(v) for k, v in groups.items()},
            "decisions": sorted(cass),
            "grams": sorted(out["grams"]),
        },
    )
    return out


def check(tests: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assert zero overlap between every it3 training/dev input and every test."""
    tests = tests or test_instruments()
    docs = corpus()
    test_groups = set().union(*tests["groups"].values())
    used: dict[str, set[str]] = {}
    for name in TRAINING_INPUTS[:1]:  # v1 replay rows
        with (ROOT / name).open() as stream:
            used["v1_replay"] = {
                r["doc_id"] for r in map(json.loads, stream) if not r["doc_id"].startswith("synth")
            }
    used["it1_sol_units"] = {
        u["group"] for u in json.loads((OUT / "units-private.json").read_text())
    }
    sources = json.loads((I3 / "pair-sources-private.json").read_text())
    used["pair_sources"] = {s["group"] for s in sources.values()}
    dev = json.loads((I3 / "dev-frame-private.json").read_text())
    used["dev"] = {docs[s]["doc_id"] for s in dev}
    report: dict[str, Any] = {"test_groups": {k: len(v) for k, v in tests["groups"].items()}}
    report["group_overlap_with_tests"] = {k: len(v & test_groups) for k, v in used.items()}
    report["dev_overlap_with_training"] = {
        k: len(used["dev"] & used[k]) for k in ("v1_replay", "it1_sol_units", "pair_sources")
    }
    report["non_train_split_groups"] = {
        k: sum(docs[s]["split"] != "train" for s in srcs)
        for k, srcs in (
            ("pair_sources", {s["source_doc_id"] for s in sources.values()}),
            ("dev", set(dev)),
        )
    }
    cass = json.loads((I3 / "cass-private.json").read_text())
    report["cass_decisions_in_test"] = len({p["decision"] for p in cass} & tests["decisions"])
    texts: list[str] = [s["target"] for s in sources.values()]
    texts += [u["target"] for u in json.loads((I3 / "cass-units.json").read_text())]
    for name in ("dev-units.json", "dev-pos-units.json"):
        if (I3 / name).exists():
            texts += [u["target"] for u in json.loads((I3 / name).read_text())]
    if (I3 / "twin-labels.json").exists():
        texts += [p["edited"] for p in kept_pairs()]
    report["target_12gram_overlap_with_tests"] = sum(bool(grams(t) & tests["grams"]) for t in texts)
    write_json(I3 / "exclusion-check.json", report)
    bad = (
        sum(report["group_overlap_with_tests"].values())
        + sum(report["dev_overlap_with_training"].values())
        + sum(report["non_train_split_groups"].values())
        + report["cass_decisions_in_test"]
        + report["target_12gram_overlap_with_tests"]
    )
    assert bad == 0, f"test contamination: {json.dumps(report)}"
    return report


# --- sources -----------------------------------------------------------------


def pair_sources(tests: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Certain positives of iterations 1 and 2 on train groups, outside every test."""
    docs = corpus()
    test_groups = set().union(*tests["groups"].values())
    out = {}
    for stem in ("", "iter2-"):
        labels = json.loads((OUT / f"{stem}labels.json").read_text())
        units = {u["unit_id"]: u for u in json.loads((OUT / f"{stem}units.json").read_text())}
        for p in json.loads((OUT / f"{stem}units-private.json").read_text()):
            lab = labels.get(p["unit_id"])
            if stem and p["set"] != "train":
                continue
            if lab is None or lab["uncertain"] or not lab["heads"]:
                continue
            u = units[p["unit_id"]]
            if (
                docs[p["source_doc_id"]]["split"] != "train"
                or p["group"] in test_groups
                or p["se"] - p["ss"] > MAX_TARGET
                or grams(u["target"]) & tests["grams"]
            ):
                continue
            out[p["unit_id"]] = {
                **{k: p[k] for k in ("source_doc_id", "group", "ss", "se")},
                "heads": sorted(lab["heads"]),
                "unit": u,
                "target": u["target"],
            }
    return out


def assign_edits(sources: dict[str, dict[str, Any]], seed: int = 20261009) -> dict[str, list[str]]:
    """Per edit type, the eligible sources with the fewest edits so far (one edit per type)."""
    rng = random.Random(seed)
    load: Counter[str] = Counter()
    jobs: dict[str, list[str]] = {}
    for kind, quota in QUOTA.items():
        eligible = [
            s
            for s, v in sources.items()
            if EDITS[kind][0] != "famiglia" or "famiglia" in v["heads"]
        ]
        rng.shuffle(eligible)
        eligible.sort(key=lambda s: load[s])
        jobs[kind] = eligible[:quota]
        load.update(jobs[kind])
    return jobs


def cass_units(
    tests: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Negation-plus-category sentences from decisions outside the test set."""
    rng = random.Random(20261011)
    docs = {d["id"]: d for d in decisions() if d["id"] not in tests["decisions"]}
    hits: dict[str, list[Key]] = defaultdict(list)
    for did in sorted(docs):
        text = docs[did]["text"]
        for s0, s1 in candidate_sentences(text):
            sent = text[s0:s1]
            if s1 - s0 > MAX_TARGET * 2 or not NEGATION.search(sent):
                continue
            if grams(" ".join(sent.split())) & tests["grams"]:
                continue
            for c, rx in CUE.items():
                if rx.search(sent):
                    hits[c].append((did, s0, s1))
    chosen = round_robin(hits, CASS_PER_CLASS, CASS_PER_DECISION, rng)
    units, private = [], []
    for (did, s0, s1), origin in sorted(chosen.items()):
        text = docs[did]["text"]
        spans = sentence_spans(text)
        uid = opaque("it3-cass", did, s0, s1)
        units.append({"unit_id": uid, **unit(text, spans, spans.index((s0, s1)))})
        private.append({"unit_id": uid, "decision": did, "ss": s0, "se": s1, "origin": origin})
    stats = {
        "decisions_outside_test": len(docs),
        "negation_category_hits": {c: len(v) for c, v in hits.items()},
        "units": len(units),
        "units_by_origin": Counter(chosen.values()),
        "unit_decisions": len({p["decision"] for p in private}),
    }
    return units, private, stats


def dev_frame(tests: dict[str, Any], pair_groups: set[str]) -> list[str]:
    """Train appraisal documents in groups no model trained or calibrated on."""
    docs = corpus()
    used: set[str] = set()
    for name in TRAINING_INPUTS:
        with (ROOT / name).open() as stream:
            used |= {json.loads(line)["doc_id"] for line in stream if line.strip()}
    for name in ("units-private.json", "iter2-units-private.json"):
        used |= {u["group"] for u in json.loads((OUT / name).read_text())}
    used |= {p["group"] for p in json.loads((P0 / "probe-private.json").read_text())}
    used |= set().union(*tests["groups"].values()) | pair_groups
    suite = set().union(*(grams(r["paragraph"]) for r in edge_rows()))
    out = []
    for d in sorted(docs.values(), key=lambda d: opaque("it3-dev", d["doc_id"])):
        if d["split"] != "train" or d["audit"] or d["doc_id"] in used:
            continue
        meta = json.loads((TESTO / (Path(d["path"]).stem + ".json")).read_text())
        if meta.get("tipo") not in ("PERIZ", "STIMA"):
            continue
        if grams(Path(d["path"]).read_text(encoding="utf-8")) & suite:
            continue
        out.append(d["source_doc_id"])
        if len(out) == DEV_DOCS:
            break
    return out


def sources() -> None:
    private_dir(I3)
    tests = test_instruments()
    src = pair_sources(tests)
    jobs = assign_edits(src)
    write_json(I3 / "pair-sources-private.json", src)
    write_json(I3 / "edit-jobs-private.json", jobs)
    units, private, cass_stats = cass_units(tests)
    write_json(I3 / "cass-units.json", units)
    write_json(I3 / "cass-private.json", private)
    dev = dev_frame(tests, {s["group"] for s in src.values()})
    write_json(I3 / "dev-frame-private.json", dev)
    summary = {
        "pair_sources": len(src),
        "pair_source_groups": len({s["group"] for s in src.values()}),
        "pair_source_heads": Counter(h for s in src.values() for h in s["heads"]),
        "edit_jobs": {k: len(v) for k, v in jobs.items()},
        "max_edits_per_source": max(Counter(s for v in jobs.values() for s in v).values()),
        "cass": cass_stats,
        "dev_documents": len(dev),
        "dev_groups": len({corpus()[s]["doc_id"] for s in dev}),
        "exclusion_check": check(tests),
    }
    write_json(I3 / "sources-summary.json", summary)
    print(json.dumps(summary, indent=1))


# --- Sol: edits --------------------------------------------------------------


def run_edit_batch(name: str, kind: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    work = private_dir(I3 / "sol" / "edits" / name)
    packet = private_dir(work / "packet")
    (packet / "schema.json").write_text(json.dumps(cf.SCHEMA))
    output = work / "output.json"
    prompt = cf.PROMPT.format(instruction=EDITS[kind][2])
    prompt += "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n"
    argv = build_codex_argv(packet, output, packet / "schema.json")
    argv[argv.index("gpt-6-sol")] = phase0.MODEL
    proc = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=3000)
    (work / "events.jsonl").write_text(proc.stdout)
    (work / "stderr.txt").write_text(proc.stderr)
    tokens = usage(proc.stdout)
    try:
        edits = cf.validate_edits(output.read_text(), {u["unit_id"] for u in units})
    except Exception as error:  # noqa: BLE001 - any failure means retry the batch
        return {"name": name, "ok": False, "tokens": tokens, "error": type(error).__name__}
    write_json(work / "edits.json", edits)
    (work / "accepted").touch()
    return {"name": name, "ok": True, "tokens": tokens}


def edit_units() -> dict[str, tuple[str, dict[str, Any]]]:
    """Edit inputs: opaque id -> (kind, unit with one context sentence a side)."""
    src = json.loads((I3 / "pair-sources-private.json").read_text())
    out = {}
    for kind, ids in json.loads((I3 / "edit-jobs-private.json").read_text()).items():
        for sid in ids:
            u = src[sid]["unit"]
            uid = opaque("it3-edit", kind, sid)
            out[uid] = (
                kind,
                {
                    "unit_id": uid,
                    "before": u["before"][-1:],
                    "target": u["target"],
                    "after": u["after"][:1],
                },
            )
    return out


def edit() -> None:
    led = ledger()
    todo = edit_units()
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kind, u in todo.values():
        by_kind[kind].append(u)
    batches = {
        f"{kind}-{i // EDIT_BATCH:02d}": (kind, us[i : i + EDIT_BATCH])
        for kind, us in by_kind.items()
        for i in range(0, len(us), EDIT_BATCH)
    }
    root = I3 / "sol" / "edits"
    for attempt in range(3):
        pending = {n: b for n, b in batches.items() if not (root / n / "accepted").exists()}
        if not pending:
            break
        projected = sum(
            len(json.dumps(u, ensure_ascii=False)) // 2 + 11_000 for _, u in pending.values()
        )
        if led["tokens"] + projected > CAP:
            raise SystemExit(f"Sol token cap would be exceeded: {led['tokens']} + {projected}")
        with ThreadPoolExecutor(min(len(pending), 6)) as pool:
            results = list(pool.map(lambda nb: run_edit_batch(nb[0], *nb[1]), pending.items()))
        for r in results:
            led["tokens"] += r["tokens"]
            led["batches"].append({k: r.get(k) for k in ("name", "ok", "tokens", "error")})
        write_json(I3 / "sol-ledger.json", led)
        print("edits attempt", attempt, "spent", led["tokens"], flush=True)
    edits: dict[str, dict[str, Any]] = {}
    for n in batches:
        if (root / n / "accepted").exists():
            edits |= json.loads((root / n / "edits.json").read_text())
    write_json(I3 / "edits-private.json", edits)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for uid, (kind, u) in todo.items():
        stats[kind][edit_status(u["target"], edits.get(uid))] += 1
    print(json.dumps({"sol_tokens": led["tokens"], "per_type": stats}, indent=1))


def edit_status(original: str, e: dict[str, Any] | None) -> str:
    if e is None:
        return "no_output"
    if not e["applicable"] or e["edited_target"] == original:
        return "not_applicable"
    if cf.similarity(original, e["edited_target"]) < cf.MIN_SIMILARITY:
        return "too_different"
    return "edited"


# --- Sol: labels ---------------------------------------------------------------


def twin_units() -> list[dict[str, Any]]:
    src = json.loads((I3 / "pair-sources-private.json").read_text())
    edits = json.loads((I3 / "edits-private.json").read_text())
    jobs = json.loads((I3 / "edit-jobs-private.json").read_text())
    out = []
    for kind, ids in jobs.items():
        for sid in ids:
            uid = opaque("it3-edit", kind, sid)
            u = src[sid]["unit"]
            if edit_status(u["target"], edits.get(uid)) != "edited":
                continue
            out.append(
                {
                    "unit_id": opaque("it3-twin", uid),
                    "before": u["before"],
                    "target": edits[uid]["edited_target"],
                    "after": u["after"],
                }
            )
    return out


def two_passes(tag: str, units: list[dict[str, Any]], note: str = "") -> dict[str, Any]:
    led = ledger()
    base = phase0.PROMPT
    phase0.PROMPT = note + base
    try:
        a = phase0.run_pass(f"{tag}1", units, LABEL_BATCH, led, I3, CAP)
        b = phase0.run_pass(f"{tag}2", units, LABEL_BATCH, led, I3, CAP)
    finally:
        phase0.PROMPT = base
    labels = {u["unit_id"]: consensus(a.get(u["unit_id"]), b.get(u["unit_id"])) for u in units}
    both = [u["unit_id"] for u in units if u["unit_id"] in a and u["unit_id"] in b]
    return {
        "labels": labels,
        "passes": {"1": a, "2": b},
        "stats": {
            "sol_tokens": led["tokens"],
            "units": len(units),
            "labelled_both": len(both),
            "uncertain": sum(v["uncertain"] for v in labels.values()),
            "positive": sum(
                bool(v["remove_heads"]) and not v["uncertain"] for v in labels.values()
            ),
            "pass_agreement": phase0.agreement({i: a[i] for i in both}, {i: b[i] for i in both})
            if both
            else None,
        },
    }


def flipped(kind: str, original_heads: list[str], twin: dict[str, Any]) -> bool:
    """The certain twin label moved as intended (cf.flipped, with the hypothetical edit)."""
    if twin["uncertain"]:
        return False
    source, wanted, _ = EDITS[kind]
    if source == "famiglia":
        return "famiglia" in original_heads and set(wanted) <= set(twin["remove_heads"])
    return bool(original_heads) and not twin["remove_heads"]


def kept_pairs() -> list[dict[str, Any]]:
    """Training pairs: source, kind, edited target and the twin's consensus label."""
    src = json.loads((I3 / "pair-sources-private.json").read_text())
    edits = json.loads((I3 / "edits-private.json").read_text())
    labels = json.loads((I3 / "twin-labels.json").read_text())
    test_grams = test_instruments()["grams"]
    out = []
    for kind, ids in json.loads((I3 / "edit-jobs-private.json").read_text()).items():
        for sid in ids:
            uid = opaque("it3-edit", kind, sid)
            lab = labels.get(opaque("it3-twin", uid))
            if lab is None or not flipped(kind, src[sid]["heads"], lab):
                continue
            if grams(edits[uid]["edited_target"]) & test_grams:
                continue
            s = src[sid]
            out.append(
                {
                    "uid": uid,
                    "kind": kind,
                    "source_unit": sid,
                    **{k: s[k] for k in ("source_doc_id", "group", "ss", "se", "heads")},
                    "edited": edits[uid]["edited_target"],
                    "twin_heads": lab["remove_heads"],
                }
            )
    return out


def twins() -> None:
    units = twin_units()
    res = two_passes("T", units)
    write_json(I3 / "twin-labels.json", res["labels"])
    write_json(I3 / "twin-labels-passes.json", res["passes"])
    pairs = kept_pairs()
    sent = Counter(
        k for k, ids in json.loads((I3 / "edit-jobs-private.json").read_text()).items() for _ in ids
    )
    kept = Counter(p["kind"] for p in pairs)
    print(
        json.dumps(
            res["stats"]
            | {
                "pairs_kept": kept,
                "pairs_sent": sent,
                "pairs_kept_total": len(pairs),
                "pair_source_groups": len({p["group"] for p in pairs}),
            },
            indent=1,
        )
    )
    write_json(I3 / "twins-summary.json", res["stats"] | {"pairs_kept": kept, "pairs_sent": sent})


def cass() -> None:
    units = json.loads((I3 / "cass-units.json").read_text())
    res = two_passes("N", units, DOMAIN_NOTE)
    write_json(I3 / "cass-labels.json", res["labels"])
    write_json(I3 / "cass-labels-passes.json", res["passes"])
    heads = Counter(
        h for v in res["labels"].values() if not v["uncertain"] for h in v["remove_heads"]
    )
    stats = res["stats"] | {"positive_heads": heads}
    write_json(I3 / "cass-summary.json", stats)
    print(json.dumps(stats, indent=1))


# --- dev -----------------------------------------------------------------------


def dev_scores() -> dict[Key, float]:
    """v1 max-head score of every candidate sentence of the dev documents (cached)."""
    cache = I3 / "dev-v1-scores.json"
    if cache.exists():
        return {tuple(k): v for k, v in json.loads(cache.read_text())}
    items = []
    for src in json.loads((I3 / "dev-frame-private.json").read_text()):
        text = doc_text(src)
        wins = sentence_windows(text)
        for s0, s1 in candidate_sentences(text):
            items += [((src, s0, s1), e) for e in wins[(s0, s1)]]
    sc = {k: max(v) for k, v in score_v1({"A0": V1}, items)["A0"].items()}
    write_json(cache, [[list(k), v] for k, v in sc.items()])
    return sc


def bin_of(score: float) -> int:
    return next(i for i in range(len(BINS) - 1) if BINS[i] <= score < BINS[i + 1])


def dev_sample() -> None:
    rng = random.Random(20261012)
    sc = dev_scores()
    test_grams = test_instruments()["grams"]
    by_bin: dict[int, list[Key]] = defaultdict(list)
    for (src, s0, s1), v in sorted(sc.items()):
        if not grams(doc_text(src)[s0:s1]) & test_grams:
            by_bin[bin_of(v)].append((src, s0, s1))
    docs = corpus()
    units, private = [], []
    spans_of: dict[str, list[tuple[int, int]]] = {}
    for b, n in enumerate(DEV_ALLOC):
        pop = by_bin[b]
        take = rng.sample(pop, min(n, len(pop)))
        for src, s0, s1 in take:
            text = doc_text(src)
            spans = spans_of.setdefault(src, sentence_spans(text))
            uid = opaque("it3-dev", src, s0, s1)
            units.append({"unit_id": uid, **unit(text, spans, spans.index((s0, s1)))})
            private.append(
                {
                    "unit_id": uid,
                    "source_doc_id": src,
                    "group": docs[src]["doc_id"],
                    "ss": s0,
                    "se": s1,
                    "bin": b,
                    "weight": len(pop) / len(take),
                }
            )
    write_json(I3 / "dev-units.json", units)
    write_json(I3 / "dev-private.json", private)
    stats = {
        "sentences": len(sc),
        "population_per_bin": [len(by_bin[b]) for b in range(len(DEV_ALLOC))],
        "units": len(units),
        "units_per_bin": Counter(p["bin"] for p in private),
    }
    write_json(I3 / "dev-summary.json", stats)
    print(json.dumps(stats, indent=1))


def dev_label() -> None:
    units = json.loads((I3 / "dev-units.json").read_text())
    res = two_passes("D", units)
    write_json(I3 / "dev-labels.json", res["labels"])
    write_json(I3 / "dev-labels-passes.json", res["passes"])
    heads = Counter(
        h for v in res["labels"].values() if not v["uncertain"] for h in v["remove_heads"]
    )
    stats = (
        json.loads((I3 / "dev-summary.json").read_text()) | res["stats"] | {"positive_heads": heads}
    )
    write_json(I3 / "dev-summary.json", stats)
    print(json.dumps(stats, indent=1))


def dev_positives() -> None:
    """Cue-mined positive supplement in the dev documents, two blind passes.

    The score-binned sample of the dev documents holds too few positives to
    set a threshold, so add every-head cue hits (the lexical miner, no model
    score) outside that sample. These units count for dev recall only.
    """
    rng = random.Random(20261013)
    test_grams = test_instruments()["grams"]
    taken = {
        (p["source_doc_id"], p["ss"], p["se"])
        for p in json.loads((I3 / "dev-private.json").read_text())
    }
    docs = corpus()
    hits: dict[str, list[Key]] = defaultdict(list)
    for src in json.loads((I3 / "dev-frame-private.json").read_text()):
        text = doc_text(src)
        for s0, s1 in candidate_sentences(text):
            sent = text[s0:s1]
            if (src, s0, s1) in taken or grams(sent) & test_grams:
                continue
            for h in ("salute", "penale", "minori", "famiglia"):
                if MINING[h].search(sent):
                    hits[h].append((src, s0, s1))
    chosen = round_robin(hits, DEV_POS_PER_HEAD, 3, rng)
    units, private = [], []
    for (src, s0, s1), origin in sorted(chosen.items()):
        text = doc_text(src)
        spans = sentence_spans(text)
        uid = opaque("it3-devpos", src, s0, s1)
        units.append({"unit_id": uid, **unit(text, spans, spans.index((s0, s1)))})
        private.append(
            {
                "unit_id": uid,
                "source_doc_id": src,
                "group": docs[src]["doc_id"],
                "ss": s0,
                "se": s1,
                "origin": origin,
            }
        )
    write_json(I3 / "dev-pos-units.json", units)
    write_json(I3 / "dev-pos-private.json", private)
    res = two_passes("P", units)
    write_json(I3 / "dev-pos-labels.json", res["labels"])
    write_json(I3 / "dev-pos-labels-passes.json", res["passes"])
    heads = Counter(
        h for v in res["labels"].values() if not v["uncertain"] for h in v["remove_heads"]
    )
    stats = {
        "cue_hits": {h: len(v) for h, v in hits.items()},
        "units_by_origin": Counter(chosen.values()),
    }
    stats |= res["stats"] | {"positive_heads": heads}
    write_json(I3 / "dev-pos-summary.json", stats)
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "step",
        choices=(
            "sources",
            "edit",
            "twins",
            "cass",
            "dev-sample",
            "dev-label",
            "dev-positives",
            "check",
        ),
    )
    args = ap.parse_args()
    private_dir(I3)
    steps = {
        "sources": sources,
        "edit": edit,
        "twins": twins,
        "cass": cass,
        "dev-sample": dev_sample,
        "dev-label": dev_label,
        "dev-positives": dev_positives,
        "check": lambda: print(json.dumps(check(), indent=1)),
    }
    steps[args.step]()


if __name__ == "__main__":
    main()
