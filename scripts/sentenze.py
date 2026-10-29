"""Out-of-domain rare-class test set from public Cassazione decisions.

Perizie carry too few salute, penale and minori positives for a recall
estimate, so this builds a real test set from public Italian court decisions:

* fetch: civil and criminal decisions from the Corte di cassazione SentenzeWeb
  archive (free, no registration; official acts, art. 5 l. 633/1941), through
  its public search index, at most one request every 1.5 s, 20 decisions per
  request, topic strata plus uniform strata;
* mine: the clean set's broad cue lists plus a famiglia list, NO model score;
  per class a uniform round-robin sample capped at PER_GROUP units per
  decision (across all origins), plus a uniform sample of non-cue sentences;
* label: two blind Sol passes with the frozen Phase 0 schema and policy
  prompt, behind a short domain note that maps decision parties onto the
  prompt's roles; disagreement or either pass uncertain = uncertain;
* score: A0 at its production thresholds, only after labelling.

Prints aggregates only; decision text, unit text and Sol events stay in the
private directory, which is never committed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
import urllib.parse
from collections import Counter, defaultdict
from typing import Any

from perizia_privacy.corpus import sentence_spans
from scripts import phase0_label as phase0
from scripts.clean_mining import CUES, consensus, removed, wilson
from scripts.final_instrument import MAX_CHARS, score_v1, sentence_windows
from scripts.proof_package import unit
from scripts.proof_score import (
    HEADS,
    ROOT,
    V1,
    opaque,
    private_dir,
    v1_thresholds,
    write_json,
)

S = ROOT / "sentenze-20261008"
CAP = 2_000_000
SOLR = "https://www.italgiure.giustizia.it/sncass/isapi/hc.dll/sn.solr/sn-collection/select"
AGENT = "Mozilla/5.0 (academic research on privacy redaction; University of Bologna)"
DELAY = 1.5
ROWS = 20
PER_GROUP = 3
NON_CUE = 300
Key = tuple[str, int, int]

# (stratum, Solr query, decisions wanted). Topic strata raise the positive
# density; the two uniform strata keep ordinary decisions in the frame.
STRATA = (
    ("civ_famiglia", 'kind:snciv AND (ocr:"separazione personale" OR ocr:divorzio)', 450),
    ("civ_minori", "kind:snciv AND ocr:affidamento AND ocr:minore", 350),
    (
        "civ_salute",
        'kind:snciv AND (ocr:"invalidità civile" OR ocr:"danno biologico" OR '
        'ocr:"indennità di accompagnamento" OR ocr:"amministrazione di sostegno")',
        500,
    ),
    ("civ_successioni", "kind:snciv AND ocr:successione AND ocr:eredi", 250),
    ("civ_uniform", "kind:snciv", 300),
    (
        "pen_minori",
        'kind:snpen AND ocr:minore AND (ocr:maltrattamenti OR ocr:"violenza sessuale")',
        300,
    ),
    (
        "pen_salute",
        'kind:snpen AND (ocr:"regime carcerario" OR ocr:"vizio di mente" OR '
        'ocr:"perizia psichiatrica")',
        250,
    ),
    ("pen_uniform", "kind:snpen", 600),
)

CLASSES = HEADS  # salute, penale, minori, famiglia
FAMIGLIA = re.compile(
    r"coniug|matrimon|sposat|separazion|divorz|vedov|celib|nubil|convivi|more\s+uxorio|"
    r"unione\s+civil|casa\s+(?:coniugale|familiare)|assegno\s+(?:di\s+mantenimento|divorzile)|"
    r"mantenimento|fondo\s+patrimonial|comunione\s+(?:legale|dei\s+beni)|separazione\s+dei\s+beni|"
    r"\bmoglie|\bmarit|compagn[oa]\b|fidanzat|madre|padre|fratell|sorell|suocer|cognat|"
    r"parent|famigli|nucleo\s+familiare|ered[ei]\b|eredit|de\s+cuius|successi|testament",
    re.I,
)
CUE = {"salute": CUES["salute"], "penale": CUES["penale"], "minori": CUES["minori"]} | {
    "famiglia": FAMIGLIA
}

DOMAIN_NOTE = """Domain note: the units below come from public decisions of the \
Italian Supreme Court (Corte di cassazione), not from perizie. Apply the \
categories and the policy below unchanged. The parties of the decision \
(ricorrente, controricorrente, resistente, imputato, appellante, persona \
offesa, their relatives) are the persons "in this procedure": a party named in \
clear is living; a party masked by initials, omissis or a role word ("il \
ricorrente", "l'imputato") is masked_party. A statement of law or of case law \
("in tema di ...", "questa Corte ha affermato ...") is hypothetical.

"""


def cue_classes(text: str) -> list[str]:
    return [c for c in CLASSES if CUE[c].search(text)]


# --- fetch -------------------------------------------------------------------


def solr(params: dict[str, Any]) -> dict[str, Any]:
    """One index request through curl (the host omits an intermediate certificate
    that Python's bundle lacks; curl uses the system trust store)."""
    url = SOLR + "?" + urllib.parse.urlencode(params | {"wt": "json"})
    for attempt in range(3):
        time.sleep(DELAY * (1 + 4 * attempt))
        proc = subprocess.run(
            ["curl", "-sS", "--fail", "-m", "90", "-A", AGENT, url],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
    raise RuntimeError(f"index request failed: {proc.stderr.decode()[:200]}")


def fetch() -> None:
    private_dir(S)
    out = S / "decisions.jsonl"
    have = {}
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            have[d["id"]] = d["stratum"]
    rng = random.Random(1134)
    stats: dict[str, Any] = {}
    requests = 0
    with out.open("a") as stream:
        for name, query, want in STRATA:
            found = solr({"q": query, "rows": 0})["response"]["numFound"]
            requests += 1
            got = sum(s == name for s in have.values())
            pages = list(range(0, found, ROWS))
            rng.shuffle(pages)
            for start in pages:
                if got >= want:
                    break
                docs = solr(
                    {
                        "q": query,
                        "rows": ROWS,
                        "start": start,
                        "sort": "id asc",
                        "fl": "id,kind,szdec,tipoprov,anno,ocr",
                    }
                )["response"]["docs"]
                requests += 1
                for d in docs:
                    text = "\n".join(d.get("ocr") or [])
                    if d["id"] in have or len(text) < 1000 or got >= want:
                        continue
                    have[d["id"]] = name
                    got += 1
                    row = {k: d.get(k) for k in ("id", "kind", "szdec", "tipoprov", "anno")}
                    stream.write(json.dumps(row | {"stratum": name, "text": text}) + "\n")
                stream.flush()
            stats[name] = {"index_hits": found, "decisions": got}
            print(name, stats[name], flush=True)
    summary = {"strata": stats, "decisions": len(have), "requests_this_run": requests}
    write_json(S / "fetch-summary.json", summary)
    print(json.dumps(summary, indent=1))


def decisions() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (S / "decisions.jsonl").read_text().splitlines()]


# --- mining ------------------------------------------------------------------


def candidate_sentences(text: str) -> list[tuple[int, int]]:
    return [
        (s0, s1)
        for s0, s1 in sentence_windows(text)
        if s1 - s0 <= MAX_CHARS and text[s0:s1].strip()
    ]


def round_robin(
    hits: dict[str, list[Key]], quota: int, cap: int, rng: random.Random
) -> dict[Key, str]:
    """Uniform per-class samples, one class at a time, at most cap units per decision."""
    pools = {c: rng.sample(v, len(v)) for c, v in hits.items()}
    taken: Counter[str] = Counter()
    per: Counter[str] = Counter()
    chosen: dict[Key, str] = {}
    active = [c for c in pools if pools[c]]
    while active:
        for c in list(active):
            while pools[c]:
                k = pools[c].pop()
                if k not in chosen and per[k[0]] < cap:
                    chosen[k] = f"cue_{c}"
                    per[k[0]] += 1
                    taken[c] += 1
                    break
            if not pools[c] or taken[c] >= quota:
                active.remove(c)
    return chosen


def mine(quota: int) -> None:
    rng = random.Random(20261008)
    docs = {d["id"]: d for d in decisions()}
    hits: dict[str, list[Key]] = defaultdict(list)
    cues: dict[Key, list[str]] = {}
    none: list[Key] = []
    long_sentences = 0
    for did in sorted(docs):
        text = docs[did]["text"]
        long_sentences += sum(s1 - s0 > MAX_CHARS for s0, s1 in sentence_spans(text))
        for s0, s1 in candidate_sentences(text):
            found = cue_classes(text[s0:s1])
            cues[(did, s0, s1)] = found
            for c in found:
                hits[c].append((did, s0, s1))
            if not found:
                none.append((did, s0, s1))
    chosen = round_robin(hits, quota, PER_GROUP, rng)
    per = Counter(k[0] for k in chosen)
    free = [k for k in none if per[k[0]] < PER_GROUP]
    for k in rng.sample(free, len(free)):
        if sum(o == "non_cue" for o in chosen.values()) >= NON_CUE:
            break
        if per[k[0]] < PER_GROUP:
            chosen[k] = "non_cue"
            per[k[0]] += 1
    units, private = [], []
    spans_of: dict[str, list[tuple[int, int]]] = {}
    for (did, s0, s1), origin in sorted(chosen.items()):
        text = docs[did]["text"]
        spans = spans_of.setdefault(did, sentence_spans(text))
        uid = opaque("sentenze-unit", did, s0, s1)
        units.append({"unit_id": uid, **unit(text, spans, spans.index((s0, s1)))})
        private.append(
            {
                "unit_id": uid,
                "decision": did,
                "kind": docs[did]["kind"],
                "stratum": docs[did]["stratum"],
                "ss": s0,
                "se": s1,
                "origin": origin,
                "cues": cues[(did, s0, s1)],
            }
        )
    write_json(S / "units.json", units)
    write_json(S / "units-private.json", private)
    chars = [len(json.dumps(u, ensure_ascii=False)) for u in units]
    summary = {
        "decisions": len(docs),
        "decisions_by_kind": Counter(d["kind"] for d in docs.values()),
        "frame_sentences": len(cues),
        "sentences_over_max_chars_excluded": long_sentences,
        "cue_hits": {c: len(hits[c]) for c in CLASSES},
        "cue_hit_decisions": {c: len({k[0] for k in hits[c]}) for c in CLASSES},
        "non_cue_sentences": len(none),
        "units": len(units),
        "units_by_origin": Counter(chosen.values()),
        "unit_decisions": len(per),
        "max_units_per_decision": max(per.values()),
        "unit_chars_mean": round(sum(chars) / len(chars)),
        "projected_sol_tokens_two_passes": 2 * projected(chars, 120),
    }
    write_json(S / "mine-summary.json", summary)
    print(json.dumps(summary, indent=1))


def projected(chars: list[int], size: int) -> int:
    """run_pass's projection with a 15% margin observed on the clean set."""
    batches = -(-len(chars) // size)
    return round(1.15 * (sum(chars) // 3 + 11_000 * batches + 75 * len(chars)))


# --- labelling ---------------------------------------------------------------


def load_ledger() -> dict[str, Any]:
    path = S / "sol-ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"tokens": 0, "batches": []}


def label(size: int) -> None:
    ledger = load_ledger()
    units = json.loads((S / "units.json").read_text())
    if not phase0.PROMPT.startswith(DOMAIN_NOTE):
        phase0.PROMPT = DOMAIN_NOTE + phase0.PROMPT
    l1 = phase0.run_pass("L1", units, size, ledger, S, CAP)
    l2 = phase0.run_pass("L2", units, size, ledger, S, CAP)
    labels = {u["unit_id"]: consensus(l1.get(u["unit_id"]), l2.get(u["unit_id"])) for u in units}
    write_json(S / "labels.json", labels)
    write_json(S / "labels-passes.json", {"L1": l1, "L2": l2})
    both = [u["unit_id"] for u in units if u["unit_id"] in l1 and u["unit_id"] in l2]
    print(
        json.dumps(
            {
                "sol_tokens": ledger["tokens"],
                "units": len(units),
                "labelled_both": len(both),
                "uncertain": sum(v["uncertain"] for v in labels.values()),
                "pass_agreement": phase0.agreement(
                    {i: l1[i] for i in both}, {i: l2[i] for i in both}
                ),
            },
            indent=1,
        )
    )


# --- scoring -----------------------------------------------------------------


def score() -> None:
    private = json.loads((S / "units-private.json").read_text())
    labels = json.loads((S / "labels.json").read_text())
    cache = S / "a0-scores.json"
    if cache.exists():
        sc = {tuple(k): v for k, v in json.loads(cache.read_text())}
    else:
        texts = {d["id"]: d["text"] for d in decisions()}
        wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for p in private:
            wanted[p["decision"]].add((p["ss"], p["se"]))
        items = [
            ((did, *span), e)
            for did, spans in sorted(wanted.items())
            for span, examples in sentence_windows(texts[did]).items()
            if span in spans
            for e in examples
        ]
        sc = score_v1({"A0": V1}, items)["A0"]
        write_json(cache, [[list(k), v] for k, v in sc.items()])
    thr = v1_thresholds()
    report = summarise(private, labels, sc, thr)
    report["thresholds"] = [round(t, 5) for t in thr]
    report["sol_tokens"] = load_ledger()["tokens"]
    write_json(S / "report.json", report)
    print(json.dumps(report, indent=1))


def summarise(
    private: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    sc: dict[Key, list[float]],
    thr: list[float],
) -> dict[str, Any]:
    pos: dict[str, list[tuple[dict[str, Any], bool, bool]]] = defaultdict(list)
    harmless: dict[str, list[bool]] = defaultdict(list)
    leak: dict[str, dict[str, bool]] = {"all_heads": {}, "salute_penale_minori": {}}
    uncertain: Counter[str] = Counter()
    for p in private:
        lab = labels[p["unit_id"]]
        s = sc[(p["decision"], p["ss"], p["se"])]
        hit = removed(s, thr)
        if lab["uncertain"]:
            uncertain[p["origin"]] += 1
            continue
        if not lab["remove_heads"]:
            for c in p["cues"]:
                harmless[c].append(hit)
            harmless["non_cue" if p["origin"] == "non_cue" else "any_cue"].append(hit)
            continue
        for h in lab["remove_heads"]:
            pos[h].append((p, hit, s[HEADS.index(h)] >= thr[HEADS.index(h)]))
        for scope, heads in (("all_heads", HEADS), ("salute_penale_minori", HEADS[:3])):
            if set(lab["remove_heads"]) & set(heads):
                d = p["decision"]
                leak[scope][d] = leak[scope].get(d, False) or not hit
    out: dict[str, Any] = {
        "units": len(private),
        "certain": len(private) - sum(uncertain.values()),
        "uncertain_by_origin": dict(uncertain),
    }
    for h in HEADS:
        rows = pos[h]
        out[h] = {
            "positives": len(rows),
            "decisions": len({p["decision"] for p, _, _ in rows}),
            "by_kind": dict(Counter(p["kind"] for p, _, _ in rows)),
            "by_stratum": dict(Counter(p["stratum"] for p, _, _ in rows)),
            "from_non_cue_sample": sum(p["origin"] == "non_cue" for p, _, _ in rows),
            "max_per_decision": max(Counter(p["decision"] for p, _, _ in rows).values(), default=0),
            "recall_removed": wilson(sum(hit for _, hit, _ in rows), len(rows)),
            "recall_same_head": wilson(sum(same for _, _, same in rows), len(rows)),
            "recall_removed_by_kind": {
                k: wilson(
                    sum(hit for p, hit, _ in rows if p["kind"] == k),
                    sum(p["kind"] == k for p, _, _ in rows),
                )
                for k in ("snciv", "snpen")
            },
        }
    out["harmless_deleted"] = {c: wilson(sum(v), len(v)) for c, v in sorted(harmless.items())}
    out["decision_leak"] = {scope: wilson(sum(v.values()), len(v)) for scope, v in leak.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fetch", "mine", "label", "score"))
    ap.add_argument("--quota", type=int, default=250, help="cue units per class")
    ap.add_argument("--size", type=int, default=120)
    args = ap.parse_args()
    {
        "fetch": fetch,
        "mine": lambda: mine(args.quota),
        "label": lambda: label(args.size),
        "score": score,
    }[args.step]()


if __name__ == "__main__":
    main()
