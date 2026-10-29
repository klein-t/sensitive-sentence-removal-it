"""Local LLM zero-shot baseline for the perizia redactor, plus a cascade.

A local instruct model (LM Studio on loopback, MLX 4-bit) labels each unit
with the frozen Phase 0 policy prompt and factored JSON schema the Sol
labeller used (plus the Cassazione domain note on the sentenze set). Decoding
is greedy (temperature 0) and schema-constrained; an answer that fails
validation is re-asked once with the error. The prediction is the unit's
``remove_heads`` set (raw): a unit counts as removed when it is non-empty.
Every metric is also reported for the policy applied to the model's own
topics, subject and assertion (composed), from the same answers.

Instruments, in order: the counterfactual pairs, the Cassazione set, the final
instrument (IPW deletion sample, held-out positives, neighbours) and the clean
set. ``cascade`` lets the LLM judge only the audit100 sentences A0 removes at
its production thresholds, and it may only un-delete.

Every prediction is cached per unit (append-only JSONL in the private
directory), so a killed run resumes where it stopped. Prints aggregates only;
unit text never leaves the private directory.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np

from scripts import phase0_label as phase0
from scripts.proof_score import (
    HEADS,
    OUT,
    ROOT,
    opaque,
    private_dir,
    v1_thresholds,
    write_json,
)

L = ROOT / "llm-baseline-20261009"
URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "privacy-llm-baseline"
CHECKPOINT = "lmstudio-community/gemma-4-12b-it-MLX-4bit"
WORKERS = 1
# A prediction is a 0/1 "score" per head; any head at 1 means removed.
ON = [0.5] * 4
MODES = ("raw", "composed")
Key = tuple[str, int, int]

ONE_SCHEMA = json.loads(json.dumps(phase0.SCHEMA))
ONE_SCHEMA["properties"]["labels"] |= {"minItems": 1, "maxItems": 1}


def prompt_for(u: dict[str, Any], note: str = "") -> str:
    """The Sol input for a one-unit batch: prompt, then the unit as one JSON line."""
    row = {k: u[k] for k in ("unit_id", "before", "target", "after")}
    return note + phase0.PROMPT + json.dumps(row, ensure_ascii=False) + "\n"


def ask(client: httpx.Client, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": 1134,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 600,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "labels", "strict": True, "schema": ONE_SCHEMA},
        },
    }
    r = client.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def label_one(client: httpx.Client, u: dict[str, Any], note: str) -> dict[str, Any]:
    """Greedy answer, validated; one re-ask carrying the validation error."""
    started = time.monotonic()
    messages = [{"role": "user", "content": prompt_for(u, note)}]
    tokens = [0, 0]
    error = None
    for attempt in range(2):
        raw, usage = ask(client, messages)
        tokens[0] += usage.get("prompt_tokens") or 0
        tokens[1] += usage.get("completion_tokens") or 0
        try:
            lab = phase0.validate(raw, {u["unit_id"]})[u["unit_id"]]
        except Exception as e:  # noqa: BLE001 - any invalid answer is re-asked once
            error = type(e).__name__
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That answer was invalid ({error}). Return only the JSON "
                    f'object {{"labels": [...]}} with one entry for unit_id {u["unit_id"]}.',
                },
            ]
            continue
        lab = {k: v for k, v in lab.items() if k != "unit_id"}
        return {
            "unit_id": u["unit_id"],
            "label": lab,
            "attempts": attempt + 1,
            "seconds": round(time.monotonic() - started, 3),
            "tokens": tokens,
        }
    return {
        "unit_id": u["unit_id"],
        "label": None,
        "error": error,
        "attempts": 2,
        "seconds": round(time.monotonic() - started, 3),
        "tokens": tokens,
    }


def predict(tag: str, units: list[dict[str, Any]], note: str = "") -> dict[str, dict[str, Any]]:
    """Label every unit once; resumable through the per-unit JSONL cache."""
    d = private_dir(L / tag)
    cache = d / "preds.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["unit_id"]] = row
    todo = [u for u in {u["unit_id"]: u for u in units}.values() if u["unit_id"] not in done]
    if todo:
        lock = threading.Lock()
        started = time.monotonic()
        with (
            httpx.Client() as client,
            cache.open("a") as stream,
            ThreadPoolExecutor(WORKERS) as pool,
        ):
            for n, row in enumerate(pool.map(lambda u: label_one(client, u, note), todo), 1):
                with lock:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                done[row["unit_id"]] = row
                if n % 50 == 0:
                    rate = n / (time.monotonic() - started)
                    print(f"{tag}: {n}/{len(todo)} ({rate:.2f} units/s)", flush=True)
        wall = time.monotonic() - started
        with (d / "sessions.jsonl").open("a") as s:
            s.write(json.dumps({"units": len(todo), "wall_seconds": round(wall, 2)}) + "\n")
    return {u["unit_id"]: done[u["unit_id"]] for u in units}


def composed(lab: dict[str, Any]) -> list[str]:
    """The prompt's policy (step 4) applied to the model's own topics, subject and
    assertion. One subject per unit, so rule e's living-heir exception cannot fire."""
    if lab["assertion"] != "affirmed":
        return []
    if lab["subject"] in ("living", "masked_party"):
        return list(lab["topics"])
    if lab["subject"] == "company":
        return [h for h in lab["topics"] if h == "penale"]
    return []


def heads(row: dict[str, Any], mode: str = "raw") -> list[str]:
    """raw = the model's remove_heads; composed = the policy over its factored answers.
    A failed unit (no valid answer after the re-ask) is kept, i.e. nothing removed."""
    if not row["label"]:
        return []
    return row["label"]["remove_heads"] if mode == "raw" else composed(row["label"])


def as_scores(row: dict[str, Any], mode: str) -> list[float]:
    return [float(h in heads(row, mode)) for h in HEADS]


def throughput(tag: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sessions = [json.loads(x) for x in (L / tag / "sessions.jsonl").read_text().splitlines()]
    units = sum(s["units"] for s in sessions)
    wall = sum(s["wall_seconds"] for s in sessions)
    vals = list(rows.values())
    return {
        "units": len(vals),
        "units_per_second": round(units / wall, 3) if wall else None,
        "concurrency": WORKERS,
        "failed_after_reask": sum(r["label"] is None for r in vals),
        "reasked": sum(r["attempts"] > 1 for r in vals),
        "mean_prompt_tokens": round(np.mean([r["tokens"][0] for r in vals])),
        "mean_completion_tokens": round(np.mean([r["tokens"][1] for r in vals])),
    }


# --- instruments ---------------------------------------------------------------


def counterfactuals() -> dict[str, Any]:
    from scripts.clean_mining import C, wilson
    from scripts.counterfactuals import CF, EDITS, flipped

    sources = json.loads((CF / "edit-sources-private.json").read_text())
    edits = json.loads((CF / "edits-private.json").read_text())
    labels = json.loads((CF / "edited-labels.json").read_text())
    by_source = {
        u["unit_id"]: u
        for name in ("units.json", "counterfactuals/supplement-units.json")
        for u in json.loads((C / name).read_text())
    }
    pairs, units = [], []
    for uid, s in sources.items():
        blind = opaque("cf-blind", uid)
        lab = labels.get(blind)
        if lab is None or not flipped(s["kind"], s["label"], lab):
            continue
        orig = by_source[s["source_unit"]]
        pairs.append((s["kind"], orig["unit_id"], blind))
        units += [orig, {**orig, "unit_id": blind, "target": edits[uid]["edited_target"]}]
    rows = predict("counterfactuals", units)
    out: dict[str, Any] = {"pairs": Counter(k for k, _, _ in pairs)}
    for mode in MODES:
        per: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for kind, o_id, e_id in pairs:
            o, e = bool(heads(rows[o_id], mode)), bool(heads(rows[e_id], mode))
            if EDITS[kind][0] == "famiglia":
                per[kind]["target_removed (edited, new class)"].append(e)
                per[kind]["source_removed (original famiglia)"].append(o)
            else:
                per[kind]["target_removed (original positive)"].append(o)
                per[kind]["twin_kept (edited harmless)"].append(not e)
                per[kind]["pair_correct (removed and kept)"].append(o and not e)
        out[f"llm_{mode}"] = {
            k: {m: wilson(sum(v), len(v)) for m, v in d.items()} for k, d in per.items()
        }
    return out | {"throughput": throughput("counterfactuals", rows)}


def sentenze() -> dict[str, Any]:
    from scripts import sentenze as sz

    private = json.loads((sz.S / "units-private.json").read_text())
    labels = json.loads((sz.S / "labels.json").read_text())
    units = [
        u
        for u in json.loads((sz.S / "units.json").read_text())
        if not labels[u["unit_id"]]["uncertain"]
    ]
    rows = predict("sentenze", units, sz.DOMAIN_NOTE)
    out: dict[str, Any] = {"throughput": throughput("sentenze", rows)}
    for mode in MODES:
        sc = {
            (p["decision"], p["ss"], p["se"]): as_scores(rows[p["unit_id"]], mode)
            if p["unit_id"] in rows
            else [0.0] * 4
            for p in private
        }
        out[f"llm_{mode}"] = sz.summarise(private, labels, sc, ON)
    return out


def ipw(rows: list[tuple[float, bool]]) -> str:
    """Weighted deletion rate with the final instrument's 1,000-draw bootstrap."""
    w = np.array([x for x, _ in rows])
    r = np.array([y for _, y in rows], dtype=float)
    rng = random.Random(1134)
    boot = []
    for _ in range(1000):
        idx = [rng.randrange(len(w)) for _ in range(len(w))]
        boot.append(float((w[idx] * r[idx]).sum() / w[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{(w * r).sum() / w.sum():.2%} [{lo:.2%}, {hi:.2%}] (n={len(w)})"


def final(deletion: int | None) -> dict[str, Any]:
    """Held-out positives first, then the IPW deletion sample (optionally a seeded
    uniform subsample, which keeps the IPW estimate unbiased); A0 at production
    thresholds is scored on exactly the same units."""
    from scripts.final_eval import F, ci, removed

    a0 = {tuple(k): v for k, v in json.loads((F / "eval-scores.json").read_text())["A0"]}
    thr = v1_thresholds()
    units = {
        part: {u["unit_id"]: u for u in json.loads((F / f"{part}-units.json").read_text())}
        for part in ("positive", "deletion")
    }
    lp = json.loads((F / "labels-positives.json").read_text())
    pos = [
        p
        for p in json.loads((F / "positive-private.json").read_text())
        if not lp[p["unit_id"]]["uncertain"] and lp[p["unit_id"]]["remove_heads"]
    ]
    rows = predict("final-positive", [units["positive"][p["unit_id"]] for p in pos])
    out: dict[str, Any] = {"throughput": {"positive": throughput("final-positive", rows)}}
    for name, hit in (
        ("llm_raw", lambda p: bool(heads(rows[p["unit_id"]], "raw"))),
        ("llm_composed", lambda p: bool(heads(rows[p["unit_id"]], "composed"))),
        ("a0_production", lambda p: removed(a0[(p["source_doc_id"], p["ss"], p["se"])], thr)),
    ):
        per_head = {h: [0, 0] for h in HEADS}
        docs: dict[str, bool] = {}
        k = 0
        for p in pos:
            h = hit(p)
            k += h
            for head in lp[p["unit_id"]]["remove_heads"]:
                per_head[head][0] += 1
                per_head[head][1] += h
            docs[p["source_doc_id"]] = docs.get(p["source_doc_id"], False) or not h
        out[name] = {
            "held_out_recall": ci(k, len(pos)),
            "held_out_recall_by_head": {h: ci(b, a) for h, (a, b) in per_head.items()},
            "held_out_doc_leak": f"{sum(docs.values())}/{len(docs)}",
        }
    dl = json.loads((F / "labels-deletion.json").read_text())
    harm = [
        p
        for p in json.loads((F / "deletion-private.json").read_text())
        if not dl[p["unit_id"]]["uncertain"] and not dl[p["unit_id"]]["remove_heads"]
    ]
    if deletion is not None and deletion < len(harm):
        harm = random.Random(20261009).sample(harm, deletion)
    rows = predict("final-deletion", [units["deletion"][p["unit_id"]] for p in harm])
    out["throughput"]["deletion"] = throughput("final-deletion", rows)
    for mode in MODES:
        out[f"llm_{mode}"]["ipw_harmless_deleted"] = ipw(
            [(float(p["weight"]), bool(heads(rows[p["unit_id"]], mode))) for p in harm]
        )
    out["a0_production"]["ipw_harmless_deleted"] = ipw(
        [
            (float(p["weight"]), removed(a0[(p["source_doc_id"], p["ss"], p["se"])], thr))
            for p in harm
        ]
    )
    return out


def clean() -> dict[str, Any]:
    from scripts.clean_mining import C, summarise

    private = json.loads((C / "units-private.json").read_text())
    labels = json.loads((C / "labels.json").read_text())
    units = [
        u
        for u in json.loads((C / "units.json").read_text())
        if not labels[u["unit_id"]]["uncertain"]
    ]
    rows = predict("clean", units)
    out: dict[str, Any] = {"throughput": throughput("clean", rows)}
    for mode in MODES:
        sc = {
            (p["source_doc_id"], p["ss"], p["se"]): as_scores(rows[p["unit_id"]], mode)
            if p["unit_id"] in rows
            else [0.0] * 4
            for p in private
        }
        out[f"llm_{mode}"] = summarise(private, labels, sc, ON)
    return out


def cascade(harmless: int | None) -> dict[str, Any]:
    """A0 at production thresholds, then the LLM may only un-delete what A0 removed.

    Every audit100 sensitive sentence A0 removes is judged. Of the harmless ones A0
    removes, a seeded uniform subsample is judged when ``harmless`` is set; the
    cascade's harmless deletion is then A0's rate times the share the LLM confirms.
    A span that does not match the sentence splitter stays removed, as under A0."""
    from perizia_privacy.corpus import sentence_spans
    from scripts.clean_mining import wilson
    from scripts.phase0_eval import audit_labels, doc_text
    from scripts.proof_eval import audit_frames
    from scripts.proof_package import unit

    thr = v1_thresholds()
    a0: dict[Key, list[float]] = {}
    for f, row in zip(audit_frames(), np.load(OUT / "eval" / "A0-audit.npy"), strict=True):
        key = (f["source_doc_id"], f["sentence_start"], f["sentence_end"])
        prev = a0.get(key, [0.0] * 4)
        a0[key] = [max(a, float(b)) for a, b in zip(prev, row[:4], strict=True)]
    labels = audit_labels()
    sens = [(d, r) for d, rs in labels.items() for r in rs if set(r["labels"]) & set(HEADS)]
    harm = [(d, r) for d, rs in labels.items() for r in rs if not r["labels"]]

    def a0_hit(d: str, r: dict[str, Any]) -> bool:
        s = a0.get((d, r["start"], r["end"]))
        return s is not None and any(x >= t for x, t in zip(s, thr, strict=True))

    sens_a0 = [x for x in sens if a0_hit(*x)]
    harm_a0 = [x for x in harm if a0_hit(*x)]
    judged = sorted(harm_a0, key=lambda x: (x[0], x[1]["start"]))
    if harmless is not None and harmless < len(judged):
        judged = random.Random(20261010).sample(judged, harmless)
    texts: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    units, ids = [], {}
    for d, r in sens_a0 + judged:
        if d not in texts:
            text = doc_text(d)
            texts[d] = (text, sentence_spans(text))
        text, spans = texts[d]
        if (r["start"], r["end"]) in spans:
            uid = opaque("audit100-cascade", d, r["start"], r["end"])
            ids[(d, r["start"], r["end"])] = uid
            units.append({"unit_id": uid, **unit(text, spans, spans.index((r["start"], r["end"])))})
    rows = predict("cascade-audit100", units)

    def kept_removed(d: str, r: dict[str, Any], mode: str) -> bool:
        uid = ids.get((d, r["start"], r["end"]))
        return uid is None or bool(heads(rows[uid], mode))

    out: dict[str, Any] = {
        "sensitive_removed_a0": f"{len(sens_a0)}/{len(sens)}",
        "harmless_deleted_a0": f"{len(harm_a0) / len(harm):.2%} ({len(harm_a0)}/{len(harm)})",
        "harmless_a0_removed_judged": len(judged),
        "unmatched_spans_left_removed": len(sens_a0) + len(judged) - len(units),
        "throughput": throughput("cascade-audit100", rows),
    }
    for mode in MODES:
        hit = sum(kept_removed(d, r, mode) for d, r in sens_a0)
        confirmed = sum(kept_removed(d, r, mode) for d, r in judged)
        out[f"cascade_{mode}"] = {
            "sensitive_removed": f"{hit}/{len(sens)}",
            "a0_harmless_deletions_confirmed": wilson(confirmed, len(judged)),
            "harmless_deleted": f"{len(harm_a0) / len(harm) * confirmed / len(judged):.2%}",
        }
    return out


STEPS = {
    "counterfactuals": counterfactuals,
    "sentenze": sentenze,
    "final": final,
    "clean": clean,
    "cascade": cascade,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=STEPS)
    ap.add_argument("--deletion", type=int, help="uniform subsample of the deletion set")
    ap.add_argument("--harmless", type=int, help="cascade: subsample of A0's harmless removals")
    args = ap.parse_args()
    private_dir(L)
    for step in args.steps:
        if step == "final":
            report = final(args.deletion)
        elif step == "cascade":
            report = cascade(args.harmless)
        else:
            report = STEPS[step]()
        report |= {
            "model": CHECKPOINT,
            "runtime": "LM Studio MLX 4-bit",
        }
        write_json(L / f"report-{step}.json", report)
        print(step, json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
