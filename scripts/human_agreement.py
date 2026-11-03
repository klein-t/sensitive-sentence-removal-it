"""A0's training labels (September small-model agreement) against the human annotator.

The two local annotator models were re-run on the 194-item blind validation sample
with the September prompt, schema, batching and input view (private runner). This
script only reads their per-window outputs and computes aggregates:

1. fidelity: on the 54 items whose reference is the September small-model label,
   the re-run is compared with the stored September outputs, per model and per
   window (the original batches, same members and order), and on the agreed
   item-level labels;
2. survival: A0's rule, a window counts only when both models are certain and
   give the same label set (five labels); an item survives only when every window
   of its sentence does, and its label is the union over windows (the convention
   the validation sample used for the v1 rows);
3. agreement, with the definitions of the annotation tool's ``validation_metrics``
   (pooled item x head kappa, sentence-any kappa, exact set agreement, per-head
   kappa): first the existing human-vs-reference figures are reproduced, then
   human vs small, human vs Sol and small vs Sol on identical item sets.

Sol labels for the 54 v1 items are optional (``sol-v1val/``): without them the Sol
comparisons cover the 140 items whose reference is already Sol.

Run: PRIVACY_DATA_ROOT=<private dir> uv run python -m scripts.human_agreement
Aggregates only go to ``results/human-agreement-20261103/report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("PRIVACY_DATA_ROOT", "data/private"))
TOOL = ROOT / "annotation-tool"
WORK = ROOT / "human-agreement-20261103"
SMALL = WORK / "small-v1"
SOL_V1 = WORK / "sol-v1val"
OUT = Path("results/human-agreement-20261103/report.json")
HEADS = ("salute", "penale", "minori", "famiglia")
LABELS = (*HEADS, "altre_categorie")
MODELS = ("privacy-qwen", "privacy-gemma")
V1 = "v1_validation_rows"


# ---- metric definitions, as in annotation-tool/scripts/validation_metrics.py -------
def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """pairs = (reference side, other side)."""
    c = Counter(pairs)
    return {
        "tp": c[(True, True)],
        "fn": c[(True, False)],
        "fp": c[(False, True)],
        "tn": c[(False, False)],
    }


def agreement(rows: list[dict], a: str, b: str) -> dict[str, Any]:
    """``a`` (reference side) vs ``b``: per head, pooled item x head, sentence-any, exact set."""
    out: dict[str, Any] = {"n_items": len(rows), "per_head": {}}
    pooled = []
    for h in HEADS:
        pairs = [(h in r[a], h in r[b]) for r in rows]
        pooled += pairs
        out["per_head"][h] = {
            f"{a}_positives": sum(x for x, _ in pairs),
            f"{b}_positives": sum(y for _, y in pairs),
            "kappa": kappa(pairs),
        }
    out["pooled_item_x_head"] = {"kappa": kappa(pooled), "confusion": confusion(pooled)}
    anyp = [(bool(r[a]), bool(r[b])) for r in rows]
    out["sentence_any_category"] = {"kappa": kappa(anyp), "confusion": confusion(anyp)}
    out["exact_set_agreement"] = (
        round(sum(sorted(r[a]) == sorted(r[b]) for r in rows) / len(rows), 4) if rows else None
    )
    return out


def three_way(rows: list[dict]) -> dict[str, Any]:
    return {
        "n_items": len(rows),
        "human_vs_small": agreement(rows, "human", "small"),
        "human_vs_sol": agreement(rows, "human", "sol"),
        "small_vs_sol": agreement(rows, "small", "sol"),
    }


# ---- inputs -----------------------------------------------------------------------
def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def human_rows() -> list[dict]:
    """Answered items, calibration exclusions removed (as validation_metrics.main)."""
    key = json.loads((TOOL / "data/key/validation_key.json").read_text())["items"]
    answers = json.loads((TOOL / "answers/validazione.json").read_text()).get("items", {})
    excl_path = TOOL / "data/key/calibration_exclusions.json"
    excluded = (
        json.loads(excl_path.read_text())["excluded_from_metrics"] if excl_path.exists() else {}
    )
    rows = []
    for k in key:
        if k["id"] in excluded:
            continue
        ans = answers.get(k["id"]) or {}
        cats = [c for c in ans.get("cats") or [] if c in HEADS]
        if not cats and ans.get("none") is not True:
            continue
        rows.append(
            {
                "id": k["id"],
                "human": cats,
                "teacher": k["teacher"],
                "provenance": k["label_provenance"],
                "source_set": k["source_set"],
            }
        )
    return rows


def windows(kind: str) -> dict[str, dict[str, dict]]:
    """model -> window id -> record, for the 'fidelity' or 'new' run."""
    out: dict[str, dict[str, dict]] = {}
    for m in MODELS:
        p = SMALL / f"{m}-{kind}.jsonl"
        out[m] = {r["id"]: r for r in jsonl(p)} if p.exists() else {}
    return out


def eligible(a: dict, b: dict) -> bool:
    return set(a["labels"]) == set(b["labels"]) and not a["uncertain"] and not b["uncertain"]


def item_labels(per_model: dict[str, dict[str, dict]], stored: bool = False) -> dict[str, dict]:
    """item -> {complete, survives, labels (heads, union over windows)} under A0's rule."""
    lab, unc = ("stored_labels", "stored_uncertain") if stored else ("labels", "uncertain")
    q, g = per_model[MODELS[0]], per_model[MODELS[1]]
    items: dict[str, list[str]] = {}
    for w, r in q.items():
        if r.get("item"):
            items.setdefault(r["item"], []).append(w)
    out = {}
    for item, ws in items.items():
        if not all(w in g for w in ws):
            out[item] = {"complete": False, "survives": False, "labels": []}
            continue
        pairs = [
            (
                {"labels": q[w][lab], "uncertain": q[w][unc]},
                {"labels": g[w][lab], "uncertain": g[w][unc]},
            )
            for w in ws
        ]
        ok = all(eligible(a, b) for a, b in pairs)
        union = {h for a, _ in pairs for h in a["labels"]}
        out[item] = {
            "complete": True,
            "survives": ok,
            "labels": [h for h in HEADS if h in union] if ok else [],
            "windows": len(ws),
            "both_certain": all(not a["uncertain"] and not b["uncertain"] for a, b in pairs),
            "same_set": all(set(a["labels"]) == set(b["labels"]) for a, b in pairs),
            "heads_agree": {
                h: all((h in a["labels"]) == (h in b["labels"]) for a, b in pairs) for h in LABELS
            },
        }
    return out


def sol_v1() -> dict[str, list[str]]:
    """Sol labels for the v1 items, certain only. Accepts JSON or JSONL records with an
    item ``id`` and ``labels``/``remove_heads``/``heads`` (+ optional ``uncertain``),
    or a JSON object keyed by item id."""
    out: dict[str, list[str]] = {}
    if not SOL_V1.is_dir():
        return out
    for p in sorted(SOL_V1.rglob("*")):
        if p.suffix not in (".json", ".jsonl") or "manifest" in p.name:
            continue
        try:
            data: Any = jsonl(p) if p.suffix == ".jsonl" else json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        recs = (
            [{"id": k, **v} for k, v in data.items() if isinstance(v, dict)]
            if isinstance(data, dict)
            else [r for r in data if isinstance(r, dict)]
        )
        for r in recs:
            iid = r.get("id") or r.get("item_id")
            labs = next(
                (r[k] for k in ("labels", "remove_heads", "heads") if isinstance(r.get(k), list)),
                None,
            )
            if not isinstance(iid, str) or labs is None or r.get("uncertain"):
                continue
            out[iid] = [h for h in HEADS if h in labs]
    return out


# ---- blocks -----------------------------------------------------------------------
def fidelity(
    fid: dict[str, dict[str, dict]], v1_ids: set[str], teacher: dict[str, list[str]]
) -> dict:
    res: dict[str, Any] = {"per_model": {}}
    for m in MODELS:
        recs = list(fid[m].values())
        blk: dict[str, Any] = {}
        for scope, rs in (
            ("all_rows_in_original_batches", recs),
            ("v1_target_windows", [r for r in recs if r.get("item")]),
        ):
            n = len(rs)
            blk[scope] = {
                "rows": n,
                "exact_label_set_match": round(
                    sum(set(r["labels"]) == set(r["stored_labels"]) for r in rs) / n, 4
                )
                if n
                else None,
                "uncertain_flag_match": round(
                    sum(r["uncertain"] == r["stored_uncertain"] for r in rs) / n, 4
                )
                if n
                else None,
                "per_label": {
                    h: {
                        "match_rate": round(
                            sum((h in r["labels"]) == (h in r["stored_labels"]) for r in rs) / n, 4
                        ),
                        "stored_positives": sum(h in r["stored_labels"] for r in rs),
                        "rerun_positives": sum(h in r["labels"] for r in rs),
                        "kappa": kappa([(h in r["stored_labels"], h in r["labels"]) for r in rs]),
                    }
                    for h in LABELS
                }
                if n
                else {},
                "batch_size_changed": sum(r["batch_size"] != r["stored_batch_size"] for r in rs),
            }
        res["per_model"][m] = blk
    rerun, stored = item_labels(fid), item_labels(fid, stored=True)
    ids = sorted(v1_ids & set(rerun))
    res["stored_items_reconstructed"] = len(ids)
    res["stored_agreed_label_equals_key_reference"] = sum(
        stored[i]["survives"] and stored[i]["labels"] == sorted(teacher[i], key=HEADS.index)
        for i in ids
    )
    surv = [i for i in ids if rerun[i]["survives"]]
    res["agreed_items"] = {
        "items": len(ids),
        "rerun_survives": len(surv),
        "rerun_exact_set_equal_stored_on_survivors": sum(
            rerun[i]["labels"] == stored[i]["labels"] for i in surv
        ),
        "per_head_match_rate_strict": {
            h: round(
                sum(
                    rerun[i]["survives"]
                    and ((h in rerun[i]["labels"]) == (h in stored[i]["labels"]))
                    for i in ids
                )
                / len(ids),
                4,
            )
            for h in HEADS
        }
        if ids
        else {},
        "per_head_match_rate_on_survivors": {
            h: round(
                sum((h in rerun[i]["labels"]) == (h in stored[i]["labels"]) for i in surv)
                / len(surv),
                4,
            )
            for h in HEADS
        }
        if surv
        else {},
    }
    return res


def survival(items: dict[str, dict], ids: list[str]) -> dict[str, Any]:
    rs = [items[i] for i in ids if i in items]
    done = [r for r in rs if r["complete"]]
    return {
        "items": len(ids),
        "labelled_by_both": len(done),
        "both_certain": sum(r["both_certain"] for r in done),
        "same_label_set": sum(r["same_set"] for r in done),
        "survive_a0_rule": sum(r["survives"] for r in done),
        "windows": sum(r["windows"] for r in done),
        "head_agreement_items": {h: sum(r["heads_agree"][h] for r in done) for h in LABELS},
        "head_agreement_items_both_certain": {
            h: sum(r["heads_agree"][h] and r["both_certain"] for r in done) for h in LABELS
        },
        "surviving_positive_items": {
            h: sum(r["survives"] and h in r["labels"] for r in done) for h in HEADS
        },
    }


def check(value: Any, path: str = "") -> None:
    """Data rules: short strings only, no codici fiscali, no identifiers."""
    if isinstance(value, dict):
        for k, v in value.items():
            check(k, path)
            check(v, f"{path}/{k}")
    elif isinstance(value, list):
        for v in value:
            check(v, path)
    elif isinstance(value, str):
        bad = (
            len(value) > 120
            or re.search(r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]", value)
            or re.search(r"[0-9a-f]{16,}", value)
            or re.fullmatch(r"v[0-9a-f]{10}", value)
        )
        if bad:
            sys.exit(f"aggregate check failed at {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    key = json.loads((TOOL / "data/key/validation_key.json").read_text())["items"]
    teacher = {k["id"]: k["teacher"] for k in key}
    v1_ids = {k["id"] for k in key if k["source_set"] == V1}
    other_ids = sorted({k["id"] for k in key} - v1_ids)
    fid, new = windows("fidelity"), windows("new")
    small = item_labels(fid) | item_labels(new)
    sol1 = sol_v1()
    rows = human_rows()

    # 1. reproduce the existing human-vs-reference figures
    reproduce = {
        "all_answered": agreement(rows, "human", "teacher"),
        "by_label_provenance": {
            p: agreement([r for r in rows if r["provenance"] == p], "human", "teacher")
            for p in sorted({r["provenance"] for r in rows})
        },
    }

    # 2. identical item sets: answered, small survives, Sol label available
    for r in rows:
        s = small.get(r["id"])
        r["small"] = s["labels"] if s and s["survives"] else None
        r["sol"] = teacher[r["id"]] if r["source_set"] != V1 else sol1.get(r["id"])
    core = [r for r in rows if r["small"] is not None and r["sol"] is not None]
    comparisons: dict[str, Any] = {
        "all_with_small_and_sol": three_way(core),
        "sol_reference_items": three_way([r for r in core if r["source_set"] != V1]),
        "v1_items": three_way([r for r in core if r["source_set"] == V1]),
    }
    dropped = [r for r in rows if r["small"] is None and r["source_set"] != V1]
    comparisons["sol_reference_non_survivors_human_vs_sol"] = agreement(dropped, "human", "sol")
    surv_answered = [r for r in rows if r["small"] is not None]
    comparisons["human_vs_small_all_survivors"] = agreement(surv_answered, "human", "small")
    v1_surv = [r for r in surv_answered if r["source_set"] == V1]
    comparisons["v1_survivors_human_vs_september_stored"] = agreement(v1_surv, "human", "teacher")
    comparisons["v1_survivors_human_vs_rerun"] = agreement(v1_surv, "human", "small")

    report = {
        "note": "stratified sample; kappas are sample-level, not population estimates",
        "a0_rule": "window kept iff both certain, same 5-label set; item iff all windows kept",
        "answered_items": len(rows),
        "fidelity_v1": fidelity(fid, v1_ids, teacher),
        "survival": {
            "v1_items": survival(small, sorted(v1_ids)),
            "other_items": survival(small, other_ids),
            "answered_items_surviving": len(surv_answered),
        },
        "sol_v1_labels_available": len(sol1),
        "reproduce_existing": reproduce,
        "comparisons": comparisons,
    }
    for m in MODELS:
        for kind in ("fidelity", "new"):
            p = SMALL / f"{m}-{kind}.run.json"
            if p.exists():
                run = json.loads(p.read_text())
                report.setdefault("runs", {})[f"{m}/{kind}"] = {
                    k: run[k]
                    for k in (
                        "rows_this_run",
                        "batches_this_run",
                        "errors",
                        "schema_recovery_requests",
                        "wall_seconds",
                        "concurrency",
                    )
                }
    check(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
