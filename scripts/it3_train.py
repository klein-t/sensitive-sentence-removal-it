"""it3: fine-tune FROM v1 with counterfactual pairs, then pick thresholds on dev.

The it1 recipe (v1 init, lr 1e-5, batch 32, half v1 replay per update) with a
new half drawn from four streams: the training half of the fictional edge
suite (current-policy relabels), it1's Sol units, the real-derived
counterfactual pairs and the Cassazione negation sentences. A pair (original
and edited twin, every window of each) is drawn as one bundle and always lands
in the same microbatch. `--margin` adds a small hinge on the logits so a
should-keep twin scores below its original on the original's heads.

`thresholds` chooses per-head thresholds on the train-side dev set: the
minimum IPW harmless deletion with dev recall (overall and per head) at least
v1's at its production thresholds. Heads with fewer than MIN_DEV_POSITIVES dev
positives may only go down from A0's threshold. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from typing import Any

from perizia_privacy.corpus import example_text, windows
from scripts.final_instrument import score_v1, sentence_windows
from scripts.it3_data import I3, check, kept_pairs
from scripts.phase0_eval import doc_text
from scripts.proof_score import (
    HEADS,
    LABELS,
    V1,
    encode,
    load_model,
    private_dir,
    v1_thresholds,
    write_json,
)
from scripts.proof_train import (
    NEW_ROW_MASK,
    edge_examples,
    edge_relabels,
    replay_examples,
    sol_examples,
)
from scripts.sentenze import decisions

MIX = {"edge": 0.40, "pairs": 0.35, "sol": 0.15, "cass": 0.10}
KEEP_KINDS = ("negated", "deceased", "company", "hypothetical")
MARGIN = 1.0
MARGIN_WEIGHT = 0.2
MIN_DEV_POSITIVES = 20
Row = dict[str, Any]


def span_rows(text: str, start: int, end: int, heads: list[str], side: str) -> list[Row]:
    """Every window overlapping [start, end), labelled with the sentence's heads."""
    return [
        {"text": example_text(text, a, b), "labels": heads, "mask": NEW_ROW_MASK, "side": side}
        for a, b, _, _ in windows(text)
        if a < end and b > start
    ]


def pair_bundles() -> list[dict[str, Any]]:
    out = []
    for p in kept_pairs():
        text = doc_text(p["source_doc_id"])
        new = text[: p["ss"]] + p["edited"] + text[p["se"] :]
        rows = span_rows(text, p["ss"], p["se"], p["heads"], "original")
        rows += span_rows(new, p["ss"], p["ss"] + len(p["edited"]), p["twin_heads"], "twin")
        out.append(
            {
                "kind": p["kind"],
                "rows": rows,
                "margin_heads": [h for h in p["heads"] if h in HEADS]
                if p["kind"] in KEEP_KINDS
                else [],
            }
        )
    return out


def cass_rows() -> list[Row]:
    private = json.loads((I3 / "cass-private.json").read_text())
    labels = json.loads((I3 / "cass-labels.json").read_text())
    texts = {d["id"]: d["text"] for d in decisions()}
    out = []
    for p in private:
        lab = labels.get(p["unit_id"])
        if lab is None or lab["uncertain"]:  # uncertain = loss-masked, i.e. no gradient
            continue
        out += span_rows(texts[p["decision"]], p["ss"], p["se"], lab["remove_heads"], "cass")
    return out


def pack(singles: list[Row], bundles: list[list[Row]], micro: int) -> list[list[Row]]:
    """Microbatches of about `micro` rows; a bundle is never split."""
    total = len(singles) + sum(map(len, bundles))
    out: list[list[Row]] = [[] for _ in range(max(1, math.ceil(total / micro)))]
    for b in sorted(bundles, key=len, reverse=True):
        min(out, key=len).extend(b)
    for r in singles:
        min(out, key=len).append(r)
    return [mb for mb in out if mb]


def mix_counts(n: int, rng: random.Random) -> Counter[str]:
    return Counter(rng.choices(list(MIX), weights=list(MIX.values()), k=n))


def train(args: argparse.Namespace) -> None:
    exclusion = check()
    out_dir = private_dir(I3 / args.out)
    replay = replay_examples()
    streams: dict[str, list[Any]] = {
        "edge": edge_examples(train=True, relabel=edge_relabels()),
        "sol": sol_examples(dev=False),
        "cass": cass_rows(),
        "pairs": pair_bundles(),
    }
    counts = {
        "replay_rows": len(replay),
        "edge_rows": len(streams["edge"]),
        "it1_sol_rows": len(streams["sol"]),
        "cass_rows": len(streams["cass"]),
        "cass_positive_rows": sum(bool(r["labels"]) for r in streams["cass"]),
        "pairs": len(streams["pairs"]),
        "pairs_by_kind": Counter(b["kind"] for b in streams["pairs"]),
        "pair_rows": sum(len(b["rows"]) for b in streams["pairs"]),
        "mix_of_new_half": MIX,
    }
    print(json.dumps(counts), flush=True)

    torch, tok, model = load_model(V1)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    pos_weight = torch.tensor(
        [json.loads((V1 / "manifest.json").read_text())["pos_weight"][k] for k in LABELS],
        device="mps",
    )
    every = replay + streams["edge"] + streams["sol"] + streams["cass"]
    every += [r for b in streams["pairs"] for r in b["rows"]]
    for r in every:
        r["ids"] = encode(tok, r["text"])
        r["y"] = [1.0 if k in r["labels"] else 0.0 for k in LABELS]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, args.updates // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (args.updates - s) / (args.updates - warm)))
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    buffers: dict[str, list[Any]] = {}

    def draw(name: str, pool: list[Any]) -> Any:
        buf = buffers.setdefault(name, [])
        if not buf:
            buf.extend(rng.sample(pool, len(pool)))
        return buf.pop()

    half = args.batch // 2
    t0 = time.time()
    log, margins = [], []
    for step in range(args.updates):
        singles = [draw("replay", replay) for _ in range(half)]
        want = mix_counts(args.batch - half, rng)
        bundles: list[dict[str, Any]] = []
        for name in ("edge", "sol", "cass"):
            singles += [draw(name, streams[name]) for _ in range(want[name])]
        got = 0
        while got < want["pairs"]:
            b = draw("pairs", streams["pairs"])
            bundles.append(b)
            got += len(b["rows"])
        ids = {id(r): b for b in bundles for r in b["rows"]}
        n_rows = len(singles) + got
        n_margin = sum(bool(b["margin_heads"]) for b in bundles)
        opt.zero_grad(set_to_none=True)
        total = margin_total = 0.0
        for mb in pack(singles, [b["rows"] for b in bundles], args.micro):
            width = max(len(r["ids"]) for r in mb)
            x = torch.full((len(mb), width), tok.pad_token_id, dtype=torch.long)
            a = torch.zeros((len(mb), width), dtype=torch.long)
            for j, r in enumerate(mb):
                x[j, : len(r["ids"])] = torch.tensor(r["ids"])
                a[j, : len(r["ids"])] = 1
            y = torch.tensor([r["y"] for r in mb], device="mps")
            m = torch.tensor([r["mask"] for r in mb], device="mps")
            logits = model(input_ids=x.to("mps"), attention_mask=a.to("mps")).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none"
            )
            loss = (loss * m).sum() / (m.sum() * n_rows / len(mb))
            if args.margin and n_margin:
                hinge = margin_loss(torch, logits, mb, ids)
                if hinge is not None:
                    loss = loss + MARGIN_WEIGHT * hinge / n_margin
                    margin_total += float(hinge.item())
            loss.backward()
            total += float(loss.item())
        if not math.isfinite(total):
            raise SystemExit("nonfinite loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log.append(round(total, 5))
        margins.append(round(margin_total / max(n_margin, 1), 5))
        if (step + 1) % 25 == 0:
            print(
                "update",
                step + 1,
                "loss",
                round(sum(log[-25:]) / 25, 4),
                "hinge",
                round(sum(margins[-25:]) / 25, 4),
                round(time.time() - t0),
                "s",
                flush=True,
            )
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    write_json(
        out_dir / "it3-train.json",
        counts
        | {
            "init": str(V1),
            "updates": args.updates,
            "lr": args.lr,
            "batch": args.batch,
            "mix": "half v1 replay, half new (edge, it1 Sol, pairs, Cassazione)",
            "pair_aware_batching": True,
            "margin": {"on": args.margin, "margin": MARGIN, "weight": MARGIN_WEIGHT},
            "loss": log,
            "hinge": margins,
            "seconds": round(time.time() - t0),
            "seed": args.seed,
            "exclusion_check": exclusion,
            "production_ready": False,
        },
    )
    print("saved", round(time.time() - t0), "s", flush=True)


def margin_loss(torch: Any, logits: Any, mb: list[Row], bundle_of: dict[int, Any]) -> Any:
    """Sum over pairs in the microbatch of relu(MARGIN - (original - twin)) per head."""
    pairs: dict[int, dict[str, list[int]]] = {}
    for j, r in enumerate(mb):
        b = bundle_of.get(id(r))
        if b is not None and b["margin_heads"]:
            pairs.setdefault(id(b), {"b": b, "original": [], "twin": []})[r["side"]].append(j)
    terms = []
    for p in pairs.values():
        if not p["original"] or not p["twin"]:
            continue
        cols = [LABELS.index(h) for h in p["b"]["margin_heads"]]
        o = logits[p["original"]][:, cols].max(dim=0).values
        t = logits[p["twin"]][:, cols].max(dim=0).values
        terms.append(torch.relu(MARGIN - (o - t)).mean())
    return torch.stack(terms).sum() if terms else None


# --- thresholds on dev -----------------------------------------------------------


def dev_scores(model: str) -> dict[tuple[str, int, int], list[float]]:
    path = private_dir(I3 / "scores") / f"{model}-dev.json"
    if path.exists():
        return {tuple(k): v for k, v in json.loads(path.read_text())}
    private = json.loads((I3 / "dev-private.json").read_text())
    private += json.loads((I3 / "dev-pos-private.json").read_text())
    by_doc: dict[str, set[tuple[int, int]]] = {}
    for p in private:
        by_doc.setdefault(p["source_doc_id"], set()).add((p["ss"], p["se"]))
    items = [
        ((src, *span), e)
        for src, spans in sorted(by_doc.items())
        for span, examples in sentence_windows(doc_text(src)).items()
        if span in spans
        for e in examples
    ]
    sc = score_v1({model: V1 if model == "A0" else I3 / MODEL_DIRS[model]}, items)[model]
    write_json(path, [[list(k), v] for k, v in sc.items()])
    return sc


MODEL_DIRS = {"it3": "model", "it3m": "model-margin"}


def dev_rows(model: str) -> list[dict[str, Any]]:
    labels = json.loads((I3 / "dev-labels.json").read_text())
    sc = dev_scores(model)
    out = []
    for p in json.loads((I3 / "dev-private.json").read_text()):
        lab = labels[p["unit_id"]]
        if lab["uncertain"]:
            continue
        s = sc.get((p["source_doc_id"], p["ss"], p["se"]), [0.0] * 4)
        out.append({"s": s, "heads": lab["remove_heads"], "w": p["weight"]})
    # Cue-mined supplement: its positives count for recall, never for deletion.
    labels = json.loads((I3 / "dev-pos-labels.json").read_text())
    for p in json.loads((I3 / "dev-pos-private.json").read_text()):
        lab = labels[p["unit_id"]]
        if not lab["uncertain"] and lab["remove_heads"]:
            s = sc.get((p["source_doc_id"], p["ss"], p["se"]), [0.0] * 4)
            out.append({"s": s, "heads": lab["remove_heads"], "w": 0.0})
    return out


def dev_metrics(rows: list[dict[str, Any]], thr: list[float]) -> dict[str, Any]:
    hit = [any(x >= t for x, t in zip(r["s"], thr, strict=True)) for r in rows]
    pos = [i for i, r in enumerate(rows) if r["heads"]]
    harm = [i for i, r in enumerate(rows) if not r["heads"]]
    w = sum(rows[i]["w"] for i in harm)
    return {
        "recall": sum(hit[i] for i in pos),
        "positives": len(pos),
        "per_head": {
            h: [
                sum(hit[i] for i in pos if h in rows[i]["heads"]),
                sum(h in rows[i]["heads"] for i in pos),
            ]
            for h in HEADS
        },
        "ipw_deletion": sum(rows[i]["w"] * hit[i] for i in harm) / w if w else 0.0,
    }


def choose(rows: list[dict[str, Any]], floor: dict[str, Any], start: list[float]) -> list[float]:
    """Coordinate descent: lowest deletion with recall (overall and per head) >= floor."""

    def ok(m: dict[str, Any]) -> bool:
        return m["recall"] >= floor["recall"] and all(
            m["per_head"][h][0] >= floor["per_head"][h][0] for h in HEADS
        )

    thr = start[:]
    grid = sorted({round(10 ** (-4 + 4 * i / 400), 6) for i in range(401)} | set(start))
    best = dev_metrics(rows, thr)["ipw_deletion"] if ok(dev_metrics(rows, thr)) else math.inf
    for _ in range(5):
        moved = False
        for h, head in enumerate(HEADS):
            tunable = floor["per_head"][head][1] >= MIN_DEV_POSITIVES
            for t in grid:
                if not tunable and t > start[h]:
                    continue
                trial = thr[:h] + [t] + thr[h + 1 :]
                m = dev_metrics(rows, trial)
                if ok(m) and m["ipw_deletion"] < best - 1e-12:
                    best, thr, moved = m["ipw_deletion"], trial, True
        if not moved:
            break
    return thr


def thresholds(model: str) -> None:
    a0_rows = dev_rows("A0")
    floor = dev_metrics(a0_rows, v1_thresholds())
    rows = dev_rows(model)
    thr = choose(rows, floor, v1_thresholds())
    report = {
        "model": model,
        "thresholds": thr,
        "rule": "min IPW dev deletion s.t. dev recall overall and per head >= A0's at production",
        "min_dev_positives_to_raise_a_head": MIN_DEV_POSITIVES,
        "dev_A0_production": floor,
        "dev_model_A0_fixed": dev_metrics(rows, v1_thresholds()),
        "dev_model_chosen": dev_metrics(rows, thr),
    }
    write_json(I3 / f"thresholds-{model}.json", report)
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("train", "thresholds"))
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin", action="store_true")
    ap.add_argument("--model", default="it3")
    args = ap.parse_args()
    args.out = MODEL_DIRS["it3m" if args.margin else "it3"]
    if args.step == "train":
        train(args)
    else:
        thresholds(args.model)


if __name__ == "__main__":
    main()
