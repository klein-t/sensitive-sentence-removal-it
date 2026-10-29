"""One-day proof, step 5: fine-tune FROM v1 with replay plus new rows.

Every update mixes half v1's original training rows (replay, keeps recall) and
half new rows: Sol-labeled positives and hard negatives (train groups only),
and the training half of the edge suite expanded to every sentence of each
paragraph (sensitive target with harmless neighbours). altre_categorie is
loss-masked on new rows; uncertain Sol units are excluded. Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import example_text, sentence_spans, windows
from scripts.proof_score import (
    HEADS,
    LABELS,
    OUT,
    ROOT,
    V1,
    edge_rows,
    encode,
    load_model,
    opaque,
    private_dir,
    write_json,
)

NEW_ROW_MASK = [1.0, 1.0, 1.0, 1.0, 0.0]


def edge_family_train(row: dict[str, Any]) -> bool:
    """Half of the template families (group, pattern) train; the rest are held out."""
    return int(opaque("edge", row["group"], row["pattern"]), 16) % 2 == 0


def target_sentences(text: str, start: int, end: int) -> set[tuple[int, int]]:
    """Sentences overlapping the target's non-space characters."""
    return {
        (a, b)
        for a, b in sentence_spans(text)
        if any(not text[i].isspace() for i in range(max(a, start), min(b, end)))
    }


def sentence_examples(text: str) -> list[tuple[tuple[int, int], str]]:
    return [((s0, s1), example_text(text, a, b)) for a, b, s0, s1 in windows(text)]


def edge_relabels() -> dict[str, list[str]]:
    """Iteration 2: Sol's current-policy heads per edge row (uncertain rows keep theirs)."""
    labels = json.loads((OUT / "iter2-labels.json").read_text())
    rows = json.loads((OUT / "iter2-edge-map.json").read_text())
    return {row: labels[u]["heads"] for u, row in rows.items() if not labels[u]["uncertain"]}


def edge_examples(train: bool, relabel: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    out = []
    for row in edge_rows():
        if edge_family_train(row) != train:
            continue
        if relabel is not None and row["id"] in relabel:
            heads = relabel[row["id"]]
            row = row | {"expected_labels": heads, "polarity": "remove" if heads else "keep"}
        text = row["paragraph"]
        start = text.index(row["target"])
        targets = target_sentences(text, start, start + len(row["target"]))
        for span, example in sentence_examples(text):
            labels = row["expected_labels"] if span in targets else []
            out.append(
                {
                    "text": example,
                    "labels": labels,
                    "mask": NEW_ROW_MASK,
                    "row": row["id"],
                    "is_target": span in targets,
                    "polarity": row["polarity"],
                }
            )
    return out


def sol_examples(
    dev: bool | None,
    stem: str = "",
    unit_set: str | None = None,
    groups: set[str] | None = None,
    exclude: bool = False,
) -> list[dict[str, Any]]:
    """Windows of Sol-labeled units; dev=None takes both iteration 1 slices."""
    labels = json.loads((OUT / f"{stem}labels.json").read_text())
    units = json.loads((OUT / f"{stem}units-private.json").read_text())
    paths = {
        d["source_doc_id"]: d["path"]
        for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }
    out = []
    for u in units:
        lab = labels.get(u["unit_id"])
        if lab is None or lab["uncertain"]:
            continue
        if (dev is not None and u.get("dev") != dev) or (unit_set and u.get("set") != unit_set):
            continue
        if groups is not None and (u["group"] in groups) == exclude:
            continue
        text = Path(paths[u["source_doc_id"]]).read_text(encoding="utf-8")
        for a, b, s0, s1 in windows(text):
            if (s0, s1) == (u["ss"], u["se"]):
                out.append(
                    {
                        "text": example_text(text, a, b),
                        "labels": lab["heads"],
                        "mask": NEW_ROW_MASK,
                        "kind": u["kind"],
                        "unit": u["unit_id"],
                    }
                )
    return out


def replay_examples() -> list[dict[str, Any]]:
    with (ROOT / "development-preparation-v3/training-inputs-v3/train.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    return [{"text": r["text"], "labels": r["labels"], "mask": [1.0] * 5} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--out", default="model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iter2", action="store_true", help="add iteration 2 rows and edge relabels")
    args = ap.parse_args()
    out_dir = private_dir(OUT / args.out)
    replay = replay_examples()
    if args.iter2:
        from scripts.proof_iter2 import calibration_groups

        cal = calibration_groups()
        new = sol_examples(None, groups=cal, exclude=True)
        new += sol_examples(None, "iter2-", "train", groups=cal, exclude=True)
        new += edge_examples(train=True, relabel=edge_relabels())
    else:
        new = sol_examples(dev=False) + edge_examples(train=True)
    counts = {
        "replay_rows": len(replay),
        "sol_rows": sum("unit" in r for r in new),
        "sol_positive_rows": sum(bool(r["labels"]) for r in new if "unit" in r),
        "edge_rows": sum("row" in r for r in new),
        "edge_positive_rows": sum(bool(r["labels"]) for r in new if "row" in r),
    }
    print(json.dumps(counts), flush=True)

    torch, tok, model = load_model(V1)
    model.gradient_checkpointing_enable(
        {"use_reentrant": False}
    )  # v1 was fit this way; keeps MPS memory bounded
    model.train()
    pos_weight = torch.tensor(
        [json.loads((V1 / "manifest.json").read_text())["pos_weight"][k] for k in LABELS],
        device="mps",
    )
    for rows in (replay, new):
        for r in rows:
            r["ids"] = encode(tok, r["text"])
            r["y"] = [1.0 if k in r["labels"] else 0.0 for k in LABELS]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, args.updates // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (args.updates - s) / (args.updates - warm)))
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    streams = {"replay": [], "new": []}

    def draw(name: str, pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        buf = streams[name]
        while len(buf) < k:
            more = pool[:]
            rng.shuffle(more)
            buf.extend(more)
        taken, streams[name] = buf[:k], buf[k:]
        return taken

    half = args.batch // 2
    t0 = time.time()
    log = []
    for step in range(args.updates):
        batch = draw("replay", replay, half) + draw("new", new, args.batch - half)
        rng.shuffle(batch)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(0, len(batch), args.micro):
            mb = batch[k : k + args.micro]
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
            loss = (loss * m).sum() / (m.sum() * len(batch) / len(mb))
            loss.backward()
            total += float(loss.item())
        if not torch.isfinite(torch.tensor(total)):
            raise SystemExit("nonfinite loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log.append(round(total, 5))
        if (step + 1) % 25 == 0:
            recent = sum(log[-25:]) / 25
            print(
                "update",
                step + 1,
                "loss",
                round(recent, 4),
                round(time.time() - t0),
                "s",
                flush=True,
            )
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    write_json(
        out_dir / "proof-train.json",
        counts
        | {
            "init": str(V1),
            "updates": args.updates,
            "lr": args.lr,
            "batch": args.batch,
            "mix": "half replay, half new",
            "loss": log,
            "seconds": round(time.time() - t0),
            "heads_trained_on_new_rows": list(HEADS),
            "seed": args.seed,
            "iter2": args.iter2,
            "production_ready": False,
        },
    )
    print("saved", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
