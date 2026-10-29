"""Phase 0: train the factored student fresh from the base encoder.

Real rows only, no synthetic: Sol-labelled units from iterations 1 and 2
(old schema, mapped to the factors it determines) plus v1's real training rows
(remove labels only; topics and factors loss-masked). Sol supersedes v1 on the
same sentence. The probe groups are held out. ``--fraction`` takes a nested
subset of document groups for the learning curve. Writes weights, a manifest
and a progress log to the private directory; prints counts and losses only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from perizia_privacy import factored as fs
from perizia_privacy.corpus import sentence_spans, windows
from scripts.proof_score import OUT, ROOT, opaque, private_dir

P0 = ROOT / "phase0-20261003"
BASE = ROOT / "base-model-safe"
NO_TOPIC = {"building_or_property", "civil_enforcement", "procedural_boilerplate"}
NO_TOPIC |= {"no_special_category"}
SUBJECT_OF = {
    "not_living": "deceased",
    "company_only_non_criminal": "company",
    "hypothetical_person": "hypothetical",
}
AUX_WEIGHT = 0.5


def factors_from_old(lab: dict[str, Any]) -> dict[str, Any]:
    """What the old Sol schema determines; everything else stays unknown (None)."""
    topic: list[float | None] = [None] * 4
    subject = SUBJECT_OF.get(lab["reason_code"])
    assertion = "negated" if lab["reason_code"] == "negated_or_absent" else None
    if lab["heads"]:
        assertion = "affirmed"
        for h in lab["heads"]:
            topic[fs.HEADS.index(h)] = 1.0
    elif lab["reason_code"] in NO_TOPIC:
        topic = [0.0] * 4
    return {"topic": topic, "subject": subject, "assertion": assertion}


def real_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One row per (document, window); Sol units first, then v1 real rows."""
    probe = {p["group"] for p in json.loads((P0 / "probe-private.json").read_text())}
    docs = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    paths = {d["source_doc_id"]: d["path"] for d in docs}
    texts: dict[str, str] = {}

    def text_of(src: str) -> str:
        if src not in texts:
            texts[src] = Path(paths[src]).read_text(encoding="utf-8")
        return texts[src]

    rows: list[dict[str, Any]] = []
    stats = {"sol_units": 0, "sol_uncertain": 0, "probe_held_out": 0, "v1_superseded": 0}
    seen: set[tuple[str, int, int]] = set()
    labels = json.loads((OUT / "labels.json").read_text())
    labels |= json.loads((OUT / "iter2-labels.json").read_text())
    units = json.loads((OUT / "units-private.json").read_text())
    units += json.loads((OUT / "iter2-units-private.json").read_text())
    for u in units:
        lab = labels.get(u["unit_id"])
        if lab is None:
            continue
        if u["group"] in probe:
            stats["probe_held_out"] += 1
            continue
        if lab["uncertain"]:
            stats["sol_uncertain"] += 1
            continue
        stats["sol_units"] += 1
        src, key = u["source_doc_id"], (u["source_doc_id"], u["ss"], u["se"])
        seen.add(key)
        text = text_of(src)
        index = sentence_spans(text).index((u["ss"], u["se"]))
        for a, b, s0, s1 in windows(text):
            if (s0, s1) == (u["ss"], u["se"]):
                rows.append(
                    {
                        "src": src,
                        "group": u["group"],
                        "start": a,
                        "end": b,
                        "index": index,
                        "remove": [float(h in lab["heads"]) for h in fs.HEADS],
                        "source": "sol",
                    }
                    | factors_from_old(lab)
                )
    path = ROOT / "development-preparation-v3/training-inputs-v3/train.jsonl"
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["source"] != "real":
                continue
            if r["doc_id"] in probe:
                stats["probe_held_out"] += 1
                continue
            key = (r["source_doc_id"], r["sentence_start"], r["sentence_end"])
            if key in seen:
                stats["v1_superseded"] += 1
                continue
            spans = sentence_spans(text_of(r["source_doc_id"]))
            rows.append(
                {
                    "src": r["source_doc_id"],
                    "group": r["doc_id"],
                    "start": r["start"],
                    "end": r["end"],
                    "index": spans.index(key[1:]),
                    "remove": [float(h in r["labels"]) for h in fs.HEADS],
                    "source": "v1",
                    "topic": [None] * 4,
                    "subject": None,
                    "assertion": None,
                }
            )
    for r in rows:
        r["text"] = text_of(r["src"])
    return rows, stats


def subset(rows: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    """Nested subsets: groups ordered by a fixed hash, the first `fraction` kept."""
    groups = sorted({r["group"] for r in rows}, key=lambda g: opaque("curve", g))
    keep = set(groups[: round(len(groups) * fraction)])
    return [r for r in rows if r["group"] in keep]


def factored_loss(torch: Any, model: Any, tok: Any, mb: list[dict[str, Any]], pw: Any) -> Any:
    """Remove BCE plus masked topic BCE and masked subject/assertion CE."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    ce = torch.nn.functional.cross_entropy
    logits = model(**fs.batch_tensors(torch, [r["enc"] for r in mb], tok.pad_token_id, "mps"))
    y = torch.tensor([r["remove"] for r in mb], device="mps")
    loss = bce(logits[:, fs.REMOVE], y, pos_weight=pw)
    topic_y = torch.tensor(
        [[0.0 if v is None else v for v in r["topic"]] for r in mb], device="mps"
    )
    topic_m = torch.tensor(
        [[0.0 if v is None else 1.0 for v in r["topic"]] for r in mb], device="mps"
    )
    if topic_m.sum() > 0:
        t_loss = bce(logits[:, fs.TOPIC], topic_y, reduction="none")
        loss = loss + AUX_WEIGHT * (t_loss * topic_m).sum() / topic_m.sum()
    for part, names, key in (
        (fs.SUBJECT, fs.SUBJECTS, "subject"),
        (fs.ASSERTION, fs.ASSERTIONS, "assertion"),
    ):
        target = torch.tensor(
            [-100 if r[key] is None else names.index(r[key]) for r in mb], device="mps"
        )
        if (target >= 0).any():
            loss = loss + AUX_WEIGHT * ce(logits[:, part], target, ignore_index=-100)
    return loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--min-updates", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1134)
    args = ap.parse_args()
    out_dir = P0 / f"student-{round(args.fraction * 100)}"
    if (out_dir / "manifest.json").exists():
        print("exists", out_dir.name)
        return
    private_dir(out_dir)
    log = (P0 / f"{out_dir.name}.log").open("a")

    def note(*parts: object) -> None:
        line = " ".join(map(str, parts))
        print(line, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        log.flush()

    rows, stats = real_rows()
    rows = subset(rows, args.fraction)
    import torch
    import transformers

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = transformers.AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    fs.add_special_tokens(tok)
    encoder = transformers.AutoModel.from_pretrained(BASE, local_files_only=True)
    if len(tok) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tok))
    # As v1: without checkpointing the MPS allocator grew past 30 GB and swapped.
    encoder.gradient_checkpointing_enable({"use_reentrant": False})
    model = fs.build_model(torch, encoder).to("mps").train()
    spans: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        sp = spans.setdefault(r["src"], sentence_spans(r["text"]))
        r["enc"] = fs.encode_target(tok, r["text"], sp, r["index"], r["start"], r["end"])
    positives = [sum(r["remove"][h] for r in rows) for h in range(4)]
    pos_weight = [min(10.0, max(1.0, (len(rows) - p) / max(p, 1))) for p in positives]
    counts = stats | {
        "fraction": args.fraction,
        "rows": len(rows),
        "groups": len({r["group"] for r in rows}),
        "rows_by_source": {s: sum(r["source"] == s for r in rows) for s in ("sol", "v1")},
        "remove_positives": dict(zip(fs.HEADS, positives, strict=True)),
        "known_subject": sum(r["subject"] is not None for r in rows),
        "known_assertion": sum(r["assertion"] is not None for r in rows),
        "known_topic_rows": sum(r["topic"][0] is not None for r in rows),
        "synthetic_rows": 0,
    }
    note(json.dumps(counts))
    updates = max(args.min_updates, args.epochs * len(rows) // args.batch)
    head_params = list(model.heads.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )
    warm = max(1, updates // 16)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (updates - s) / (updates - warm)))
    )
    pw = torch.tensor(pos_weight, device="mps")
    stream: list[dict[str, Any]] = []
    t0 = time.time()
    losses = []
    for step in range(updates):
        while len(stream) < args.batch:
            more = rows[:]
            rng.shuffle(more)
            stream.extend(more)
        batch, stream = stream[: args.batch], stream[args.batch :]
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(0, len(batch), args.micro):
            mb = batch[k : k + args.micro]
            loss = factored_loss(torch, model, tok, mb, pw)
            (loss * len(mb) / len(batch)).backward()
            total += float(loss.item()) * len(mb) / len(batch)
        if not torch.isfinite(torch.tensor(total)):
            raise SystemExit("nonfinite loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(round(total, 5))
        if (step + 1) % 25 == 0:
            torch.mps.empty_cache()
            note(
                "update",
                step + 1,
                "/",
                updates,
                "loss",
                round(sum(losses[-25:]) / 25, 4),
                round(time.time() - t0),
                "s",
            )
    model.eval()
    fs.save(
        model,
        tok,
        out_dir,
        {
            "base": BASE.name,
            "thresholds": [0.5] * 4,
            "threshold_note": "placeholder; the evaluation selects the operating point",
            "train": counts,
            "updates": updates,
            "lr": args.lr,
            "batch": args.batch,
            "aux_weight": AUX_WEIGHT,
            "pos_weight": pos_weight,
            "seconds": round(time.time() - t0),
            "loss": losses,
            "production_ready": False,
        },
    )
    note("saved", out_dir.name, round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
