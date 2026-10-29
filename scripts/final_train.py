"""Final step, Run A: factored heads and encoding on it1's encoder.

The single remaining run. it1's mix is kept (half v1 replay including the
synthetic rows the rare heads need, half new rows including the edge-suite
training half), every row re-encoded with the factored input; the new half
also carries Phase 0's real Sol rows and the probe units with their new-schema
factors. Factor and topic losses are masked wherever a row does not know them.
Prints counts and losses only.
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
from scripts.phase0_train import factored_loss, real_rows
from scripts.proof_score import OUT, ROOT, V1, edge_rows, private_dir
from scripts.proof_train import edge_family_train, target_sentences

F = ROOT / "final-20261003"
P0 = ROOT / "phase0-20261003"
IT1 = OUT / "model"
MASKED = {"topic": [None] * 4, "subject": None, "assertion": None}


def heads_of(labels: list[str]) -> list[float]:
    return [float(h in labels) for h in fs.HEADS]


def replay_rows(tok: Any, text_of: Any) -> list[dict[str, Any]]:
    """v1's training rows: real ones re-cut from their document, synthetic from text."""
    rows = []
    path = ROOT / "development-preparation-v3/training-inputs-v3/train.jsonl"
    spans: dict[str, list[tuple[int, int]]] = {}
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["source"] == "real":
                text = text_of(r["source_doc_id"])
                sp = spans.setdefault(r["source_doc_id"], sentence_spans(text))
                index = sp.index((r["sentence_start"], r["sentence_end"]))
                enc = fs.encode_target(tok, text, sp, index, r["start"], r["end"])
            else:
                t = r["text"]
                before = t[: t.index("\n[TARGET]")].removeprefix("Prima:")
                target = t[t.index("[TARGET]") + 8 : t.index("[/TARGET]")]
                after = t[t.index("[/TARGET]") + 9 :].removeprefix("\nDopo:")
                enc = fs.encode_pieces(tok, before, target, after)
            rows.append({"enc": enc, "remove": heads_of(r["labels"]), **MASKED})
    return rows


def edge_train_rows(tok: Any) -> list[dict[str, Any]]:
    rows = []
    for row in edge_rows():
        if not edge_family_train(row):
            continue
        text = row["paragraph"]
        start = text.index(row["target"])
        targets = target_sentences(text, start, start + len(row["target"]))
        spans = sentence_spans(text)
        for a, b, s0, s1 in windows(text):
            labels = row["expected_labels"] if (s0, s1) in targets else []
            enc = fs.encode_target(tok, text, spans, spans.index((s0, s1)), a, b)
            rows.append({"enc": enc, "remove": heads_of(labels), **MASKED})
    return rows


def probe_rows(tok: Any, text_of: Any) -> list[dict[str, Any]]:
    """Phase 0 probe units where Sol A and B agree on removal; factors where they agree."""
    a = json.loads((P0 / "labels-A.json").read_text())
    b = json.loads((P0 / "labels-B.json").read_text())
    rows = []
    for p in json.loads((P0 / "probe-private.json").read_text()):
        x, y = a.get(p["unit_id"]), b.get(p["unit_id"])
        if (
            x is None
            or y is None
            or x["uncertain"]
            or y["uncertain"]
            or sorted(x["remove_heads"]) != sorted(y["remove_heads"])
        ):
            continue
        same_topics = sorted(x["topics"]) == sorted(y["topics"])
        factors = {
            "topic": [float(h in x["topics"]) for h in fs.HEADS] if same_topics else [None] * 4,
            "subject": x["subject"] if x["subject"] == y["subject"] else None,
            "assertion": x["assertion"] if x["assertion"] == y["assertion"] else None,
        }
        text = text_of(p["source_doc_id"])
        spans = sentence_spans(text)
        index = spans.index((p["ss"], p["se"]))
        for s, e, s0, s1 in windows(text):
            if (s0, s1) == (p["ss"], p["se"]):
                enc = fs.encode_target(tok, text, spans, index, s, e)
                rows.append({"enc": enc, "remove": heads_of(x["remove_heads"]), **factors})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1134)
    args = ap.parse_args()
    out_dir = F / "run-a"
    if (out_dir / "manifest.json").exists():
        print("exists")
        return
    private_dir(out_dir)
    log = (F / "run-a.log").open("a")

    def note(*parts: object) -> None:
        line = " ".join(map(str, parts))
        print(line, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        log.flush()

    import torch
    import transformers

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = transformers.AutoTokenizer.from_pretrained(IT1, local_files_only=True)
    fs.add_special_tokens(tok)
    encoder = transformers.AutoModel.from_pretrained(IT1, local_files_only=True)
    if len(tok) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tok))
    encoder.gradient_checkpointing_enable({"use_reentrant": False})
    model = fs.build_model(torch, encoder).to("mps").train()

    paths = {
        d["source_doc_id"]: d["path"]
        for d in json.loads((ROOT / "corpus-v1/documents.json").read_text())
    }
    texts: dict[str, str] = {}

    def text_of(src: str) -> str:
        if src not in texts:
            texts[src] = Path(paths[src]).read_text(encoding="utf-8")
        return texts[src]

    replay = replay_rows(tok, text_of)
    sol, _ = real_rows()
    sol_rows = []
    sol_spans: dict[str, list[tuple[int, int]]] = {}
    for r in sol:
        if r["source"] != "sol":
            continue
        sp = sol_spans.setdefault(r["src"], sentence_spans(r["text"]))
        enc = fs.encode_target(tok, r["text"], sp, r["index"], r["start"], r["end"])
        sol_rows.append(
            {k: r[k] for k in ("remove", "topic", "subject", "assertion")} | {"enc": enc}
        )
    probe = probe_rows(tok, text_of)
    edge = edge_train_rows(tok)
    new = sol_rows + probe + edge
    counts = {
        "replay_rows": len(replay),
        "sol_rows": len(sol_rows),
        "probe_rows": len(probe),
        "edge_rows": len(edge),
        "new_positive_rows": sum(any(r["remove"]) for r in new),
        "known_subject": sum(r["subject"] is not None for r in new),
        "known_assertion": sum(r["assertion"] is not None for r in new),
    }
    note(json.dumps(counts))
    pos = json.loads((V1 / "manifest.json").read_text())["pos_weight"]
    pw = torch.tensor([pos[h] for h in fs.HEADS], device="mps")
    opt = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": args.lr},
            {"params": list(model.heads.parameters()), "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )
    warm = max(1, args.updates // 16)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (args.updates - s) / (args.updates - warm)))
    )
    streams: dict[str, list[dict[str, Any]]] = {"replay": [], "new": []}

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
    losses = []
    for step in range(args.updates):
        batch = draw("replay", replay, half) + draw("new", new, args.batch - half)
        rng.shuffle(batch)
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
            recent = round(sum(losses[-25:]) / 25, 4)
            note(
                "update", step + 1, "/", args.updates, "loss", recent, round(time.time() - t0), "s"
            )
    model.eval()
    fs.save(
        model,
        tok,
        out_dir,
        {
            "init": "it1 encoder (proof-20261001/model)",
            "thresholds": [0.5] * 4,
            "threshold_note": "placeholder; the final evaluation selects the operating point",
            "train": counts,
            "mix": "half v1 replay (incl. synthetic), half new (Sol, probe, edge train half)",
            "updates": args.updates,
            "lr": args.lr,
            "batch": args.batch,
            "seconds": round(time.time() - t0),
            "loss": losses,
            "production_ready": False,
        },
    )
    note("saved run-a", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
