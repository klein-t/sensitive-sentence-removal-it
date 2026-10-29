"""One-day proof, step 1-2: mining frame and local v1 scoring.

Builds the train-side appraisal frame, applies the existing lexical miner
labels as a prefilter, and scores the hits with v1 on MPS. Scores are appended
per batch so an interrupted run resumes. Writes ids, offsets and floats only;
never prints document text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("PRIVACY_DATA_ROOT", "data/private"))
OUT = ROOT / "proof-20261001"
V1 = ROOT / "student-v1-amended-v2"
TESTO = Path(os.environ.get("PERIZIA_TEXT_ROOT", str(ROOT / "texts")))
HEADS = ("salute", "penale", "minori", "famiglia")
LABELS = (*HEADS, "altre_categorie")
EDGE_FILES = ("famiglia_minori", "format", "penale", "rulings", "salute")
WORD = re.compile(r"\w+")


def private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def opaque(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def grams(text: str, n: int = 12) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def edge_rows() -> list[dict[str, Any]]:
    rows = []
    for name in EDGE_FILES:
        with (ROOT / "edge-cases-v1" / f"{name}.jsonl").open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


# --- model helpers shared by the packager, trainer and evaluator -------------


def load_model(path: Path, dtype: str = "float32") -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    model = model.to("mps", dtype=getattr(torch, dtype)).eval()
    return torch, tok, model


def encode(tok: Any, text: str, max_length: int = 384) -> list[int]:
    """Same truncation as training._encode: target kept, context split evenly."""
    s = text.index("[TARGET]")
    e = text.index("[/TARGET]") + len("[/TARGET]")
    target = tok.encode(text[s:e], add_special_tokens=False)
    room = max_length - 2 - len(target)
    if room < 0:
        target, room = target[: max_length - 2], 0
    left = tok.encode(text[:s], add_special_tokens=False)
    right = tok.encode(text[e:], add_special_tokens=False)
    ls = min(len(left), room // 2)
    rs = min(len(right), room - ls)
    ls = min(len(left), room - rs)
    return [tok.cls_token_id, *(left[-ls:] if ls else []), *target, *right[:rs], tok.sep_token_id]


def score_ids(torch: Any, model: Any, pad: int, rows: list[list[int]], batch: int = 32):
    """Yield (indices, sigmoid scores) in length-sorted batches."""
    import numpy as np

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    with torch.inference_mode():
        for k in range(0, len(order), batch):
            idx = order[k : k + batch]
            width = max(len(rows[i]) for i in idx)
            x = np.full((len(idx), width), pad, dtype=np.int64)
            a = np.zeros((len(idx), width), dtype=np.int64)
            for j, i in enumerate(idx):
                x[j, : len(rows[i])] = rows[i]
                a[j, : len(rows[i])] = 1
            logits = model(
                input_ids=torch.from_numpy(x).to("mps"),
                attention_mask=torch.from_numpy(a).to("mps"),
            ).logits
            yield idx, torch.sigmoid(logits.float()).cpu().numpy()


def score_texts(torch: Any, tok: Any, model: Any, texts: list[str], batch: int = 32):
    import numpy as np

    encoded = [encode(tok, t) for t in texts]
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for idx, s in score_ids(torch, model, tok.pad_token_id, encoded, batch):
        out[idx] = s
    return out


def v1_thresholds() -> list[float]:
    return json.loads((ROOT / "final-evaluation-audit100-v1/manifest.json").read_text())[
        "thresholds"
    ][:4]


# --- frame -------------------------------------------------------------------


def build_frame() -> dict[str, Any]:
    documents = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    excluded_groups = {d["doc_id"] for d in documents if d["split"] != "train" or d["audit"]}
    for path in ("training-inputs-v2/calibration.jsonl",):  # quarantined file
        with (ROOT / path).open() as stream:
            excluded_groups |= {json.loads(line)["doc_id"] for line in stream}
    appraisal = {}
    for d in documents:
        if d["split"] != "train" or d["doc_id"] in excluded_groups:
            continue
        meta = json.loads((TESTO / (Path(d["path"]).stem + ".json")).read_text())
        if meta.get("tipo") in ("PERIZ", "STIMA"):
            appraisal[d["source_doc_id"]] = d["doc_id"]
    return {"appraisal": appraisal, "excluded_groups": len(excluded_groups)}


def collect(limit_minori: int, random_none: int, seed: int = 1134) -> dict[str, Any]:
    frame = build_frame()
    appraisal = frame["appraisal"]
    suite = set().union(*(grams(r["paragraph"]) for r in edge_rows()))
    gram_docs: dict[str, set[str]] = defaultdict(set)
    rows: dict[str, dict[str, Any]] = {}
    by_head: dict[str, list[str]] = {h: [] for h in HEADS}
    none: list[str] = []
    rng = random.Random(seed)
    seen_none = 0
    with (ROOT / "corpus-v1/train_pool.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["source_doc_id"] not in appraisal:
                continue
            target = r["text"][r["text"].index("[TARGET]") + 8 : r["text"].index("[/TARGET]")]
            for g in grams(target) & suite:
                if len(gram_docs[g]) < 3:
                    gram_docs[g].add(r["doc_id"])
            mined = [h for h in HEADS if h in r["mined_labels"]]
            keep = {k: r[k] for k in ("id", "doc_id", "source_doc_id", "start", "end")}
            keep |= {"ss": r["sentence_start"], "se": r["sentence_end"], "mined": mined}
            if mined:
                rows[r["id"]] = keep | {"text": r["text"]}
                for h in mined:
                    by_head[h].append(r["id"])
            else:
                seen_none += 1
                if len(none) < random_none:
                    none.append(r["id"])
                    rows[r["id"]] = keep | {"text": r["text"]}
                else:
                    j = rng.randrange(seen_none)
                    if j < random_none:
                        rows.pop(none[j], None)
                        none[j] = r["id"]
                        rows[r["id"]] = keep | {"text": r["text"]}
    edge_sources = {d for docs in gram_docs.values() if len(docs) < 3 for d in docs}
    selected: set[str] = set(none)
    for h in HEADS:
        ids = by_head[h]
        if len(ids) > limit_minori:
            ids = rng.sample(ids, limit_minori)
        selected |= set(ids)
    selected = {i for i in selected if rows[i]["doc_id"] not in edge_sources}
    return {
        "rows": [rows[i] for i in sorted(selected)],
        "stats": {
            "frame_source_docs": len(appraisal),
            "frame_groups": len(set(appraisal.values())),
            "edge_source_groups_excluded": len(edge_sources),
            "lexical_hits_per_head": {h: len(v) for h, v in by_head.items()},
            "random_non_lexical": len(none),
            "selected_windows": len(selected),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-head-cap", type=int, default=30000)
    ap.add_argument("--random-none", type=int, default=15000)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=0, help="benchmark only")
    args = ap.parse_args()
    private_dir(OUT)
    t0 = time.time()
    data = collect(args.per_head_cap, args.random_none)
    rows = data["rows"][: args.limit] if args.limit else data["rows"]
    print("frame", json.dumps(data["stats"]), round(time.time() - t0), "s", flush=True)
    if not args.limit:
        write_json(OUT / "frame-stats.json", data["stats"])
        meta = [{k: v for k, v in r.items() if k != "text"} for r in rows]
        write_json(OUT / "candidates.json", meta)
    path = OUT / ("bench.jsonl" if args.limit else "scores.jsonl")
    done = set()
    if path.exists() and not args.limit:
        with path.open() as stream:
            done = {json.loads(line)["id"] for line in stream}
    todo = [r for r in rows if r["id"] not in done]
    torch, tok, model = load_model(V1, args.dtype)
    encoded = [encode(tok, r["text"]) for r in todo]
    t1 = time.time()
    n = 0
    with path.open("a" if not args.limit else "w") as stream:
        os.chmod(path, 0o600)
        for idx, s in score_ids(torch, model, tok.pad_token_id, encoded):
            for i, row in zip(idx, s, strict=True):
                stream.write(
                    json.dumps({"id": todo[i]["id"], "s": [round(float(v), 5) for v in row]}) + "\n"
                )
            stream.flush()
            n += len(idx)
            if n % 3200 < len(idx):
                print(
                    "scored", n, "/", len(todo), round(n / (time.time() - t1), 1), "w/s", flush=True
                )
    print("done", n, round(time.time() - t1), "s", flush=True)


if __name__ == "__main__":
    main()
