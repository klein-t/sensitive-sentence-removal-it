"""Inference speed of it3m_c, Presidio and GLiNER-PII on the same test documents.

One process, one machine, the same 20 test-split documents for every method.
Each method is timed end to end per document, from the raw document string to a
keep/remove decision for every sentence. Sentence splitting is included for all
three and is the same function (`corpus.sentence_spans`):

* it3m_c: `LocalRedactor.redact` with its default settings (fp32, MPS, batch 8; it
  splits, windows with +-200 chars of context, scores and renders).
* Presidio: `sentence_spans`, then the `baselines_ner.py` engine (spaCy
  `it_core_news_lg`, predefined Italian registry, BatchAnalyzerEngine) on the
  sentences alone; removed if any entity scores >= 0.5. CPU (spaCy's path).
* GLiNER-PII: `sentence_spans`, then `urchade/gliner_multi_pii-v1` with the
  `baselines_ner.py` labels, 200-word chunks, non-flat multi-label decoding,
  inference batch 16, threshold 0.3. MPS.

Protocol: models loaded once; one unrecorded warm-up pass over all documents;
then 3 recorded passes. Per pass: total sentences/s, median and p90 seconds per
document. The report gives the median over passes. Only aggregates are written;
document identifiers and text never leave the private directory.

    PRIVACY_DATA_ROOT=... python -m scripts.inference_speed --out results/speed-20261029/report.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from perizia_privacy.corpus import sentence_spans

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(os.environ.get("PRIVACY_DATA_ROOT", "data/private"))
MODEL = ROOT / "student-it3m-c-20261010"
N_DOCS = 20
PASSES = 3
OURS_BATCH = 8  # LocalRedactor default batch size

# Exactly the baselines_ner.py configuration (kept local so this script needs no
# private files from that stage).
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_LABELS = [
    "health condition / disease / disability of a person",
    "criminal offence, conviction, criminal proceeding",
    "minor child",
    "family relationship, marriage, separation, divorce",
]
GLINER_THRESHOLD = 0.3
GLINER_WORDS = 200
GLINER_BATCH = 16
PRESIDIO_THRESHOLD = 0.5
BUSY_CPU = 50.0  # a foreign process above this % CPU counts as contention
BUSY_LOAD = 6.0  # 1-minute load average above this counts as contention


def test_documents() -> tuple[list[str], list[int]]:
    """20 test-split documents at the 2.5, 7.5, ..., 97.5 sentence-count quantiles."""
    docs = json.loads((ROOT / "corpus-v1/documents.json").read_text())
    texts = [Path(d["path"]).read_text(encoding="utf-8") for d in docs if d["split"] == "test"]
    ranked = sorted(((len(sentence_spans(t)), t) for t in texts), key=lambda x: x[0])
    ranked = [x for x in ranked if x[0] > 0]
    picks = [ranked[int((i + 0.5) / N_DOCS * len(ranked))] for i in range(N_DOCS)]
    split_counts = [n for n, _ in ranked]
    print(
        json.dumps(
            {"test_docs": len(ranked), "test_median_sentences": statistics.median(split_counts)}
        ),
        flush=True,
    )
    return [t for _, t in picks], [n for n, _ in picks]


def ours() -> tuple[Callable[[str], int], dict[str, str]]:
    from perizia_privacy.redactor import LocalRedactor

    red = LocalRedactor(MODEL, device="mps", batch_size=OURS_BATCH)

    def run(text: str) -> int:
        return len(red.redact(text).spans)

    return run, {
        "model": "it3m_c (mDeBERTa-v3-base, 4 sigmoid heads + unused 5th)",
        "device": "mps",
        "dtype": "float32",
        "batch_size": str(OURS_BATCH),
        "input": "target sentence window + 200 chars context each side, max 384 tokens",
    }


def presidio() -> tuple[Callable[[str], int], dict[str, str]]:
    from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "it", "model_name": "it_core_news_lg"}],
        }
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=["it"])
    registry.load_predefined_recognizers(languages=["it"], nlp_engine=nlp)
    engine = BatchAnalyzerEngine(
        AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["it"])
    )

    def run(text: str) -> int:
        sents = [text[a:b] for a, b in sentence_spans(text)]
        found = engine.analyze_iterator(sents, language="it")
        return sum(any(r.score >= PRESIDIO_THRESHOLD for r in f) for f in found)

    return run, {
        "model": "presidio-analyzer + spaCy it_core_news_lg, predefined Italian registry",
        "recognizers": ",".join(sorted({r.name for r in registry.recognizers})),
        "device": "cpu",
        "rule": "remove sentence if any entity score >= 0.5",
        "input": "sentence alone",
    }


def gliner() -> tuple[Callable[[str], int], dict[str, str]]:
    import torch
    from gliner import GLiNER

    net = GLiNER.from_pretrained(GLINER_MODEL).to("mps")
    net.eval()

    def run(text: str) -> int:
        flat = []
        for k, (a, b) in enumerate(sentence_spans(text)):
            words = text[a:b].split()
            for i in range(0, max(len(words), 1), GLINER_WORDS):
                flat.append((k, " ".join(words[i : i + GLINER_WORDS])))
        with torch.no_grad():
            preds = net.inference(
                [c for _, c in flat],
                GLINER_LABELS,
                threshold=GLINER_THRESHOLD,
                flat_ner=False,
                multi_label=True,
                batch_size=GLINER_BATCH,
            )
        return len({k for (k, _), spans in zip(flat, preds, strict=True) if spans})

    return run, {
        "model": GLINER_MODEL,
        "device": "mps",
        "threshold": str(GLINER_THRESHOLD),
        "batch_size": str(GLINER_BATCH),
        "chunk_words": str(GLINER_WORDS),
        "decoding": "flat_ner=False, multi_label=True",
        "input": "sentence alone",
    }


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(0.9 * (len(s) - 1)))]


def time_method(run: Callable[[str], int], texts: list[str], sents: list[int]) -> dict:
    for t in texts:  # warm-up pass, unrecorded
        run(t)
    passes = []
    for _ in range(PASSES):
        per_doc = []
        for t in texts:
            t0 = time.perf_counter()
            run(t)
            per_doc.append(time.perf_counter() - t0)
        total = sum(per_doc)
        passes.append(
            {
                "total_s": round(total, 3),
                "sentences_per_s": round(sum(sents) / total, 2),
                "s_per_doc_median": round(statistics.median(per_doc), 3),
                "s_per_doc_p90": round(p90(per_doc), 3),
            }
        )
        print(json.dumps(passes[-1]), flush=True)
    med = {k: statistics.median(p[k] for p in passes) for k in passes[0]}
    return {"median_of_passes": med, "passes": passes}


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def load_snapshot() -> dict:
    """Load average and foreign processes above BUSY_CPU (top, 2 s sample)."""
    out = subprocess.run(
        ["top", "-l", "2", "-s", "2", "-n", "15", "-o", "cpu", "-stats", "pid,cpu,command"],
        capture_output=True,
        text=True,
    ).stdout
    rows = out.split("PID")[-1].splitlines()[1:]
    busy = []
    for row in rows:
        parts = row.split(None, 2)
        if len(parts) == 3 and int(parts[0]) != os.getpid():
            cpu = float(parts[1])
            if cpu >= BUSY_CPU:
                busy.append(f"{parts[2].strip()[:30]} {cpu:.0f}%")
    load1 = float(sysctl("vm.loadavg").strip("{} ").split()[0])
    return {"load1": load1, "busy_processes": busy}


def wait_quiet(max_wait_s: int = 3600) -> dict:
    """Block until no contention (or max_wait_s), return the conditions at start."""
    t0 = time.perf_counter()
    while True:
        snap = load_snapshot()
        quiet = not snap["busy_processes"] and snap["load1"] < BUSY_LOAD
        if quiet or time.perf_counter() - t0 > max_wait_s:
            snap["waited_s"] = round(time.perf_counter() - t0)
            snap["quiet"] = quiet
            print(json.dumps({"load_before": snap}), flush=True)
            return snap
        print(json.dumps({"waiting": snap}), flush=True)
        time.sleep(30)


def versions() -> dict[str, str]:
    pkgs = ("torch", "transformers", "gliner", "presidio-analyzer", "spacy", "it_core_news_lg")
    return {p: md.version(p) for p in pkgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    texts, sents = test_documents()
    report = {
        "what": "end-to-end inference time per document, sentence splitting included",
        "documents": {
            "n": len(texts),
            "split": "test",
            "selection": "sentence-count quantiles 2.5%..97.5% in 5% steps",
            "sentences_total": sum(sents),
            "sentences_median": statistics.median(sents),
            "sentences_min": min(sents),
            "sentences_max": max(sents),
        },
        "protocol": (
            f"models preloaded, 1 warm-up pass, {PASSES} recorded passes, median; "
            f"each method starts only when no foreign process >= {BUSY_CPU:.0f}% CPU "
            f"and load1 < {BUSY_LOAD}"
        ),
        "hardware": f"{sysctl('machdep.cpu.brand_string')}, {int(sysctl('hw.memsize')) >> 30} GB",
        "os": platform.platform(),
        "versions": versions(),
        "methods": {},
    }
    for name, load in (("it3m_c", ours), ("presidio_any", presidio), ("gliner_pii@0.3", gliner)):
        print(json.dumps({"method": name}), flush=True)
        before = wait_quiet()
        t0 = time.perf_counter()
        run, config = load()
        load_s = round(time.perf_counter() - t0, 1)
        timing = time_method(run, texts, sents)
        after = load_snapshot()
        report["methods"][name] = {
            "config": config,
            "load_s": load_s,
            "load_conditions": {"before": before, "after": after},
            **timing,
        }
        del run
        gc.collect()
    base = report["methods"]["it3m_c"]["median_of_passes"]["sentences_per_s"]
    for m in report["methods"].values():
        m["slowdown_vs_it3m_c"] = round(base / m["median_of_passes"]["sentences_per_s"], 2)
    report["gemma_12b_reference"] = {
        "source": "results/llm-baseline-20261009/report-*.json throughput blocks (not re-measured)",
        "model": "Gemma 4 12B instruct, MLX 4-bit, LM Studio, concurrency 1",
        "input": "frozen Phase 0 prompt + one unit (target + up to 2 sentences context each side)",
        "units_per_second_by_instrument": {
            "final_positive": 0.137,
            "final_deletion": 0.143,
            "clean": 0.244,
            "counterfactuals": 0.086,
            "sentenze": 0.133,
            "cascade": 0.304,
        },
        "caveat": "measured on the M4 Max while it3 training shared the GPU",
    }
    ups = report["gemma_12b_reference"]["units_per_second_by_instrument"].values()
    report["gemma_12b_reference"]["slowdown_vs_it3m_c_range"] = [
        round(base / max(ups)),
        round(base / min(ups)),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({m: v["median_of_passes"] for m, v in report["methods"].items()}, indent=1))


if __name__ == "__main__":
    main()
