"""Paper baselines: Presidio and zero-shot GLiNER on A0's frozen instruments.

Three steps, all local, no hosted teacher:

* ``collect`` (project env): every sentence the instruments score (final IPW
  deletion sample, held-out positives and their neighbours, the clean frame
  units, the counterfactual pairs, the Cassazione units, audit100) is written,
  deduplicated, to a private file outside the repository.
* ``score --model presidio|gliner_multi|gliner_pii`` (isolated venv with
  presidio-analyzer + it_core_news_lg, or gliner): one score vector per
  sentence, appended to a private JSONL so an interrupted run resumes. The
  baselines see the sentence alone, without document context.
* ``report`` (project env): the instruments' own metric functions, applied to
  each baseline at its operating points, next to A0's frozen report numbers.

Decision rules. Presidio: a sentence is removed if any entity scores >= 0.5
(variant: person entities only). GLiNER: removed for class c if any span of c
scores >= t, for t in {0.3, 0.5, 0.7}; removed if removed for any class.
Prints aggregates only; sentence text stays in the private directory.

The score step imports nothing from the package, so it runs in a venv that has
only the baseline libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("PRIVACY_DATA_ROOT", "data/private"))
B = ROOT / "baselines-ner-20261009"
HEADS = ("salute", "penale", "minori", "famiglia")
GLINER_LABELS = {
    "salute": "health condition / disease / disability of a person",
    "penale": "criminal offence, conviction, criminal proceeding",
    "minori": "minor child",
    "famiglia": "family relationship, marriage, separation, divorce",
}
GLINER_MODELS = {
    "gliner_multi": "urchade/gliner_multi-v2.1",
    "gliner_pii": "urchade/gliner_multi_pii-v1",
}
GLINER_THRESHOLDS = (0.3, 0.5, 0.7)
GLINER_WORDS = 200  # chunk long sentences well under the 384-word window
PRESIDIO_THRESHOLD = 0.5
PERSON = {"PERSON"}
Key = tuple[str, ...]


def sid(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def chunks(text: str, size: int = GLINER_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def variants() -> dict[str, tuple[str, int, float]]:
    """Variant name -> (score file, score column, threshold applied to every head)."""
    out = {
        "presidio_any": ("presidio", 0, PRESIDIO_THRESHOLD),
        "presidio_person": ("presidio", 1, PRESIDIO_THRESHOLD),
    }
    for m in GLINER_MODELS:
        for t in GLINER_THRESHOLDS:
            out[f"{m}@{t}"] = (m, -1, t)
    return out


def head_vector(row: list[float], column: int) -> list[float]:
    """Per-head scores: Presidio has no class, so its one score fills all four."""
    return list(row) if column < 0 else [row[column]] * len(HEADS)


# --- collect (project env) ---------------------------------------------------


def cf_pairs() -> list[dict[str, Any]]:
    """The counterfactual pairs exactly as their score step keeps them."""
    from scripts.clean_mining import frame_docs
    from scripts.counterfactuals import CF, flipped, texts_of
    from scripts.proof_score import opaque

    sources = json.loads((CF / "edit-sources-private.json").read_text())
    edits = json.loads((CF / "edits-private.json").read_text())
    labels = json.loads((CF / "edited-labels.json").read_text())
    texts = texts_of(frame_docs())
    pairs = []
    for uid, s in sources.items():
        lab = labels.get(opaque("cf-blind", uid))
        if lab is None or not flipped(s["kind"], s["label"], lab):
            continue
        text = texts[s["source_doc_id"]]
        pairs.append(
            {
                "uid": uid,
                "kind": s["kind"],
                "original": text[s["ss"] : s["se"]],
                "edited": edits[uid]["edited_target"],
            }
        )
    return pairs


def collect() -> None:
    from scripts.final_eval import F
    from scripts.phase0_eval import audit_labels, doc_text
    from scripts.proof_score import private_dir, write_json
    from scripts.sentenze import S, decisions

    private_dir(B)
    text_of: dict[str, str] = {}
    keys: dict[str, list[list[str | int]]] = defaultdict(list)

    def add(instrument: str, key: Key | tuple[str | int, ...], text: str) -> None:
        i = sid(text)
        text_of[i] = text
        keys[instrument].append([*key, i])

    for part in ("deletion", "positive", "neighbour"):
        for p in json.loads((F / f"{part}-private.json").read_text()):
            src, a, b = p["source_doc_id"], p["ss"], p["se"]
            add("final", (src, a, b), doc_text(src)[a:b])
    for p in json.loads((ROOT / "clean-20261008/units-private.json").read_text()):
        src, a, b = p["source_doc_id"], p["ss"], p["se"]
        add("clean", (src, a, b), doc_text(src)[a:b])
    for p in cf_pairs():
        add("cf", (p["uid"], "original"), p["original"])
        add("cf", (p["uid"], "edited"), p["edited"])
    texts = {d["id"]: d["text"] for d in decisions()}
    for p in json.loads((S / "units-private.json").read_text()):
        d, a, b = p["decision"], p["ss"], p["se"]
        add("sentenze", (d, a, b), texts[d][a:b])
    for doc, rows in audit_labels().items():
        text = doc_text(doc)
        for r in rows:
            add("audit100", (doc, r["start"], r["end"]), text[r["start"] : r["end"]])
    write_json(B / "sentences-private.json", text_of)
    write_json(B / "keys-private.json", keys)
    print(json.dumps({"sentences": len(text_of), **{k: len(v) for k, v in keys.items()}}))


# --- score (isolated venv) ---------------------------------------------------


def pending(model: str) -> tuple[dict[str, str], Path]:
    text_of = json.loads((B / "sentences-private.json").read_text())
    path = B / f"scores-{model}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
    keys = json.loads((B / "keys-private.json").read_text())
    small = {k[-1] for name, rows in keys.items() if name != "audit100" for k in rows}
    # The small instruments first, then audit100 (the bulk), each shortest first.
    order = sorted(text_of.items(), key=lambda x: (x[0] not in small, len(x[1])))
    todo = {i: t for i, t in order if i not in done}
    print(json.dumps({"model": model, "done": len(done), "todo": len(todo)}), flush=True)
    return todo, path


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)


def score_presidio(todo: dict[str, str], path: Path, batch: int = 256) -> None:
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
    recognizers = sorted({r.name for r in registry.recognizers})
    print(json.dumps({"recognizers": recognizers}), flush=True)
    ids = list(todo)
    for start in range(0, len(ids), batch):
        part = ids[start : start + batch]
        results = engine.analyze_iterator([todo[i] for i in part], language="it")
        rows = []
        for i, found in zip(part, results, strict=True):
            types: dict[str, float] = {}
            for r in found:
                types[r.entity_type] = max(types.get(r.entity_type, 0.0), float(r.score))
            person = max((v for k, v in types.items() if k in PERSON), default=0.0)
            rows.append({"id": i, "s": [max(types.values(), default=0.0), person], "t": types})
        append(path, rows)
        print(json.dumps({"scored": start + len(part), "of": len(ids)}), flush=True)


def score_gliner(model: str, todo: dict[str, str], path: Path, batch: int = 64) -> None:
    import torch
    from gliner import GLiNER

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = GLiNER.from_pretrained(GLINER_MODELS[model]).to(device)
    net.eval()
    labels = [GLINER_LABELS[h] for h in HEADS]
    ids = list(todo)
    for start in range(0, len(ids), batch):
        part = ids[start : start + batch]
        flat = [(i, c) for i in part for c in chunks(todo[i])]
        with torch.no_grad():
            preds = net.inference(
                [c for _, c in flat],
                labels,
                threshold=min(GLINER_THRESHOLDS),
                flat_ner=False,
                multi_label=True,
                batch_size=16,
            )
        best = {i: [0.0] * len(HEADS) for i in part}
        for (i, _), spans in zip(flat, preds, strict=True):
            for span in spans:
                h = labels.index(span["label"])
                best[i][h] = max(best[i][h], float(span["score"]))
        append(path, [{"id": i, "s": best[i]} for i in part])
        print(json.dumps({"scored": start + len(part), "of": len(ids)}), flush=True)


# --- report (project env) ----------------------------------------------------


def load_scores(model: str) -> dict[str, list[float]]:
    out = {}
    for line in (B / f"scores-{model}.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            out[r["id"]] = r["s"]
    return out


def keyed(instrument: str, scores: dict[str, list[float]], column: int) -> dict[Key, list[float]]:
    keys = json.loads((B / "keys-private.json").read_text())[instrument]
    return {tuple(k[:-1]): head_vector(scores[k[-1]], column) for k in keys}


def cf_metrics(sc: dict[Key, list[float]], thr: list[float]) -> dict[str, Any]:
    from scripts.clean_mining import removed, wilson
    from scripts.counterfactuals import EDITS

    per: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for p in cf_pairs():
        o = removed(sc[(p["uid"], "original")], thr)
        e = removed(sc[(p["uid"], "edited")], thr)
        if EDITS[p["kind"]][0] == "famiglia":
            per[p["kind"]]["target_removed"].append(e)
            per[p["kind"]]["source_removed"].append(o)
        else:
            per[p["kind"]]["target_removed"].append(o)
            per[p["kind"]]["twin_kept"].append(not e)
            per[p["kind"]]["pair_correct"].append(o and not e)
    return {k: {m: wilson(sum(v), len(v)) for m, v in d.items()} for k, d in per.items()}


def a0_reference() -> dict[str, Any]:
    """A0 at its production thresholds, from the frozen reports."""
    final = json.loads((ROOT / "final-20261003/eval-final.json").read_text())["A0_production"]
    cf = json.loads((ROOT / "clean-20261008/counterfactuals/report.json").read_text())["a0"]
    rename = {
        "target_removed (edited, new class)": "target_removed",
        "source_removed (original famiglia)": "source_removed",
        "target_removed (original positive)": "target_removed",
        "twin_kept (edited harmless)": "twin_kept",
        "pair_correct (removed and kept)": "pair_correct",
    }
    return {
        "final": final["instrument"],
        "audit100": final["audit100"],
        "clean": json.loads((ROOT / "clean-20261008/report.json").read_text()),
        "cf": {k: {rename[m]: v for m, v in d.items()} for k, d in cf.items()},
        "sentenze": json.loads((ROOT / "sentenze-20261008/report.json").read_text()),
    }


def strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not str(k).startswith("_")}
    return value


def report() -> None:
    from scripts.clean_mining import summarise as clean_summary
    from scripts.final_eval import audit_metrics, instrument_metrics
    from scripts.proof_score import write_json
    from scripts.sentenze import S
    from scripts.sentenze import summarise as sentenze_summary

    clean_units = json.loads((ROOT / "clean-20261008/units-private.json").read_text())
    clean_labels = json.loads((ROOT / "clean-20261008/labels.json").read_text())
    sent_units = json.loads((S / "units-private.json").read_text())
    sent_labels = json.loads((S / "labels.json").read_text())
    cache: dict[str, dict[str, list[float]]] = {}
    wanted = set(json.loads((B / "sentences-private.json").read_text()))
    out: dict[str, Any] = {"A0_production": a0_reference()}
    for name, (model, column, t) in variants().items():
        if not (B / f"scores-{model}.jsonl").exists():
            continue
        scores = cache.setdefault(model, load_scores(model))
        if not wanted <= scores.keys():
            out[name] = {"incomplete": f"{len(wanted & scores.keys())}/{len(wanted)}"}
            continue
        thr = [t] * len(HEADS)
        res: dict[str, Any] = {"threshold": t}
        res["final"] = instrument_metrics(keyed("final", scores, column), thr)
        res["clean"] = clean_summary(clean_units, clean_labels, keyed("clean", scores, column), thr)
        res["cf"] = cf_metrics(keyed("cf", scores, column), thr)
        res["sentenze"] = sentenze_summary(
            sent_units, sent_labels, keyed("sentenze", scores, column), thr
        )
        res["audit100"] = audit_metrics(keyed("audit100", scores, column), thr)
        out[name] = strip(res)
    write_json(B / "report.json", out)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "score", "report"))
    ap.add_argument("--model", choices=("presidio", *GLINER_MODELS))
    args = ap.parse_args()
    if args.step == "collect":
        collect()
    elif args.step == "report":
        report()
    elif args.model == "presidio":
        score_presidio(*pending("presidio"))
    elif args.model:
        score_gliner(args.model, *pending(args.model))
    else:
        sys.exit("score needs --model")


if __name__ == "__main__":
    main()
