"""Corpus diversity for the paper: per-split aggregates, counts only.

Splits: corpus-v1 train / validation / test, plus the test instruments drawn
from them (audit100, the final held-out instrument, the clean rare-class
frame) and the Cassazione decisions. Per split: distinct tribunals and the
top-10 share, macro-area and region shares, auction-year distribution,
selected attachment type, sentences per document, OCR-quality proxies and the
near-duplicate group sizes. Tribunal, region and auction date come from the
private per-document metadata table when present; otherwise from the
extracted appraisal metadata (tribunal and perizia date only). No text, name or
identifier is printed or written.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from perizia_privacy.corpus import sentence_spans
from scripts.clean_mining import C
from scripts.final_eval import F
from scripts.proof_score import ROOT, write_json
from scripts.sentenze import S, decisions

CORPUS = ROOT / "corpus-v1"
FATTI = Path(os.environ.get("PERIZIA_FACTS_ROOT", str(ROOT / "facts")))
MACRO = {
    "Nord-Ovest": ("Piemonte", "Valle d'Aosta", "Liguria", "Lombardia"),
    "Nord-Est": ("Trentino", "Veneto", "Friuli", "Emilia"),
    "Centro": ("Toscana", "Umbria", "Marche", "Lazio"),
    "Sud e Isole": (
        "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata", "Calabria", "Sicilia", "Sardegna",
    ),
}  # fmt: skip
WORD = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
BROKEN_HYPHEN = re.compile(r"[a-zà-ÿ]-[ \t]*\n[ \t]*[a-zà-ÿ]")
CLEAN_CHAR = re.compile(r"[\w\s.,;:!?'\"()\[\]/\\%&€$°ª+*=<>@#§|«»“”‘’–—…-]")
LEXICON_MIN_DF = 5  # a word is "in the lexicon" when at least this many documents use it
def macro_area(regione: str | None) -> str | None:
    if not regione:
        return None
    return next((m for m, names in MACRO.items() if regione.startswith(names)), None)


def db_meta() -> dict[str, dict[str, Any]]:
    """doc id -> tribunal, region, auction years; {} when the metadata file is absent.

    Reads ``$PRIVACY_DATA_ROOT/doc-metadata.json`` (a private table of per-document
    procedural metadata: tribunal, region, auction dates), exported once from the
    source catalogue of the corpus.
    """
    path = ROOT / "doc-metadata.json"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pid, row in json.loads(path.read_text()).items():
        out[pid] = {
            "tribunale": row.get("tribunale"),
            "regione": row.get("regione"),
            "years": [int(y) for y in row.get("years", [])],
        }
    return out


def fatti_meta(pid: str) -> dict[str, Any]:
    path = FATTI / f"{pid}.json"
    if not path.exists():
        return {}
    proc = (json.loads(path.read_text()).get("dati") or {}).get("procedura") or {}
    date = proc.get("data_perizia") or ""
    year = re.search(r"(19|20)\d\d", date)
    return {"tribunale": proc.get("tribunale"), "perizia_year": int(year.group()) if year else None}


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def doc_features(text: str, lexicon: set[str]) -> dict[str, Any]:
    ws = words(text)
    return {
        "sentences": len(sentence_spans(text)),
        "words": len(ws),
        "oov": sum(w not in lexicon for w in ws),
        "broken_hyphens": len(BROKEN_HYPHEN.findall(text)),
        "garbage_chars": len(text) - len(CLEAN_CHAR.findall(text)),
        "chars": len(text),
    }


def quartiles(xs: list[float]) -> dict[str, float]:
    if len(xs) < 2:
        return {"n": len(xs)}
    q = statistics.quantiles(xs, n=4)
    return {"n": len(xs), "median": q[1], "q1": q[0], "q3": q[2]}


def share(c: Counter[str], n: int) -> dict[str, str]:
    return {k: f"{v}/{n} = {v / n:.1%}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def split_summary(docs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(docs)
    tribs = Counter(d["tribunale"] for d in docs if d["tribunale"])
    top10 = sum(v for _, v in tribs.most_common(10))
    known = max(1, sum(tribs.values()))
    n_words = sum(d["words"] for d in docs)
    oov = sum(d["oov"] for d in docs)
    chars = sum(d["chars"] for d in docs)
    groups = Counter(d["doc_id"] for d in docs)
    garbled = sum(d["garbage_chars"] > 0.01 * d["chars"] for d in docs)
    return {
        "documents": n,
        "tribunal_known": sum(bool(d["tribunale"]) for d in docs),
        "tribunal_source": dict(Counter(d["tribunal_source"] for d in docs)),
        "distinct_tribunals": len(tribs),
        "top10_tribunal_share": f"{top10}/{sum(tribs.values())} = {top10 / known:.1%}",
        "macro_area": share(Counter(macro_area(d["regione"]) or "unknown" for d in docs), n),
        "distinct_regions": len({d["regione"] for d in docs if d["regione"]}),
        "auction_year": dict(sorted(Counter(str(d["year"] or "unknown") for d in docs).items())),
        "perizia_year_extracted": quartiles([d["perizia_year"] for d in docs if d["perizia_year"]]),
        "attachment_type": dict(Counter(d["tipo"] or "unknown" for d in docs)),
        "sentences_per_doc": quartiles([d["sentences"] for d in docs]),
        "ocr_proxies": {
            "non_lexicon_word_rate": f"{oov / max(1, n_words):.2%}",
            "docs_with_broken_hyphenation": f"{sum(d['broken_hyphens'] > 0 for d in docs) / n:.1%}",
            "broken_hyphens_per_10k_words": round(
                1e4 * sum(d["broken_hyphens"] for d in docs) / max(1, n_words), 2
            ),
            "garbage_char_rate": f"{sum(d['garbage_chars'] for d in docs) / max(1, chars):.3%}",
            "docs_with_garbage_over_1pct": f"{garbled / n:.1%}",
        },
        "near_duplicate_groups": len(groups),
        "group_size_distribution": dict(sorted(Counter(map(str, groups.values())).items())),
    }


def instrument_sources() -> dict[str, set[str]]:
    held = set()
    for part in ("deletion", "positive", "neighbour"):
        held |= {p["source_doc_id"] for p in json.loads((F / f"{part}-private.json").read_text())}
    clean = {p["source_doc_id"] for p in json.loads((C / "units-private.json").read_text())}
    return {"heldout_instrument": held, "clean_frame": clean}


def cassazione() -> dict[str, Any]:
    used = {p["decision"] for p in json.loads((S / "units-private.json").read_text())}
    ds = [d for d in decisions() if d["id"] in used]
    years = sorted(int(d["anno"]) for d in ds if str(d.get("anno", "")).isdigit())
    return {
        "decisions": len(ds),
        "kind": dict(Counter("civil" if d["kind"] == "snciv" else "criminal" for d in ds)),
        "provision_type": dict(Counter(d.get("tipoprov") or "unknown" for d in ds)),
        "year_range": [years[0], years[-1]] if years else None,
        "year_distribution": dict(Counter(map(str, years))),
        "sentences_per_decision": quartiles([len(sentence_spans(d["text"])) for d in ds]),
    }


def main() -> None:
    meta = db_meta()
    corpus = json.loads((CORPUS / "documents.json").read_text())
    df: Counter[str] = Counter()
    for d in corpus:
        df.update(set(words(Path(d["path"]).read_text(encoding="utf-8"))))
    lexicon = {w for w, k in df.items() if k >= LEXICON_MIN_DF}
    docs = []
    for d in corpus:
        path = Path(d["path"])
        pid = path.stem
        side = path.with_suffix(".json")
        tipo = json.loads(side.read_text()).get("tipo") if side.exists() else None
        db, fa = meta.get(pid, {}), fatti_meta(pid)
        trib = db.get("tribunale") or fa.get("tribunale")
        docs.append(
            {
                "src": d["source_doc_id"],
                "doc_id": d["doc_id"],
                "split": d["split"],
                "audit": d["audit"],
                "tipo": tipo,
                "tribunale": (trib or "").strip().lower() or None,
                "tribunal_source": "db" if db.get("tribunale") else "facts" if trib else "none",
                "regione": db.get("regione"),
                "year": min(db["years"]) if db.get("years") else None,
                "perizia_year": fa.get("perizia_year"),
                **doc_features(path.read_text(encoding="utf-8"), lexicon),
            }
        )
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by[d["split"]].append(d)
        if d["audit"]:
            by["audit100"].append(d)
    for name, srcs in instrument_sources().items():
        by[name] = [d for d in docs if d["src"] in srcs]
    report: dict[str, Any] = {
        "db_available": bool(meta),
        "lexicon": f"words in >= {LEXICON_MIN_DF} corpus documents ({len(lexicon)} types)",
        "splits": {k: split_summary(v) for k, v in by.items()},
        "cassazione": cassazione(),
    }
    write_json(ROOT / "it3-20261009" / "corpus-diversity.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
