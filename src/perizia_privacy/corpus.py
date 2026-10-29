"""Build grouped local research frames. The CLI emits aggregate counts only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from . import LABELS

SEED = 1134
# Broad retrieval patterns, deliberately NOT ground-truth labels or a safety gate.
MINING_PATTERNS = {
    "salute": r"salut[e]|sanitari|malatt|malat[oaie]|disabil|handicap|invalid|infermit|ospedal|ricover|patolog|terapi|psichiat|depress|demenza|alzheimer|tumor|oncolog|diabet|autistic|autismo|tossicodip|dipendenz|non autosuff|legge\s+(?:n\.?\s*)?104|affett[oa]|assistenz|pension|certificat[oa]\s+medic|sieroposit|gravid|gestante|deambul|carrozz|badante|dialisi|chemioter|cardiop|sord|cecità|ipoacus|accompagnamento",
    "penale": r"penal|reat[oi]\b|confisc|sequestr|antimafia|prevenzione|condann|imputat|indagat|arrestat|detenut|carcer|reclus|procura|tribunale.*sorveglianza|casellario|art\.?\s*321|159\s*/\s*2011|probatori|corruzion|riciclagg|truffa|estorsion|frode|patteggiam|misura cautelare|criminal",
    "minori": r"minor[ei]|minoren|figli|bambin|ragazz|adolescen|infan|affidament|tutela|tutore|genitor|potestà|responsabilità genitoriale|nucleo familiar|scuola|scolastic|adozion|adottiv|età|anni\s+(?:[0-9]|1[0-7])\b",
    "famiglia": r"famigli|familia|coniug|coniuge|moglie|marito|figli|separazion|divorz|matrimon|convivent|conviven|celibe|nubile|vedov|stato civile|violenza|maltratt|protezione|allontanamento|assegnazion.*casa|casa.*assegnazion|mantenimento|parent[ei]|fratell|sorell|suocer|nipot|madre|padre|genitor|nucleo",
    "altre_categorie": r"religio|confession|cattolic|cristian|musulman|islam|ebraic|ebre[oi]|budd|testimon.*geova|ate[oa]|etnic|razzial|rom\b|sinti|zingar|orientamento sess|omosess|lesbic|bisess|transess|transgender|vita sess|rapporti sess|sindacal|sindacat|partito|politic|militant|iscritt.*(?:cgil|cisl|uil)|filosofic|biometric|genetic|dna\b|nazionalità|cittadinanza|provenienza",
}
MINING = {label: re.compile(pattern, re.I) for label, pattern in MINING_PATTERNS.items()}
ABBREVIATIONS = frozenset(
    [
        "art",
        "artt",
        "n",
        "nn",
        "nr",
        "sig",
        "sigg",
        "sigra",
        "ra",
        "ssa",
        "dott",
        "dr",
        "avv",
        "geom",
        "ing",
        "arch",
        "rag",
        "prof",
        "prot",
        "rif",
        "cfr",
        "es",
        "ecc",
        "lett",
        "co",
        "comma",
        "cc",
        "cp",
        "cpc",
        "cpp",
        "lgs",
        "pag",
        "pagg",
        "mq",
        "loc",
        "tel",
        "piva",
    ]
)
STOP = re.compile(r"[.;!?](?=\s|$)|\n(?=\s*\n)")
WORD = re.compile(r"([A-Za-zÀ-ÿ]+)$")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def private_json(path: Path, value: Any) -> None:
    """Replace a private checkpoint atomically, retaining the last good version."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_private_jsonl(path: Path, value: Any) -> None:
    """Append a durable local event; callers hold their experiment writer lock."""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Conservative sentence boundaries; OCR single newlines remain context."""
    spans = []
    start = 0
    for match in STOP.finditer(text):
        if match.group() == ".":
            preceding = WORD.search(text[max(0, match.start() - 24) : match.start()])
            token = preceding.group(1) if preceding else ""
            if len(token) == 1 or token.lower() in ABBREVIATIONS:
                continue
            following = match.end()
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following].islower():
                continue
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def windows(text: str, max_chars: int = 700) -> list[tuple[int, int, int, int]]:
    """Window long sentences; retain the full sentence bounds for redaction."""
    if max_chars <= 0:
        raise ValueError("Window length must be positive")
    result = []
    for sentence_start, sentence_end in sentence_spans(text):
        start = sentence_start
        while start < sentence_end:
            end = min(start + max_chars, sentence_end)
            if end < sentence_end:
                space = text.rfind(" ", start + max_chars // 2, end)
                if space > start:
                    end = space
            if text[start:end].strip():
                result.append((start, end, sentence_start, sentence_end))
            if end == sentence_end:
                break
            start = max(start + 1, end - 120)
    return result


def example_text(text: str, start: int, end: int) -> str:
    before = text[max(0, start - 200) : start].strip()
    after = text[end : end + 200].strip()
    return f"Prima: {before}\n[TARGET]{text[start:end].strip()}[/TARGET]\nDopo: {after}"


def candidate_labels(text: str) -> list[str]:
    return [label for label in LABELS if MINING[label].search(text)]


def build(source: Path, output: Path) -> dict[str, Any]:
    """Create immutable real-document partitions and local candidate pools."""
    datasketch = importlib.import_module("datasketch")

    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    if (output / "manifest.json").exists():
        raise ValueError("Corpus already exists; use a new version directory")
    paths = sorted(source.glob("*.txt"))
    parent: dict[str, str] = {}
    documents: list[dict[str, Any]] = []
    lsh = datasketch.MinHashLSH(threshold=0.85, num_perm=64)
    exact: dict[str, str] = {}
    errors: Counter[str] = Counter()

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index, path in enumerate(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors[type(exc).__name__] += 1
            continue
        content_hash = digest(normalized(text))
        if content_hash in exact:
            continue
        source_id = digest(path.name + content_hash)
        parent[source_id] = source_id
        exact[content_hash] = source_id
        # Numeric normalization groups reissued versions; body shingles also
        # group nearly identical templates instead of random-splitting them.
        tokens = re.findall(r"\w+", re.sub(r"\d+", "0", normalized(text)))
        shingles = [" ".join(tokens[i : i + 5]).encode() for i in range(0, len(tokens) - 4, 3)]
        if len(shingles) > 2500:
            shingles = random.Random(SEED).sample(shingles, 2500)
        signature = datasketch.MinHash(num_perm=64, seed=SEED)
        if shingles:
            signature.update_batch(shingles)
            for neighbor in lsh.query(signature):
                a, b = root(source_id), root(neighbor)
                parent[max(a, b)] = min(a, b)
            lsh.insert(source_id, signature)
        documents.append(
            {
                "source_doc_id": source_id,
                "content_hash": content_hash,
                "raw_hash": digest(text),
                "path": str(path),
            }
        )
        if (index + 1) % 500 == 0:
            print(json.dumps({"stage": "document_grouping", "files": index + 1}), flush=True)

    groups = Counter(root(doc["source_doc_id"]) for doc in documents)
    # Whole group assignments are stable and independent of sensitive labels.
    partitions = {}
    for group in groups:
        bucket = int(digest(f"{SEED}:{group}")[:8], 16) % 100
        partitions[group] = "train" if bucket < 60 else "validation" if bucket < 80 else "test"
    test_docs = [doc for doc in documents if partitions[root(doc["source_doc_id"])] == "test"]
    audit_ids = {
        d["source_doc_id"]
        for d in sorted(test_docs, key=lambda d: digest("audit" + d["source_doc_id"]))[:100]
    }
    counts: Counter[str] = Counter()
    mining_counts: Counter[str] = Counter()
    seen_targets: set[str] = set()
    document_rows = []
    handles = {
        split: (output / f"{split}_pool.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    for split in handles:
        os.chmod(output / f"{split}_pool.jsonl", 0o600)
    try:
        for doc in documents:
            group = root(doc["source_doc_id"])
            split = partitions[group]
            text = Path(doc["path"]).read_text(encoding="utf-8")
            if digest(text) != doc["raw_hash"]:
                raise ValueError("Source changed while constructing the corpus")
            document_rows.append(
                {**doc, "doc_id": group, "split": split, "audit": doc["source_doc_id"] in audit_ids}
            )
            counts[f"documents_{split}"] += 1
            for start, end, sentence_start, sentence_end in windows(text):
                target = text[start:end].strip()
                if len(target) < 25:
                    counts["short_windows_excluded"] += 1
                    continue
                target_hash = digest(normalized(target))
                if target_hash in seen_targets:
                    counts["duplicate_windows_excluded"] += 1
                    continue
                seen_targets.add(target_hash)
                labels = candidate_labels(target)
                identifier = digest(doc["source_doc_id"] + f":{start}:{end}")
                record = {
                    "id": identifier,
                    "doc_id": group,
                    "source_doc_id": doc["source_doc_id"],
                    "split": split,
                    "source": "real",
                    "text": example_text(text, start, end),
                    "target_hash": target_hash,
                    "start": start,
                    "end": end,
                    "sentence_start": sentence_start,
                    "sentence_end": sentence_end,
                    "mined_labels": labels,
                    "audit": doc["source_doc_id"] in audit_ids,
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[f"windows_{split}"] += 1
                for label in labels:
                    mining_counts[f"{split}_{label}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    private_json(output / "documents.json", document_rows)
    manifest = {
        "version": 1,
        "seed": SEED,
        "source_files": len(paths),
        "unique_documents": len(documents),
        "document_groups": len(groups),
        "largest_groups": sorted(groups.values(), reverse=True)[:10],
        "grouping": "normalized exact hash plus MinHash LSH 5-word shingles threshold .85, 64 permutations",
        "counts": dict(counts),
        "mining_counts_not_labels": dict(mining_counts),
        "read_errors": dict(errors),
        "audit_documents": len(audit_ids),
        "labels": list(LABELS),
    }
    private_json(output / "manifest.json", manifest)
    return manifest


def select(pool: Path, output: Path, per_category: int, negatives: int) -> dict[str, Any]:
    """Balanced broad candidates, selected without teacher/test outcomes."""
    if per_category < 0 or negatives < 0:
        raise ValueError("Selection quotas must be nonnegative")
    rows = [json.loads(line) for line in pool.open(encoding="utf-8")]
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test selection is sealed until the final evaluation protocol is locked")
    rng = random.Random(SEED)
    rng.shuffle(rows)
    selected: dict[str, Any] = {}
    for label in LABELS:
        if per_category == 0:
            continue
        hits = [row for row in rows if label in row["mined_labels"]]
        # Spread candidates across documents instead of filling a quota from one report.
        per_doc: Counter[str] = Counter()
        for row in hits:
            if per_doc[row["doc_id"]] >= 10:
                continue
            selected[row["id"]] = row
            per_doc[row["doc_id"]] += 1
            if sum(per_doc.values()) >= per_category:
                break
    nonhits = [row for row in rows if not row["mined_labels"]]
    for row in nonhits[:negatives]:
        selected[row["id"]] = row
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("w", encoding="utf-8") as handle:
        os.chmod(output, 0o600)
        for row in sorted(selected.values(), key=lambda row: row["id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "selected": len(selected),
        "mining_counts_not_labels": dict(
            Counter(label for row in selected.values() for label in row["mined_labels"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--pool", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--per-category", type=int, default=200)
    select_parser.add_argument("--negatives", type=int, default=200)
    args = parser.parse_args()
    try:
        result = (
            build(args.source, args.output)
            if args.command == "build"
            else select(args.pool, args.output, args.per_category, args.negatives)
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
