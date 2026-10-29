"""Run the two local teachers sequentially with resumable private artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import LABELS
from .annotation import (
    PROMPT_HASH,
    TEACHERS,
    _check_test_outputs,
    _recheck_test_freeze,
    _test_freeze_context,
    annotate,
)
from .corpus import private_json
from .training import target_text

ALIASES = ("privacy-qwen", "privacy-gemma")
LOAD_KEYS = {"privacy-qwen": "qwen_qwen3.6-27b", "privacy-gemma": "gemma-4-12b-it-mlx"}
ANNOTATION_KEYS = {"id", "labels", "uncertain", "model", "prompt_hash", "input_hash"}


def _validate_teachers() -> None:
    if set(TEACHERS) != set(ALIASES) or len(set(TEACHERS.values())) != 2:
        raise ValueError("Two distinct intended teachers are required")
    if "qwen" not in TEACHERS[ALIASES[0]].lower() or "gemma" not in TEACHERS[ALIASES[1]].lower():
        raise ValueError("Teacher checkpoint family mismatch")


def _loaded_models(lms: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [lms, "ps", "--json"], capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode:
        raise ValueError("Cannot inspect local model runtime")
    try:
        loaded = json.loads(result.stdout)
        if not isinstance(loaded, list) or any(not isinstance(row, dict) for row in loaded):
            raise ValueError
        owned = [row for row in loaded if row.get("identifier") in ALIASES]
        if len({row["identifier"] for row in owned}) != len(owned):
            raise ValueError
        if any(row.get("path") != TEACHERS[row["identifier"]] for row in owned):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise ValueError("Ambiguous or conflicting experiment model alias") from None
    return loaded


def _checkpoint_identity(alias: str) -> dict[str, Any]:
    path = Path.home() / ".lmstudio/models" / TEACHERS[alias]
    files = (
        [path]
        if path.is_file()
        else sorted(path.glob("*.safetensors")) + sorted(path.glob("*.json"))
    )
    if not files or not any(file.suffix in (".gguf", ".safetensors") for file in files):
        raise ValueError("Local teacher weights are missing")
    hashes = {}
    for file in files:
        with file.open("rb") as handle:
            hashes[file.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"alias": alias, "checkpoint": TEACHERS[alias], "sha256": hashes, "gpu_requested": "max"}


def load_teacher(alias: str) -> dict[str, Any]:
    _validate_teachers()
    if alias not in ALIASES:
        raise ValueError("Unknown experiment teacher alias")
    identity = _checkpoint_identity(alias)
    lms = str(Path.home() / ".lmstudio/bin/lms")
    loaded = _loaded_models(lms)
    for model in loaded:
        identifier = model.get("identifier")
        if identifier in ALIASES and identifier != alias:
            unload = subprocess.run(
                [lms, "unload", identifier], capture_output=True, timeout=60, check=False
            )
            if unload.returncode:
                raise ValueError("Cannot unload previous experiment teacher")
    if not any(
        row.get("identifier") == alias and row.get("path") == TEACHERS[alias] for row in loaded
    ):
        started = time.monotonic()
        result = subprocess.run(
            [
                lms,
                "load",
                LOAD_KEYS[alias],
                "--gpu",
                "max",
                "--context-length",
                "8192",
                "--parallel",
                "2",
                "--identifier",
                alias,
                "--yes",
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode:
            raise ValueError("Cannot load experiment teacher")
        print(
            json.dumps(
                {
                    "stage": "teacher_loaded",
                    "model": alias,
                    "seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )
    verified = _loaded_models(lms)
    if not any(
        row.get("identifier") == alias and row.get("path") == TEACHERS[alias] for row in verified
    ):
        raise ValueError("Teacher load did not establish the intended checkpoint")
    return identity


def _decode_jsonl(raw: bytes) -> list[dict[str, Any]]:
    records = []
    try:
        # JSON strings may contain literal Unicode line separators. Only split
        # record delimiters at the byte level, preserving the source string.
        lines = [line.decode("utf-8") for line in raw.splitlines()]
    except UnicodeError:
        raise ValueError("Invalid JSONL encoding") from None
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"Invalid JSONL record at line {number}") from None
        records.append(record)
    return records


def _source_rows(
    source: Path, *, allow_test: bool = False, source_bytes: bytes | None = None
) -> dict[str, dict[str, Any]]:
    """Internal reader; allow_test is used only after verifying the exact cohort."""
    rows = {}
    groups: dict[str, str] = {}
    documents: dict[str, tuple[str, str]] = {}
    for row in _decode_jsonl(source.read_bytes() if source_bytes is None else source_bytes):
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("id", "doc_id", "text")
        ):
            raise ValueError("Invalid source identifiers or text")
        if row["id"] in rows:
            raise ValueError("Duplicate source sample ID")
        if row.get("split") not in (
            ("test", "challenge") if allow_test else ("train", "validation", "calibration")
        ) or row.get("source") not in (
            "real",
            "synthetic",
        ):
            raise ValueError("Invalid source split or provenance")
        if row["split"] in ("validation", "test") and row["source"] != "real":
            raise ValueError("Evaluation examples must be real")
        if row["split"] in ("calibration", "challenge") and row["source"] != "synthetic":
            raise ValueError("Calibration and challenge examples must be explicitly synthetic")
        target_text(row["text"])
        if row["doc_id"] in groups and groups[row["doc_id"]] != row["split"]:
            raise ValueError("Source document group crosses splits")
        groups[row["doc_id"]] = row["split"]
        if "source_doc_id" in row:
            document = row["source_doc_id"]
            if not isinstance(document, str) or not document.strip():
                raise ValueError("Invalid source document ID")
            identity = (row["doc_id"], row["split"])
            if document in documents and documents[document] != identity:
                raise ValueError("Inconsistent source document assignment")
            documents[document] = identity
        rows[row["id"]] = row
    if not rows:
        raise ValueError("Source dataset is empty")
    return rows


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _annotations(
    alias: str,
    directory: Path,
    rows: dict[str, dict[str, Any]],
    source_hash: str,
    *,
    test_freeze_sha256: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    has_test = any(row.get("split") in ("test", "challenge") for row in rows.values())
    if has_test and (
        not _hash(test_freeze_sha256)
        or any(
            (row.get("split"), row.get("source"))
            not in (("test", "real"), ("challenge", "synthetic"))
            for row in rows.values()
        )
        or len({row["split"] for row in rows.values()}) != 1
    ):
        raise ValueError("Test annotations require a verified, unmixed freeze binding")
    if not has_test and test_freeze_sha256 is not None:
        raise ValueError("A test freeze cannot authorize non-test annotations")
    path = directory / f"{alias}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    weights_path = directory / f"{alias}-weights.json"
    if not path.exists() and not manifest_path.exists():
        return {}, {"status": "not_started", "weights_provenance": "not_recorded"}
    if not manifest_path.exists() or not weights_path.exists():
        raise ValueError("Annotation provenance manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    settings = manifest["settings"]
    required_settings = {
        "model",
        "prompt_hash",
        "source_hash",
        "temperature",
        "seed",
        "batch_size",
        "reasoning_effort",
        "enable_thinking",
        "runtime_identity",
    }
    if not required_settings <= settings.keys() or set(settings) - required_settings - {
        "weights_manifest_hash",
        "test_freeze_sha256",
    }:
        raise ValueError("Unexpected annotation settings schema")
    if settings.get("test_freeze_sha256") != test_freeze_sha256 or (
        "test_freeze_sha256" in settings and not _hash(settings["test_freeze_sha256"])
    ):
        raise ValueError("Annotation test freeze binding mismatch")
    if (
        settings["model"] != alias
        or settings["prompt_hash"] != PROMPT_HASH
        or settings["source_hash"] != source_hash
        or type(settings["temperature"]) not in (int, float)
        or settings["temperature"] != 0
        or type(settings["seed"]) is not int
        or settings["seed"] != 1134
        or type(settings["batch_size"]) is not int
        or settings["batch_size"] < 1
        or settings["reasoning_effort"] != "none"
        or settings["enable_thinking"] is not False
        or settings["runtime_identity"].get("path") != TEACHERS[alias]
    ):
        raise ValueError("Annotation settings or checkpoint provenance mismatch")
    weights_bytes = weights_path.read_bytes()
    weights = json.loads(weights_bytes)
    if (
        set(weights) != {"alias", "checkpoint", "sha256", "gpu_requested"}
        or weights["alias"] != alias
        or weights["checkpoint"] != TEACHERS[alias]
        or weights["gpu_requested"] != "max"
        or not isinstance(weights["sha256"], dict)
        or not weights["sha256"]
        or any(
            not isinstance(name, str) or Path(name).name != name or not _hash(value)
            for name, value in weights["sha256"].items()
        )
        or not any(name.endswith((".gguf", ".safetensors")) for name in weights["sha256"])
    ):
        raise ValueError("Invalid teacher weight provenance")
    binding = settings.get("weights_manifest_hash")
    if "weights_manifest_hash" in settings and (
        not _hash(binding) or binding != hashlib.sha256(weights_bytes).hexdigest()
    ):
        raise ValueError("Annotation weight manifest binding mismatch")
    status = manifest.get("status")
    if status not in ("running", "partial", "complete"):
        raise ValueError("Invalid annotation manifest status")
    history = manifest.get("execution_history", [])
    if not isinstance(history, list):
        raise ValueError("Invalid annotation execution history")
    for entry in history:
        try:
            if (
                not isinstance(entry, dict)
                or not set(entry) >= {"started_at", "starting_completed", "concurrency"}
                or set(entry)
                - {"started_at", "starting_completed", "concurrency", "schema_recovery"}
                or (
                    "schema_recovery" in entry
                    and entry["schema_recovery"] != "bisect_invalid_alignment_v1"
                )
                or type(entry["starting_completed"]) is not int
                or not 0 <= entry["starting_completed"] <= len(rows)
                or type(entry["concurrency"]) is not int
                or not 1 <= entry["concurrency"] <= 4
            ):
                raise ValueError
            started = datetime.fromisoformat(entry["started_at"])
            if started.utcoffset() != timedelta(0):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("Invalid annotation execution history") from None
    annotations = {}
    annotation_bytes = path.read_bytes() if path.exists() else b""
    concurrency_counts: Counter[str] = Counter()
    for record in _decode_jsonl(annotation_bytes):
        identifier = record.get("id")
        if not isinstance(identifier, str) or identifier not in rows:
            raise ValueError("Unknown annotation sample ID")
        if identifier in annotations:
            raise ValueError("Duplicate annotation sample ID")
        labels = record.get("labels")
        if (
            not record.keys() >= ANNOTATION_KEYS
            or set(record) - ANNOTATION_KEYS - {"inference_concurrency", "inference_batch_size"}
            or not isinstance(labels, list)
            or any(not isinstance(label, str) or label not in LABELS for label in labels)
            or len(set(labels)) != len(labels)
            or type(record.get("uncertain")) is not bool
            or record.get("model") != alias
            or record.get("prompt_hash") != settings["prompt_hash"]
            or record.get("input_hash")
            != hashlib.sha256(rows[identifier]["text"].encode()).hexdigest()
        ):
            raise ValueError("Invalid annotation schema or provenance")
        concurrency = record.get("inference_concurrency", 1)
        if type(concurrency) is not int or not 1 <= concurrency <= 4:
            raise ValueError("Invalid annotation scheduling provenance")
        if "inference_batch_size" in record and (
            type(record["inference_batch_size"]) is not int
            or not 1 <= record["inference_batch_size"] <= settings["batch_size"]
        ):
            raise ValueError("Invalid annotation batch-size provenance")
        if "inference_concurrency" in record and concurrency not in {
            entry["concurrency"] for entry in history
        }:
            raise ValueError("Annotation scheduling is absent from execution history")
        concurrency_counts[f"concurrency_{concurrency}"] += 1
        if "inference_concurrency" not in record:
            concurrency_counts["legacy_implicit_concurrency_1"] += 1
        annotations[identifier] = {
            **record,
            "labels": [label for label in LABELS if label in labels],
        }
    completed, total = manifest.get("completed", 0), manifest.get("total", len(rows))
    if (
        type(completed) is not int
        or type(total) is not int
        or total != len(rows)
        or not 0 <= completed <= len(annotations)
        or (status != "running" and completed != len(annotations))
        or (status == "complete" and len(annotations) != len(rows))
    ):
        raise ValueError("Annotation manifest counts disagree with records")
    return annotations, {
        "status": status,
        "recorded_completed": completed,
        "actual_completed": len(annotations),
        "settings": settings,
        "weights": weights,
        "weights_provenance": "bound_in_annotation_settings"
        if binding is not None
        else "separately_recorded",
        "inference_concurrency_counts": dict(concurrency_counts),
        "execution_history": history,
        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest()
        if path.exists()
        else None,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _publish(directory: Path, contents: dict[str, str]) -> None:
    """Stage all files privately; publish the aggregate report as commit marker."""
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for name, content in contents.items():
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, prefix=".combine-", delete=False
            ) as handle:
                path = Path(handle.name)
                staged[name] = path
                os.chmod(path, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            original = directory / name
            if original.is_symlink():
                raise ValueError("Combined artifacts must not be symbolic links")
            if original.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=directory, prefix=".combine-", delete=False
                ) as backup:
                    backups[name] = Path(backup.name)
                    os.chmod(backup.name, 0o600)
                    backup.write(original.read_bytes())
                    backup.flush()
                    os.fsync(backup.fileno())
        for name, path in staged.items():
            os.replace(path, directory / name)
            published.append(name)
    except BaseException:
        for name in reversed(published):
            if name in backups:
                os.replace(backups[name], directory / name)
            else:
                (directory / name).unlink(missing_ok=True)
        raise
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)


def combine(source: Path, directory: Path, *, test_freeze: Path | None = None) -> dict[str, Any]:
    _validate_teachers()
    source_bytes = source.read_bytes()
    context = _test_freeze_context(source, source_bytes, test_freeze)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rows = _source_rows(source, allow_test=context is not None, source_bytes=source_bytes)
    _check_test_outputs(
        source,
        [directory / name for name in ("agreement.jsonl", "review.jsonl", "agreement-report.json")],
        context,
    )
    annotations = {}
    provenance = {}
    for alias in ALIASES:
        annotations[alias], provenance[alias] = _annotations(
            alias,
            directory,
            rows,
            source_hash,
            **({"test_freeze_sha256": context["sha256"]} if context is not None else {}),
        )
    common = set(rows) & set(annotations["privacy-qwen"]) & set(annotations["privacy-gemma"])
    counts: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    outputs: dict[str, list[str]] = {name: [] for name in ("agreement", "review")}
    for identifier in sorted(rows):
        row = rows[identifier]
        missing = [alias for alias in ALIASES if identifier not in annotations[alias]]
        if missing:
            counts["review"] += 1
            counts["missing_pair"] += 1
            record = {
                **row,
                "qwen": annotations[ALIASES[0]].get(identifier),
                "gemma": annotations[ALIASES[1]].get(identifier),
                "label_source": "requires_adjudication",
                "review_reason": "missing_teacher_annotation",
                "missing_teachers": missing,
            }
            outputs["review"].append(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            row = rows[identifier]
            a, b = annotations["privacy-qwen"][identifier], annotations["privacy-gemma"][identifier]
            agrees = a["labels"] == b["labels"] and not a["uncertain"] and not b["uncertain"]
            destination = "agreement" if agrees else "review"
            counts[destination] += 1
            if agrees:
                record = {**row, "labels": a["labels"], "label_source": "dual_model_agreement"}
                counts.update("agreed_positive_" + label for label in a["labels"])
                if not a["labels"]:
                    counts["agreed_negative"] += 1
            else:
                counts["disagreement_or_uncertain"] += 1
                record = {
                    **row,
                    "qwen": a,
                    "gemma": b,
                    "label_source": "requires_adjudication",
                    "review_reason": "disagreement_or_uncertain",
                }
            outputs[destination].append(json.dumps(record, ensure_ascii=False) + "\n")
            for label in LABELS:
                per_label[label][
                    f"qwen_{int(label in a['labels'])}_gemma_{int(label in b['labels'])}"
                ] += 1
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source changed while combining annotations")
    _recheck_test_freeze(source, source_bytes, context)
    payloads = {name + ".jsonl": "".join(lines) for name, lines in outputs.items()}
    report = {
        "status": "complete"
        if len(common) == len(rows)
        and all(value["status"] == "complete" for value in provenance.values())
        else "partial",
        "examples": len(rows),
        "paired": len(common),
        "missing_pair": len(rows) - len(common),
        "missing_by_teacher": {alias: len(rows) - len(annotations[alias]) for alias in ALIASES},
        "source_hash": source_hash,
        "prompt_hash": PROMPT_HASH,
        "teachers": provenance,
        "counts": dict(counts),
        "per_label_agreement_table": {label: dict(count) for label, count in per_label.items()},
        "label_provenance": "machine labels; no human gold",
        "split": sorted({row["split"] for row in rows.values()}),
        "output_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in payloads.items()
        },
    }
    if context is not None:
        report["test_freeze_sha256"] = context["sha256"]
    payloads["agreement-report.json"] = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    _publish(directory, payloads)
    return report


def run(
    source: Path,
    directory: Path,
    batch_size: int,
    concurrency: int = 1,
    *,
    test_freeze: Path | None = None,
) -> dict[str, Any]:
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("Invalid annotation concurrency")
    source_bytes = source.read_bytes()
    context = _test_freeze_context(source, source_bytes, test_freeze)
    rows = _source_rows(source, allow_test=context is not None, source_bytes=source_bytes)
    output_paths = [
        directory / name for name in ("agreement.jsonl", "review.jsonl", "agreement-report.json")
    ]
    for alias in ALIASES:
        output_paths.extend(
            directory / f"{alias}{suffix}"
            for suffix in (".jsonl", ".lock", ".manifest.json", ".metrics.jsonl", "-weights.json")
        )
    _check_test_outputs(source, output_paths, context)
    if context is not None:
        for alias in ALIASES:
            _annotations(
                alias,
                directory,
                rows,
                hashlib.sha256(source_bytes).hexdigest(),
                test_freeze_sha256=context["sha256"],
            )
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    for alias in ALIASES:
        _recheck_test_freeze(source, source_bytes, context)
        identity = load_teacher(alias)
        weights_path = directory / f"{alias}-weights.json"
        if weights_path.exists():
            if json.loads(weights_path.read_text()) != identity:
                raise ValueError("Recorded teacher weights differ; use a new run directory")
        else:
            private_json(weights_path, identity)
        annotation_options: dict[str, Any] = {"concurrency": concurrency}
        if context is not None:
            annotation_options["test_freeze"] = test_freeze
        report = asyncio.run(
            annotate(source, directory / f"{alias}.jsonl", alias, batch_size, **annotation_options)
        )
        if report["status"] not in ("complete", "partial"):
            raise ValueError("Unexpected teacher pass status")
    return combine(source, directory, test_freeze=test_freeze)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--test-freeze", type=Path)
    args = parser.parse_args()
    try:
        report = run(
            args.source,
            args.output,
            args.batch_size,
            args.concurrency,
            test_freeze=args.test_freeze,
        )
        print(json.dumps(report), flush=True)
        if report["status"] != "complete":
            raise SystemExit(1)
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), flush=True)
        raise SystemExit(130) from None
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
