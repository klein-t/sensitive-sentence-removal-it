"""Generate explicitly fictional training candidates using a local teacher.

No archive text is supplied to generation. Generation intent is not a label:
both annotators must independently classify the resulting shuffled examples.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from . import LABELS
from .annotation import PROMPT, TEACHERS, runtime_identity, verify_local_logging
from .corpus import append_private_jsonl, digest, normalized, private_json
from .training import target_text

GENERATION_PROMPT = (
    "Scrivi esempi INTERAMENTE FITTIZI per addestrare un rilevatore di informazioni "
    "sensibili nelle perizie immobiliari italiane. Non usare nomi propri, indirizzi, "
    "recapiti, codici fiscali, numeri di procedimento o persone reali. Usa ruoli "
    "generici come occupante, proprietaria, esecutato, conduttore. Ogni esempio "
    "contiene before, target, after: contesto precedente, UNA frase centrale da "
    "classificare, contesto successivo. Il contesto descrive fatti immobiliari "
    "neutri; l'informazione richiesta deve essere nella frase centrale. Varia "
    "sintassi, lessico, ruolo, genere e ordine delle clausole. Non inserire "
    "etichette, spiegazioni, avvisi di anonimizzazione o marcatori TARGET nel testo. "
    "Non scrivere nel testo che i fatti sono fittizi: la provenienza sintetica "
    "è registrata separatamente nei metadati del dataset. "
    "Target da 35 a 500 caratteri; before e after al massimo 120 caratteri. "
    "Per positive descrivi una circostanza personale concreta del sottotipo "
    "richiesto. Per hard_negative usa il lessico del sottotipo in senso edilizio, "
    "istituzionale o astratto, senza alcuna informazione sensibile personale. "
    "Restituisci solo JSON con examples, una lista della lunghezza richiesta.\n\n"
    "Definizioni e confondenti da rispettare:\n"
    + PROMPT.split("Etichette, non mutuamente esclusive:", 1)[1].split("Restituisci", 1)[0]
)
GENERATION_HASH = digest(GENERATION_PROMPT)
_GENERATION_CUES = re.compile(
    r"\[(?:/?TARGET|omissis|redacted)\]|\bhard_negative\b|"
    r"\b(?:etichett[ae]|label|categoria|category|sottotipo|subtype|polarit[àa])\s*[:=]|"
    r"\b(?:esempi[oa]|testo|scenario|persona|dati|fatti|occupante|proprietari[oa])\s+"
    r"(?:(?:interamente|completamente)\s+)?(?:fittizi[oaie]|sintetic[oaie]|inventat[oaie])\b",
    re.IGNORECASE,
)


class GenerationTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: str
    subtype: str = Field(min_length=1, max_length=300)
    family: str = Field(min_length=1, max_length=100)
    polarity: Literal["positive", "hard_negative"]
    style: str = Field(min_length=1, max_length=200)
    count: int = Field(ge=1, le=1000)


class FictionalExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    before: str = Field(max_length=120, repr=False)
    target: str = Field(min_length=35, max_length=500, repr=False)
    after: str = Field(max_length=120, repr=False)


def parse_examples(content: str, size: int) -> list[FictionalExample]:
    try:
        body = json.loads(content)
        if not isinstance(body, dict) or set(body) != {"examples"}:
            raise ValueError
        if not isinstance(body["examples"], list) or len(body["examples"]) != size:
            raise ValueError
        examples = [FictionalExample.model_validate(row) for row in body["examples"]]
        if any(len(example.target.strip()) < 35 for example in examples):
            raise ValueError
        if any(
            _GENERATION_CUES.search(value)
            for example in examples
            for value in (example.before, example.target, example.after)
        ):
            raise ValueError
        return examples
    except (ValueError, TypeError):
        raise ValueError("Invalid fictional-generation response") from None


def _safe_usage(value: dict[str, Any]) -> dict[str, Any]:
    """Only numeric timing/token fields may reach aggregate logs."""
    result: dict[str, Any] = {}
    seconds = value.get("seconds")
    if (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and 0 <= seconds < float("inf")
    ):
        result["seconds"] = seconds
    tokens = value.get("usage")
    if isinstance(tokens, dict):
        result["usage"] = {
            key: tokens[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(tokens.get(key)) is int and tokens[key] >= 0
        }
    return result


def _replace_private(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint_bytes(path: Path, previous: dict[str, Any]) -> tuple[bytes, bytes]:
    """Return the committed prefix and any uncommitted interrupted append."""
    raw = path.read_bytes() if path.exists() else b""
    counts = previous.get("counts", {})
    if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Invalid generation checkpoint counts")
    count = sum(counts.values())
    expected = previous.get("records_sha256", hashlib.sha256(b"").hexdigest())
    lines = raw.splitlines(keepends=True)
    committed = b"".join(lines[:count])
    if len(lines) < count or hashlib.sha256(committed).hexdigest() != expected:
        raise ValueError("Generation checkpoint hash mismatch")
    if previous.get("status") == "complete" and committed != raw:
        raise ValueError("Generation checkpoint hash mismatch")
    return committed, raw[len(committed) :]


def _next_seeds(
    tasks: list[GenerationTask], previous: dict[str, Any], raw_records: bytes, events_path: Path
) -> dict[str, int]:
    starts = {task.family: 1134 + index * 10000 for index, task in enumerate(tasks)}
    if "next_seed_by_family" in previous:
        seeds = previous["next_seed_by_family"]
        if (
            not isinstance(seeds, dict)
            or set(seeds) != set(starts)
            or any(
                type(seeds[family]) is not int or seeds[family] < start
                for family, start in starts.items()
            )
        ):
            raise ValueError("Invalid generation seed checkpoint")
        return dict(seeds)
    # v1 did not bind all attempts. Avoid its recorded seeds without claiming
    # that a request interrupted before either record/log write can be recovered.
    events = events_path.read_bytes() if events_path.exists() else b""
    for line in (raw_records + b"\n" + events).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # An interrupted, uncommitted tail may end mid-record.
        if not isinstance(row, dict):
            continue
        family = row.get("generation_family", row.get("family"))
        seed = row.get("generation_seed", row.get("seed"))
        if family in starts and type(seed) is int:
            starts[family] = max(starts[family], seed + 1)
    return starts


async def request_examples(
    client: httpx.AsyncClient, model: str, task: GenerationTask, size: int, seed: int
) -> tuple[list[FictionalExample], dict[str, Any]]:
    schema = FictionalExample.model_json_schema()
    started = time.monotonic()
    response = await client.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": GENERATION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {**task.model_dump(exclude={"count"}), "count": size}, ensure_ascii=False
                    ),
                },
            ],
            "temperature": 0.75,
            "seed": seed,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 4096,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "fictional_training_candidates",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "examples": {
                                "type": "array",
                                "minItems": size,
                                "maxItems": size,
                                "items": schema,
                            }
                        },
                        "required": ["examples"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"Local generation HTTP {response.status_code}")
    body = response.json()
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete fictional generation")
    return parse_examples(choice["message"]["content"], size), {
        "seconds": round(time.monotonic() - started, 3),
        "usage": body.get("usage", {}),
    }


async def generate(
    plan_path: Path,
    output_dir: Path,
    *,
    model: str = "privacy-gemma",
    split: Literal["train", "calibration", "challenge"] = "train",
    batch_size: int = 8,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if (
        model not in TEACHERS
        or split not in ("train", "calibration", "challenge")
        or not 1 <= batch_size <= 16
        or (max_batches is not None and max_batches < 1)
    ):
        raise ValueError("Invalid generation configuration")
    os.umask(0o077)
    plan_bytes = plan_path.read_bytes()
    tasks = [GenerationTask.model_validate(row) for row in json.loads(plan_bytes)]
    if not tasks or any(task.category not in LABELS for task in tasks):
        raise ValueError("Invalid generation plan categories")
    if len({task.family for task in tasks}) != len(tasks):
        raise ValueError("Generation families must be unique")
    settings = {
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "generation_prompt_sha256": GENERATION_HASH,
        "model": model,
        "temperature": 0.75,
        "base_seed": 1134,
        "batch_size": batch_size,
        "source": "wholly_fictional_no_archive_inputs",
    }
    if split != "train":
        settings["split"] = split
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    with (output_dir / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Generation already has an active writer") from None
        manifest_path, records_path = (
            output_dir / "manifest.json",
            output_dir / "generation-records.jsonl",
        )
        previous: dict[str, Any] = {}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_bytes())
            if {
                key: value
                for key, value in previous["settings"].items()
                if key != "runtime_identity"
            } != settings:
                raise ValueError("Generation resume settings changed")
            settings = previous["settings"]
        elif records_path.exists():
            raise ValueError("Generation records lack their manifest")
        committed, uncommitted = _checkpoint_bytes(records_path, previous)
        records = [json.loads(line) for line in committed.splitlines()]
        task_by_family = {task.family: task for task in tasks}
        for row in records:
            family = row.get("generation_family")
            if family not in task_by_family:
                raise ValueError("Unknown generation checkpoint family")
            task = task_by_family[family]
            target_hash = digest(normalized(target_text(row["text"])))
            if (
                row.get("source") != "synthetic"
                or row.get("split") != split
                or row.get("target_hash") != target_hash
                or row.get("id") != digest("synthetic:" + target_hash)
                or row.get("doc_id") != "synthetic-family:" + family
                or row.get("source_doc_id") != "synthetic-example:" + target_hash
                or row.get("generation_model") != model
                or row.get("generation_prompt_sha256") != GENERATION_HASH
                or type(row.get("generation_seed")) is not int
                or row["generation_seed"] < 0
                or "labels" in row
                or "label_source" in row
                or row.get("generation_intent")
                != {"category": task.category, "subtype": task.subtype, "polarity": task.polarity}
            ):
                raise ValueError("Generation checkpoint provenance mismatch")
        seen = {row["target_hash"] for row in records}
        if len(seen) != len(records):
            raise ValueError("Duplicate generation checkpoint targets")
        counts = Counter(row["generation_family"] for row in records)
        if dict(counts) != previous.get("counts", {}):
            raise ValueError("Generation checkpoint count mismatch")
        if any(counts[task.family] > task.count for task in tasks):
            raise ValueError("Generation checkpoint exceeds planned counts")
        complete = all(counts[task.family] == task.count for task in tasks)
        if previous.get("status") == "complete":
            payload = "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in sorted(records, key=lambda row: row["id"])
            ).encode()
            candidates = output_dir / "candidates.jsonl"
            if (
                not complete
                or previous.get("generated") != len(records)
                or previous.get("requested") != sum(task.count for task in tasks)
                or previous.get("labels_verified") is not False
                or previous.get("candidates_sha256") != hashlib.sha256(payload).hexdigest()
                or not candidates.exists()
                or candidates.read_bytes() != payload
            ):
                raise ValueError("Published fictional candidates changed")
            return previous
        seeds = _next_seeds(tasks, previous, committed + uncommitted, output_dir / "events.jsonl")
        errors: Counter[str] = Counter(previous.get("errors", {}))
        if any(type(count) is not int or count < 0 for count in errors.values()):
            raise ValueError("Invalid generation error checkpoint")
        if previous.get("pending_attempt") is not None:
            errors["InterruptedAttempt"] += 1
        history = previous.get(
            "attempt_history_provenance",
            "v1_prior_attempt_history_not_fully_bound" if previous else "reserved_before_request",
        )
        legacy_count = previous.get(
            "legacy_records_not_checked_for_generation_cues",
            len(records) if previous and previous.get("response_validation_version") != 2 else 0,
        )
        if not complete:
            verify_local_logging()
            identity = runtime_identity(model)
            if "runtime_identity" in settings and settings["runtime_identity"] != identity:
                raise ValueError("Generation resume runtime changed")
            settings["runtime_identity"] = identity
        if uncommitted:
            _replace_private(records_path, committed)
        records_digest = hashlib.sha256(committed)
        pending: dict[str, Any] | None = None

        def checkpoint() -> dict[str, Any]:
            state = {
                "settings": settings,
                "checkpoint_version": 2,
                "status": "running",
                "counts": dict(counts),
                "errors": dict(errors),
                "records_sha256": records_digest.hexdigest(),
                "next_seed_by_family": seeds,
                "pending_attempt": pending,
                "attempt_history_provenance": history,
                "response_validation_version": 2,
                "legacy_records_not_checked_for_generation_cues": legacy_count,
                "recovered_uncommitted_bytes": previous.get("recovered_uncommitted_bytes", 0)
                + len(uncommitted),
            }
            private_json(manifest_path, state)
            return state

        checkpoint()
        batches = 0
        errors_this_run = 0
        async with httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=300
        ) as client:
            with records_path.open("ab") as handle:
                os.chmod(records_path, 0o600)
                for task in tasks:
                    while counts[task.family] < task.count:
                        if max_batches is not None and batches >= max_batches:
                            break
                        size = min(batch_size, task.count - counts[task.family])
                        seed = seeds[task.family]
                        seeds[task.family] += 1
                        pending = {"family": task.family, "seed": seed, "requested": size}
                        checkpoint()
                        batches += 1
                        try:
                            examples, usage = await request_examples(
                                client, model, task, size, seed
                            )
                            examples = parse_examples(
                                json.dumps({"examples": [row.model_dump() for row in examples]}),
                                size,
                            )
                            usage = _safe_usage(usage)
                        except Exception as exc:
                            errors[type(exc).__name__] += 1
                            errors_this_run += 1
                            pending = None
                            checkpoint()
                            append_private_jsonl(
                                output_dir / "events.jsonl",
                                {
                                    "at": datetime.now(UTC).isoformat(),
                                    "stage": "generation_error",
                                    "family": task.family,
                                    "seed": seed,
                                    "error_type": type(exc).__name__,
                                },
                            )
                            print(
                                json.dumps(
                                    {
                                        "stage": "fictional_generation_error",
                                        "error_type": type(exc).__name__,
                                    }
                                ),
                                flush=True,
                            )
                            if errors_this_run >= 3:
                                break
                            continue
                        added = 0
                        for example in examples:
                            target_hash = digest(normalized(example.target))
                            if target_hash in seen:
                                continue
                            record = {
                                "id": digest("synthetic:" + target_hash),
                                "doc_id": "synthetic-family:" + task.family,
                                "source_doc_id": "synthetic-example:" + target_hash,
                                "split": split,
                                "source": "synthetic",
                                "text": f"Prima: {example.before}\n[TARGET]{example.target}[/TARGET]\nDopo: {example.after}",
                                "target_hash": target_hash,
                                "generation_family": task.family,
                                "generation_intent": {
                                    "category": task.category,
                                    "subtype": task.subtype,
                                    "polarity": task.polarity,
                                },
                                "generation_model": model,
                                "generation_prompt_sha256": GENERATION_HASH,
                                "generation_seed": seed,
                            }
                            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                            handle.write(encoded)
                            records_digest.update(encoded)
                            records.append(record)
                            seen.add(target_hash)
                            counts[task.family] += 1
                            added += 1
                        handle.flush()
                        os.fsync(handle.fileno())
                        if added == 0:
                            errors["NoNovelExamples"] += 1
                            errors_this_run += 1
                        pending = None
                        checkpoint()
                        append_private_jsonl(
                            output_dir / "events.jsonl",
                            {
                                "at": datetime.now(UTC).isoformat(),
                                "stage": "generation",
                                "family": task.family,
                                "seed": seed,
                                "requested": size,
                                "added": added,
                                "generated_total": len(records),
                                **usage,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "stage": "fictional_generation",
                                    "generated": len(records),
                                    "added": added,
                                    "usage": usage,
                                }
                            ),
                            flush=True,
                        )
                        if errors_this_run >= 3:
                            break
                    if errors_this_run >= 3 or (max_batches is not None and batches >= max_batches):
                        break
        complete = all(counts[task.family] == task.count for task in tasks)
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError("Generation plan changed during inference")
        if hashlib.sha256(records_path.read_bytes()).hexdigest() != records_digest.hexdigest():
            raise ValueError("Generation checkpoint changed during inference")
        report = {
            **checkpoint(),
            "settings": settings,
            "status": "complete" if complete else "partial",
            "generated": len(records),
            "requested": sum(task.count for task in tasks),
            "counts": dict(counts),
            "errors": dict(errors),
            "labels_verified": False,
            "records_sha256": records_digest.hexdigest(),
        }
        if complete:
            candidates = output_dir / "candidates.jsonl"
            payload = "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in sorted(records, key=lambda row: row["id"])
            ).encode()
            if candidates.exists():
                if candidates.read_bytes() != payload:
                    if previous.get("candidates_sha256") is not None:
                        raise ValueError("Published fictional candidates changed")
                    _replace_private(candidates, payload)
            else:
                _replace_private(candidates, payload)
            if candidates.read_bytes() != payload:
                raise ValueError("Published fictional candidates changed during publication")
            report["candidates_sha256"] = hashlib.sha256(payload).hexdigest()
        private_json(manifest_path, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(TEACHERS), default="privacy-gemma")
    parser.add_argument("--split", choices=("train", "calibration", "challenge"), default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                asyncio.run(
                    generate(
                        args.plan,
                        args.output,
                        model=args.model,
                        split=args.split,
                        batch_size=args.batch_size,
                        max_batches=args.max_batches,
                    )
                )
            ),
            flush=True,
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), flush=True)
        raise SystemExit(130) from None
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
