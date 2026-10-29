"""Local supervised training for five-category perizia sentence redaction.

Training accepts separate train/real-validation files and optional fictional
calibration. The sealed test is an input to prediction after freezing selection.
Optional ML dependencies are imported only by training and prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LABELS = ("salute", "penale", "minori", "famiglia", "altre_categorie")


class TrainingConfig(BaseModel):
    """Frozen, recorded hyperparameters; full precision on Apple Silicon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation: int = Field(default=8, ge=1)
    max_length: int = Field(default=384, ge=16)
    epochs: int = Field(default=5, ge=1, le=5)
    max_grad_norm: float = Field(default=1.0, gt=0)
    max_pos_weight: float = Field(default=10.0, ge=1)
    seed: int = 42
    gradient_checkpointing: bool = True
    require_mps: bool = False
    local_files_only: bool = True
    requested_recall: float = Field(default=0.98, gt=0, le=1)


class TrainingRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    doc_id: str
    source_doc_id: str | None = None
    split: Literal["train", "validation", "calibration", "test", "challenge"]
    text: str = Field(repr=False)
    labels: list[str]
    label_source: str
    source: Literal["real", "synthetic"]

    @model_validator(mode="after")
    def validate_calibration_source(self) -> TrainingRow:
        if self.split in ("calibration", "challenge") and self.source != "synthetic":
            raise ValueError("Calibration and challenge examples must be synthetic")
        return self


def target_text(text: str) -> str:
    """Require exactly one nonempty target and never echo rejected input."""
    if text.count("[TARGET]") != 1 or text.count("[/TARGET]") != 1:
        raise ValueError("Each sample must have exactly one target marker pair")
    start = text.index("[TARGET]") + len("[TARGET]")
    end = text.index("[/TARGET]")
    if end <= start or not text[start:end].strip():
        raise ValueError("Sample has an empty or invalid target")
    return text[start:end]


def _load_rows_with_hash(
    path: Path, *, expected_split: str | None = None
) -> tuple[list[TrainingRow], str]:
    """Hash the exact bytes parsed, including blank lines and line endings."""
    rows: list[TrainingRow] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = TrainingRow.model_validate_json(line)
                if not all((row.id.strip(), row.doc_id.strip(), row.label_source.strip())):
                    raise ValueError("Missing provenance")
                if row.source_doc_id is not None and not row.source_doc_id.strip():
                    raise ValueError("Invalid original document ID")
                if len(set(row.labels)) != len(row.labels) or set(row.labels) - set(LABELS):
                    raise ValueError("Invalid labels")
                target_text(row.text)
            except ValueError:
                raise ValueError(f"Invalid dataset record at line {number}") from None
            if expected_split is not None and row.split != expected_split:
                raise ValueError(f"Unexpected split at line {number}; expected {expected_split}")
            if row.id in seen:
                raise ValueError(f"Duplicate sample ID at line {number}")
            if row.split in ("validation", "test") and row.source != "real":
                raise ValueError(f"Evaluation record is synthetic at line {number}")
            seen.add(row.id)
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows, digest.hexdigest()


def load_rows(path: Path, *, expected_split: str | None = None) -> list[TrainingRow]:
    """Validate local JSONL without putting document text in error messages."""
    return _load_rows_with_hash(path, expected_split=expected_split)[0]


def validate_training_splits(
    train_rows: Sequence[TrainingRow],
    validation_rows: Sequence[TrainingRow],
    calibration_rows: Sequence[TrainingRow] | None = None,
) -> None:
    """Reject split leakage before importing ML libraries or loading weights."""
    if not train_rows or not validation_rows:
        raise ValueError("Train and validation must both be nonempty")
    partitions = [(train_rows, "train"), (validation_rows, "validation")]
    if calibration_rows is not None:
        if not calibration_rows or any(row.source != "synthetic" for row in calibration_rows):
            raise ValueError("Calibration must contain separate synthetic examples only")
        partitions.append((calibration_rows, "calibration"))
    for rows, split in partitions:
        if any(row.split != split for row in rows):
            raise ValueError("Test or incorrectly assigned rows are forbidden during training")
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("Duplicate sample IDs within a split")
        original_groups: dict[str, str] = {}
        for row in rows:
            if row.source_doc_id is not None:
                if not row.source_doc_id.strip():
                    raise ValueError("Invalid original document ID")
                if original_groups.setdefault(row.source_doc_id, row.doc_id) != row.doc_id:
                    raise ValueError("Original document occurs in multiple document groups")
    if any(row.source != "real" for row in validation_rows):
        raise ValueError("Validation must contain real examples only")
    for index, (left, left_split) in enumerate(partitions):
        for right, right_split in partitions[index + 1 :]:
            for field in ("id", "doc_id", "source_doc_id"):
                left_values = {getattr(row, field) for row in left} - {None}
                if left_values & ({getattr(row, field) for row in right} - {None}):
                    raise ValueError(f"{left_split.capitalize()}/{right_split} overlap in {field}")
            left_targets = {" ".join(target_text(row.text).casefold().split()) for row in left}
            right_targets = {" ".join(target_text(row.text).casefold().split()) for row in right}
            if left_targets & right_targets:
                raise ValueError(f"Identical target text occurs in {left_split} and {right_split}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_summary(
    rows: Sequence[TrainingRow], path: Path, *, loaded_sha256: str | None = None
) -> dict[str, Any]:
    if loaded_sha256 is None:
        current_rows, loaded_sha256 = _load_rows_with_hash(path)
        if list(rows) != current_rows:
            raise ValueError("Dataset differs from loaded records")
    if sha256_file(path) != loaded_sha256:
        raise ValueError("Dataset changed after loading")
    real_rows = [row for row in rows if row.source == "real"]
    missing_original_ids = sum(row.source_doc_id is None for row in real_rows)
    return {
        "sha256": loaded_sha256,
        "samples": len(rows),
        "document_groups": len({row.doc_id for row in rows}),
        "known_original_documents": len(
            {row.source_doc_id for row in real_rows if row.source_doc_id is not None}
        ),
        "real_samples_missing_source_doc_id": missing_original_ids,
        "original_document_ids_complete": missing_original_ids == 0,
        "positive_counts": {label: sum(label in row.labels for row in rows) for label in LABELS},
        "negative_samples": sum(not row.labels for row in rows),
        "label_sources": dict(Counter(row.label_source for row in rows)),
        "sources": dict(Counter(row.source for row in rows)),
    }


def positive_weights(rows: Sequence[TrainingRow], cap: float = 10.0) -> list[float]:
    """Only training rows may contribute to the class imbalance correction."""
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("Positive weights require nonempty training rows only")
    weights: list[float] = []
    for label in LABELS:
        positive = sum(label in row.labels for row in rows)
        if positive == 0:
            raise ValueError(f"Training has no positives for {label}")
        if positive == len(rows):
            raise ValueError(f"Training has no negatives for {label}")
        weights.append(min(cap, max(1.0, (len(rows) - positive) / positive)))
    return weights


def accumulation_windows(
    size: int, batch_size: int, accumulation: int
) -> Iterator[tuple[range, ...]]:
    """Group every sample exactly once, including the last partial batch."""
    if min(size, batch_size, accumulation) < 1:
        raise ValueError("Dataset and batch sizes must be positive")
    step_size = batch_size * accumulation
    for start in range(0, size, step_size):
        end = min(start + step_size, size)
        yield tuple(
            range(batch, min(batch + batch_size, end)) for batch in range(start, end, batch_size)
        )


def _encode(tokenizer: Any, text: str, max_length: int) -> dict[str, list[int]]:
    target_text(text)
    start, end = text.index("[TARGET]"), text.index("[/TARGET]") + len("[/TARGET]")
    target = tokenizer.encode(text[start:end], add_special_tokens=False)
    remaining = max_length - tokenizer.num_special_tokens_to_add(pair=False) - len(target)
    if remaining < 0:
        raise ValueError("Target exceeds token budget; split the sample before training/inference")
    left = tokenizer.encode(text[:start], add_special_tokens=False)
    right = tokenizer.encode(text[end:], add_special_tokens=False)
    left_size = min(len(left), remaining // 2)
    right_size = min(len(right), remaining - left_size)
    left_size = min(len(left), remaining - right_size)
    ids = (left[-left_size:] if left_size else []) + target + right[:right_size]
    encoded: dict[str, list[int]] = dict(
        tokenizer.prepare_for_model(
            ids, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False
        )
    )
    return encoded


def _device(torch: Any, require_mps: bool) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if require_mps:
        raise RuntimeError("MPS is required but unavailable; CPU fallback refused")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        temporary.chmod(0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _batch(tokenizer: Any, encoded: Sequence[dict[str, list[int]]], device: str) -> dict[str, Any]:
    return {
        key: value.to(device)
        for key, value in tokenizer.pad(list(encoded), padding=True, return_tensors="pt").items()
    }


def _evaluate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    encoded: Sequence[dict[str, list[int]]],
    labels: Any,
    batch_size: int,
    device: str,
) -> tuple[float, list[list[float]]]:
    model.eval()
    total_loss = 0.0
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            stop = min(start + batch_size, len(encoded))
            logits = model(**_batch(tokenizer, encoded[start:stop], device)).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[start:stop].to(device), reduction="sum"
            )
            total_loss += float(loss.item())
            probabilities.extend(logits.sigmoid().cpu().tolist())
    return total_loss / (len(encoded) * len(LABELS)), probabilities


def train(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    model_name_or_path: str,
    *,
    config: TrainingConfig | None = None,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune and calibrate locally, without ever reading a test dataset."""
    config = config or TrainingConfig()
    train_rows, train_hash = _load_rows_with_hash(train_path, expected_split="train")
    validation_rows, validation_hash = _load_rows_with_hash(
        validation_path, expected_split="validation"
    )
    calibration_rows, calibration_hash = None, None
    if calibration_path is not None:
        calibration_rows, calibration_hash = _load_rows_with_hash(
            calibration_path, expected_split="calibration"
        )
        if config.requested_recall < 0.98:
            raise ValueError("Separate calibration requires at least 0.98 recall")
        if any(
            row.label_source != "dual_model_union_reference"
            for row in (*validation_rows, *calibration_rows)
        ):
            raise ValueError(
                "Separate calibration requires union-reference validation and calibration labels"
            )
    validate_training_splits(train_rows, validation_rows, calibration_rows)
    weights = positive_weights(train_rows, config.max_pos_weight)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest: dict[str, Any] = {
        "status": "initializing",
        "labels": list(LABELS),
        "base_model": model_name_or_path,
        "trainer_sha256": sha256_file(Path(__file__)),
        "evaluation_sha256": sha256_file(Path(__file__).with_name("evaluation.py")),
        "python_version": sys.version.split()[0],
        "hyperparameters": config.model_dump(),
        "dtype": "float32",
        "train": dataset_summary(train_rows, train_path, loaded_sha256=train_hash),
        "validation": dataset_summary(
            validation_rows, validation_path, loaded_sha256=validation_hash
        ),
        "input_integrity": "parsed_bytes_sha256; checked_before_and_after_fit",
        "pos_weight": dict(zip(LABELS, weights, strict=True)),
        "epoch_selection": "minimum_unweighted_validation_bce",
        "test_used": False,
        "production_ready": False,
        "epochs": [],
        "started_at_unix": time.time(),
    }
    manifest_path = output_dir / "manifest.json"
    inputs = [(train_path, train_hash), (validation_path, validation_hash)]
    if (
        calibration_rows is not None
        and calibration_path is not None
        and calibration_hash is not None
    ):
        manifest["calibration"] = dataset_summary(
            calibration_rows, calibration_path, loaded_sha256=calibration_hash
        )
        manifest["calibration_used_for_epoch_selection"] = False
        inputs.append((calibration_path, calibration_hash))
    _write_json(manifest_path, manifest)
    try:
        if any(sha256_file(path) != digest for path, digest in inputs):
            raise ValueError("Dataset changed before training")
        if calibration_rows is None:
            result = _train(train_rows, validation_rows, output_dir, config, manifest)
        else:
            result = _train(
                train_rows,
                validation_rows,
                output_dir,
                config,
                manifest,
                calibration_rows=calibration_rows,
            )
        if any(sha256_file(path) != digest for path, digest in inputs):
            raise ValueError("Dataset changed during training")
        return result
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        _write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Training failed ({type(error).__name__}); see aggregate manifest"
        ) from None


def _train(
    train_rows: Sequence[TrainingRow],
    validation_rows: Sequence[TrainingRow],
    output_dir: Path,
    config: TrainingConfig,
    manifest: dict[str, Any],
    *,
    calibration_rows: Sequence[TrainingRow] | None = None,
) -> dict[str, Any]:
    import numpy as np

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    device = _device(torch, config.require_mps)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "mps":
        torch.mps.manual_seed(config.seed)
    manifest.update(
        device=device,
        determinism="seeded; unsupported deterministic kernels warn, bitwise identity not guaranteed",
        package_versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "safetensors",
                "tokenizers",
                "scipy",
                "pydantic",
            )
        },
    )
    manifest["phase"] = "loading_model"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest["base_model"], local_files_only=config.local_files_only, trust_remote_code=False
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        manifest["base_model"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
        problem_type="multi_label_classification",
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device=device, dtype=torch.float32)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manifest["base_model_revision"] = getattr(model.config, "_commit_hash", None)
    manifest["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    manifest["phase"] = "tokenizing"
    train_encoded = [_encode(tokenizer, row.text, config.max_length) for row in train_rows]
    validation_encoded = [
        _encode(tokenizer, row.text, config.max_length) for row in validation_rows
    ]
    train_labels = torch.tensor(
        [[float(label in row.labels) for label in LABELS] for row in train_rows],
        dtype=torch.float32,
    )
    validation_labels = torch.tensor(
        [[float(label in row.labels) for label in LABELS] for row in validation_rows],
        dtype=torch.float32,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            list(manifest["pos_weight"].values()), dtype=torch.float32, device=device
        )
    )
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    steps_per_epoch = math.ceil(
        len(train_rows) / (config.batch_size * config.gradient_accumulation)
    )
    total_steps = steps_per_epoch * config.epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    manifest.update(status="training", optimizer_steps_planned=total_steps)
    _write_json(output_dir / "manifest.json", manifest)
    best_loss = math.inf
    best_probabilities: list[list[float]] = []
    optimizer_steps = 0
    for epoch in range(1, config.epochs + 1):
        manifest["phase"] = "training_epoch"
        started = time.monotonic()
        order = list(range(len(train_rows)))
        random.Random(config.seed + epoch).shuffle(order)
        model.train()
        total_loss = 0.0
        for window in accumulation_windows(
            len(order), config.batch_size, config.gradient_accumulation
        ):
            optimizer.zero_grad(set_to_none=True)
            window_size = sum(len(indices) for indices in window)
            for indices in window:
                chosen = [order[index] for index in indices]
                inputs = _batch(tokenizer, [train_encoded[index] for index in chosen], device)
                loss = criterion(model(**inputs).logits, train_labels[chosen].to(device))
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("Nonfinite training loss")
                total_loss += float(loss.detach().item()) * len(chosen)
                # Weight by examples, including a small final microbatch/window.
                (loss * (len(chosen) / window_size)).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
        manifest["phase"] = "validation_epoch"
        validation_loss, probabilities = _evaluate(
            torch,
            model,
            tokenizer,
            validation_encoded,
            validation_labels,
            config.batch_size,
            device,
        )
        if not math.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation loss")
        metric = {
            "epoch": epoch,
            "train_weighted_bce": total_loss / len(train_rows),
            "validation_bce": validation_loss,
            "seconds": time.monotonic() - started,
            "optimizer_steps": optimizer_steps,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        manifest["epochs"].append(metric)
        if validation_loss < best_loss:
            best_loss, best_probabilities = validation_loss, probabilities
            manifest["phase"] = "saving_best_model"
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            manifest["best_epoch"] = epoch
        _write_json(output_dir / "manifest.json", manifest)
        print(json.dumps(metric, allow_nan=False), flush=True)
    manifest["phase"] = "threshold_calibration"
    if calibration_rows is not None:
        manifest["calibration_checkpoint_epoch"] = manifest["best_epoch"]
        manifest["calibration_model_origin"] = "saved_best_real_validation_checkpoint"
    # Free optimizer state before loading the saved best checkpoint for separate
    # calibration. The last in-memory epoch may not be the selected epoch.
    del optimizer, scheduler, groups, criterion, model
    selection, utility = _calibrate(
        validation_rows,
        best_probabilities,
        output_dir,
        config,
        calibration_rows=calibration_rows,
        tokenizer=tokenizer,
        torch=torch,
        transformers=transformers,
        device=device,
    )
    _write_json(output_dir / "thresholds.json", selection)
    manifest.update(
        status="trained_pending_independent_test",
        phase="complete",
        best_validation_bce=best_loss,
        optimizer_steps=optimizer_steps,
        completed_at_unix=time.time(),
        thresholds=selection,
        validation_utility_gate=utility,
        development_gate={
            "thresholds_feasible": all(row["feasible"] for row in selection["per_label"].values()),
            "validation_utility_passed": utility["passed"],
            "passed": all(row["feasible"] for row in selection["per_label"].values())
            and utility["passed"],
            "production_ready": False,
        },
        artifact_sha256={
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.suffix in (".json", ".safetensors", ".model", ".txt")
            and path.name != "manifest.json"
        },
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _calibrate(
    validation_rows: Sequence[TrainingRow],
    best_probabilities: list[list[float]],
    output_dir: Path,
    config: TrainingConfig,
    *,
    calibration_rows: Sequence[TrainingRow] | None,
    tokenizer: Any,
    torch: Any,
    transformers: Any,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate only after epoch selection; optional scores use saved weights."""
    from .evaluation import (
        select_calibration_thresholds,
        select_validation_thresholds,
        validation_utility_gate,
    )

    validation_labels = [[int(label in row.labels) for label in LABELS] for row in validation_rows]
    provenance = "+".join(sorted({row.label_source for row in validation_rows}))
    if calibration_rows is None:
        selection = select_validation_thresholds(
            validation_labels,
            best_probabilities,
            split="validation",
            requested_recall=config.requested_recall,
            label_provenance=provenance,
        ).model_dump(mode="json")
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(output_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(device=device, dtype=torch.float32)
        encoded = [_encode(tokenizer, row.text, config.max_length) for row in calibration_rows]
        calibration_labels = torch.tensor(
            [[float(label in row.labels) for label in LABELS] for row in calibration_rows],
            dtype=torch.float32,
        )
        _, probabilities = _evaluate(
            torch, model, tokenizer, encoded, calibration_labels, config.batch_size, device
        )
        selection = select_calibration_thresholds(
            validation_labels,
            best_probabilities,
            calibration_labels.numpy(),
            probabilities,
            split="validation",
            calibration_split="calibration",
            label_provenance=provenance,
            calibration_label_provenance="+".join(
                sorted({row.label_source for row in calibration_rows})
            ),
            requested_recall=config.requested_recall,
        ).model_dump(mode="json")
    utility = validation_utility_gate(
        validation_labels,
        best_probabilities,
        selection["thresholds"],
        split="validation",
        label_provenance=provenance,
    )
    return selection, utility.model_dump(mode="json")


def verify_model_artifact(model_dir: Path) -> dict[str, Any]:
    """Verify local score-producing assets without requiring feasible thresholds."""
    try:
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "trained_pending_independent_test"
            or manifest.get("labels") != list(LABELS)
            or manifest.get("dtype") != "float32"
        ):
            raise ValueError("Invalid model state")
        hashes = manifest["artifact_sha256"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Missing artifact hashes")
        for name, expected in hashes.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            ):
                raise ValueError("Invalid artifact hash entry")
            path = model_dir / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                raise ValueError("Artifact integrity failure")
        if not {"config.json", "tokenizer_config.json", "thresholds.json"} <= hashes.keys():
            raise ValueError("Missing model configuration")
        if not any(name.endswith(".safetensors") for name in hashes):
            raise ValueError("Missing model weights")
        if not {"tokenizer.json", "spm.model", "vocab.txt"} & hashes.keys():
            raise ValueError("Missing tokenizer data")
        for path in model_dir.iterdir():
            if (
                path.name != "manifest.json"
                and path.suffix in (".json", ".safetensors", ".model", ".txt")
                and path.name not in hashes
            ):
                raise ValueError("Unverified model or tokenizer asset")
        selection = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
        if (
            selection != manifest["thresholds"]
            or selection["labels"] != list(LABELS)
            or selection["selected_on"] != "validation"
        ):
            raise ValueError("Invalid threshold provenance")
        max_length = manifest["hyperparameters"]["max_length"]
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 16:
            raise ValueError("Invalid token budget")
        return dict(manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("Model artifact is incomplete or failed integrity verification") from None


def predict(
    model_dir: Path, texts: Sequence[str], *, batch_size: int = 8, require_mps: bool = False
) -> list[list[float]]:
    """Return scores in LABELS order from a local, completed model artifact."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    manifest = verify_model_artifact(model_dir)
    if not texts:
        return []
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        device = _device(torch, require_mps)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).to(device=device, dtype=torch.float32)
        if model.config.id2label != dict(enumerate(LABELS)):
            raise ValueError("Model configuration has an incompatible label order")
        if verify_model_artifact(model_dir) != manifest:
            raise ValueError("Model manifest changed while loading")
        model.eval()
        results: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = [
                    _encode(tokenizer, text, manifest["hyperparameters"]["max_length"])
                    for text in texts[start : start + batch_size]
                ]
                logits = model(**_batch(tokenizer, encoded, device)).logits
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError("Nonfinite prediction")
                results.extend(logits.sigmoid().cpu().tolist())
        return results
    except Exception as error:
        raise RuntimeError(f"Prediction failed ({type(error).__name__})") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--train", type=Path, required=True)
    fitting.add_argument("--validation", type=Path, required=True)
    fitting.add_argument(
        "--calibration", type=Path, help="Separate fictional union-reference calibration JSONL"
    )
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--model", required=True)
    fitting.add_argument("--config", type=Path, help="Optional JSON TrainingConfig overrides")
    fitting.add_argument("--require-mps", action="store_true")
    scoring = commands.add_parser("predict")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--batch-size", type=int, default=8)
    scoring.add_argument("--require-mps", action="store_true")
    scoring.add_argument(
        "--split", choices=("train", "validation", "calibration", "test", "challenge")
    )
    args = parser.parse_args()
    try:
        if args.command == "train":
            options = json.loads(args.config.read_text()) if args.config else {}
            if args.require_mps:
                options["require_mps"] = True
            result = train(
                args.train,
                args.validation,
                args.output,
                args.model,
                config=TrainingConfig(**options),
                calibration_path=args.calibration,
            )
            print(json.dumps({"status": result["status"], "best_epoch": result["best_epoch"]}))
        else:
            rows = load_rows(args.input, expected_split=args.split)
            probabilities = predict(
                args.model,
                [row.text for row in rows],
                batch_size=args.batch_size,
                require_mps=args.require_mps,
            )
            with args.output.open("x", encoding="utf-8") as stream:
                args.output.chmod(0o600)
                for row, scores in zip(rows, probabilities, strict=True):
                    stream.write(
                        json.dumps(
                            {
                                "id": row.id,
                                "doc_id": row.doc_id,
                                "source_doc_id": row.source_doc_id,
                                "split": row.split,
                                "labels": row.labels,
                                "label_source": row.label_source,
                                "source": row.source,
                                "probabilities": scores,
                            }
                        )
                        + "\n"
                    )
            print(json.dumps({"predictions": len(rows)}))
    except Exception as error:
        # Library/validation exceptions may contain input text: print only class.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
