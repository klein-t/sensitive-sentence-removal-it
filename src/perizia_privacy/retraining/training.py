"""Local, preencoded, exploratory A1-A3 research fits. No source-text IO."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

LABELS = ("salute", "penale", "minori", "famiglia")
ROW_VERSION = "perizia-retraining-row-v1"
ARTIFACT_VERSION = "perizia-retraining-artifact-v1"
SPLITS = ("train", "dev_select", "calibrate", "regression_known", "fictional_regression")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class Row(Strict):
    schema_version: Literal["perizia-retraining-row-v1"]
    id: str
    parent_id: str
    group_id: str
    source_doc_id: str
    source_hash: str
    source: Literal["real", "synthetic"]
    split: Literal["train", "dev_select", "calibrate", "regression_known", "fictional_regression"]
    policy_sha256: str
    reference_sha256: str
    reference_tier: str
    view_hash: str
    input_ids: list[int] = Field(min_length=1, max_length=384)
    focus_mask: list[bool]
    target_hash: str
    labels: list[bool] = Field(min_length=4, max_length=4)
    loss_mask: list[bool] = Field(min_length=4, max_length=4)
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def check(self) -> Row:
        for value in (
            self.id,
            self.parent_id,
            self.group_id,
            self.source_doc_id,
            self.reference_tier,
        ):
            if not value.strip():
                raise ValueError("empty provenance")
        for value in (
            self.source_hash,
            self.policy_sha256,
            self.reference_sha256,
            self.view_hash,
            self.target_hash,
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid digest")
        if (
            len(self.focus_mask) != len(self.input_ids)
            or not any(self.focus_mask)
            or any(token < 0 for token in self.input_ids)
            or not math.isfinite(self.weight)
        ):
            raise ValueError("invalid encoded view")
        if self.split == "fictional_regression" and self.source != "synthetic":
            raise ValueError("fictional regression requires synthetic source")
        if self.split in ("dev_select", "calibrate", "regression_known") and self.source != "real":
            raise ValueError("evaluation must be real")
        return self


class Config(Strict):
    arm: Literal["A1", "A2", "A3"]
    experiment_kind: Literal["real_supported", "synthetic_supported_exploratory"] = "real_supported"
    checkpoint_rule: Literal["minimum_dev_bce", "fixed_final_epoch"] = "minimum_dev_bce"
    seed: int = 42
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    microbatch: int = Field(default=4, ge=1)
    accumulation: int = Field(default=8, ge=1)
    epochs: int = Field(default=5, ge=1, le=5)
    clip_norm: float = Field(default=1.0, gt=0)
    require_mps: bool = True
    gradient_checkpointing: bool = True

    @model_validator(mode="after")
    def check_mode(self) -> Config:
        if (
            self.checkpoint_rule == "fixed_final_epoch"
            and self.experiment_kind != "synthetic_supported_exploratory"
        ):
            raise ValueError("fixed final epoch requires explicit exploratory mode")
        if (
            self.experiment_kind == "synthetic_supported_exploratory"
            and self.checkpoint_rule != "fixed_final_epoch"
        ):
            raise ValueError("exploratory mode requires fixed final epoch")
        if self.experiment_kind == "synthetic_supported_exploratory" and self.arm == "A1":
            raise ValueError("exploratory synthetic mode requires A2 or A3")
        if self.experiment_kind == "synthetic_supported_exploratory" and (
            self.seed != 42 or self.epochs != 5
        ):
            raise ValueError("exploratory recipe requires seed 42 and five fixed epochs")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, split: str) -> tuple[list[Row], str]:
    if split not in SPLITS:
        raise ValueError("invalid split")
    digest = hashlib.sha256()
    rows: list[Row] = []
    seen: set[str] = set()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = Row.model_validate_json(line)
            except Exception:
                raise ValueError(f"invalid row at line {line_number}") from None
            if row.split != split or row.id in seen:
                raise ValueError(f"split or duplicate ID at line {line_number}")
            seen.add(row.id)
            rows.append(row)
    if not rows:
        raise ValueError("empty dataset")
    return rows, digest.hexdigest()


def validate_splits(partitions: dict[str, list[Row]]) -> tuple[str, str]:
    if not {"train", "dev_select"} <= partitions.keys():
        raise ValueError("training needs train and dev_select")
    policy = {r.policy_sha256 for rows in partitions.values() for r in rows}
    reference = {r.reference_sha256 for rows in partitions.values() for r in rows}
    if len(policy) != 1 or len(reference) != 1:
        raise ValueError("mixed policy or reference")
    for split, rows in partitions.items():
        if split not in SPLITS or not rows or any(r.split != split for r in rows):
            raise ValueError("invalid partition")
        if split == "fictional_regression" and any(r.source != "synthetic" for r in rows):
            raise ValueError("fictional regression must be synthetic")
        if split not in ("train", "fictional_regression") and any(r.source != "real" for r in rows):
            raise ValueError("nonreal evaluation")
    for key in (
        "id",
        "parent_id",
        "group_id",
        "source_doc_id",
        "source_hash",
        "target_hash",
        "view_hash",
    ):
        seen: set[str] = set()
        for rows in partitions.values():
            values = {getattr(r, key) for r in rows}
            if seen & values:
                raise ValueError(f"cross split {key}")
            seen.update(values)
    return next(iter(policy)), next(iter(reference))


def positive_weights(rows: list[Row]) -> tuple[list[float], list[dict[str, int]]]:
    """Freeze class weights from real trusted training entries for matched arms."""
    counts = [{"positive": 0, "negative": 0} for _ in LABELS]
    for row in rows:
        if row.source != "real":
            continue
        for i, trusted in enumerate(row.loss_mask):
            if trusted:
                counts[i]["positive" if row.labels[i] else "negative"] += 1
    weights = [
        min(10.0, max(1.0, c["negative"] / c["positive"])) if c["positive"] else 1.0 for c in counts
    ]
    return weights, counts


def effective_positive_weights(rows: list[Row]) -> tuple[list[float], list[dict[str, float]]]:
    """Weighted trusted entries in the actual exploratory mixture."""
    counts = [{"positive": 0.0, "negative": 0.0} for _ in LABELS]
    for row in rows:
        for i, trusted in enumerate(row.loss_mask):
            if trusted:
                counts[i]["positive" if row.labels[i] else "negative"] += row.weight
    weights = [
        min(10.0, max(1.0, c["negative"] / c["positive"])) if c["positive"] else 1.0 for c in counts
    ]
    return weights, counts


def assert_train_support(counts: list[dict[str, float]]) -> None:
    if any(not c["positive"] or not c["negative"] for c in counts):
        raise ValueError("all four heads need trusted positive and negative train entries")


def synthetic_fraction(rows: list[Row]) -> list[float]:
    total = [0.0] * 4
    synthetic = [0.0] * 4
    for row in rows:
        for i, trusted in enumerate(row.loss_mask):
            if trusted:
                total[i] += row.weight
                if row.source == "synthetic":
                    synthetic[i] += row.weight
    return [synthetic[i] / total[i] if total[i] else 0.0 for i in range(4)]


def synthetic_effective_fraction(rows: list[Row], pos_weights: list[float]) -> list[float]:
    """Synthetic share of the actual weighted per-head BCE denominator."""
    if len(pos_weights) != 4 or any(weight < 1 or weight > 10 for weight in pos_weights):
        raise ValueError("invalid positive weights")
    total = [0.0] * 4
    synthetic = [0.0] * 4
    for row in rows:
        for i, trusted in enumerate(row.loss_mask):
            if trusted:
                contribution = row.weight * (pos_weights[i] if row.labels[i] else 1.0)
                total[i] += contribution
                if row.source == "synthetic":
                    synthetic[i] += contribution
    return [synthetic[i] / total[i] if total[i] else 0.0 for i in range(4)]


def synthetic_positive_fraction(rows: list[Row]) -> list[float]:
    positives = [0.0] * 4
    synthetic = [0.0] * 4
    for row in rows:
        for i, trusted in enumerate(row.loss_mask):
            if trusted and row.labels[i]:
                positives[i] += row.weight
                if row.source == "synthetic":
                    synthetic[i] += row.weight
    return [synthetic[i] / positives[i] if positives[i] else 0.0 for i in range(4)]


def update_windows(rows: list[Row], config: Config, epoch: int) -> list[list[list[Row]]]:
    """Consume every real row once; deterministically insert eligible synthetic rows."""
    real = [r for r in rows if r.source == "real"]
    synthetic = [r for r in rows if r.source == "synthetic"]
    rng = random.Random(config.seed + epoch)
    rng.shuffle(real)
    rng.shuffle(synthetic)
    pos_weights = (
        [10.0] * 4
        if config.experiment_kind == "synthetic_supported_exploratory"
        else positive_weights(rows)[0]
    )
    if synthetic and config.microbatch < 2:
        raise ValueError("synthetic mixture needs microbatch at least two")
    stride = config.microbatch - 1 if synthetic else config.microbatch
    microbatches = [real[i : i + stride] for i in range(0, len(real), stride)]
    for row in synthetic:
        eligible = [
            batch
            for batch in microbatches
            if len(batch) < config.microbatch
            and not any(r.source == "synthetic" for r in batch)
            and all(value <= 0.30 + 1e-9 for value in synthetic_fraction(batch + [row]))
            and all(
                value <= 0.30 + 1e-9
                for value in synthetic_effective_fraction(batch + [row], pos_weights)
            )
        ]
        if not eligible:
            raise ValueError("cannot place synthetic row within per-microbatch cap")
        eligible[0].append(row)
    if config.experiment_kind == "real_supported" and any(
        value > 0.30 + 1e-9 for value in synthetic_positive_fraction(rows)
    ):
        raise ValueError("synthetic positive contribution exceeds 30%")
    return [
        microbatches[i : i + config.accumulation]
        for i in range(0, len(microbatches), config.accumulation)
    ]


def masked_loss(
    logits: Any,
    labels: Any,
    mask: Any,
    weights: Any,
    pos_weights: Any,
    denominator: Any,
    torch: Any,
) -> Any:
    """Sum over trusted entries; caller uses the whole accumulation window denominator."""
    per_entry = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weights, reduction="none"
    )
    return (per_entry * mask * weights[:, None]).sum() / denominator


def focus_mean(hidden: Any, focus_mask: Any, torch: Any) -> Any:
    mask = focus_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _logits(model: Any, ids: Any, attention: Any, focus: Any, arm: str, torch: Any) -> Any:
    if arm != "A3":
        return model(input_ids=ids, attention_mask=attention).logits
    hidden = model.deberta(input_ids=ids, attention_mask=attention).last_hidden_state
    # ContextPooler keeps the original dense/activation/dropout; singleton selects mean.
    pooled = model.pooler(focus_mean(hidden, focus, torch).unsqueeze(1))
    return model.classifier(model.dropout(pooled))


def _batch(rows: list[Row], pad_id: int, device: str, torch: Any) -> tuple[Any, ...]:
    width = max(len(row.input_ids) for row in rows)
    ids = torch.tensor(
        [row.input_ids + [pad_id] * (width - len(row.input_ids)) for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention = torch.tensor(
        [[1] * len(row.input_ids) + [0] * (width - len(row.input_ids)) for row in rows],
        dtype=torch.long,
        device=device,
    )
    focus = torch.tensor(
        [row.focus_mask + [False] * (width - len(row.input_ids)) for row in rows],
        dtype=torch.bool,
        device=device,
    )
    labels = torch.tensor([row.labels for row in rows], dtype=torch.float32, device=device)
    masks = torch.tensor([row.loss_mask for row in rows], dtype=torch.float32, device=device)
    weights = torch.tensor([row.weight for row in rows], dtype=torch.float32, device=device)
    return ids, attention, focus, labels, masks, weights


def _write_private(path: Path, data: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _replace_private(path: Path, data: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".scores-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        os.chmod(temporary, 0o600)
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _score(
    rows: list[Row],
    model: Any,
    arm: str,
    pad_id: int,
    device: str,
    torch: Any,
    microbatch: int,
    require_trusted: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), microbatch):
            block = rows[start : start + microbatch]
            ids, attention, focus, labels, masks, weights = _batch(block, pad_id, device, torch)
            logits = _logits(model, ids, attention, focus, arm, torch)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite logits")
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            numerator += float((losses * masks).sum().item())
            denominator += float(masks.sum().item())
            for row, scores in zip(block, logits.sigmoid().cpu().tolist(), strict=True):
                records.append(
                    {
                        "id": row.id,
                        "split": row.split,
                        "group_id": row.group_id,
                        "source": row.source,
                        "labels": row.labels,
                        "loss_mask": row.loss_mask,
                        "scores": scores,
                    }
                )
    if denominator <= 0 and require_trusted:
        raise ValueError("no trusted evaluation entries")
    return numerator / denominator if denominator else 0.0, records


def threshold_receipt(rows: list[Row], scores: list[list[float]], split: str) -> dict[str, Any]:
    if split not in ("dev_select", "calibrate") or any(
        r.split != split or r.source != "real" for r in rows
    ):
        raise ValueError("threshold partition must be real dev or calibration")
    if len(rows) != len(scores) or any(
        len(score) != 4 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in score)
        for score in scores
    ):
        raise ValueError("invalid score matrix")
    result: dict[str, Any] = {
        "schema_version": "perizia-threshold-receipt-v1",
        "split": split,
        "labels": list(LABELS),
        "per_label": {},
    }
    for i, label in enumerate(LABELS):
        positives = [
            (row, score[i])
            for row, score in zip(rows, scores, strict=True)
            if row.loss_mask[i] and row.labels[i]
        ]
        groups = {row.group_id for row, _ in positives}
        if len(positives) < 30 or len(groups) < 10:
            result["per_label"][label] = {
                "status": "insufficient_support",
                "positives": len(positives),
                "groups": len(groups),
                "threshold": None,
            }
            continue
        if any(not math.isfinite(score) for _, score in positives):
            raise ValueError("nonfinite score")
        chosen = max(
            t
            for t in {score for _, score in positives}
            if sum(score >= t for _, score in positives) / len(positives) >= 0.99
        )
        result["per_label"][label] = {
            "status": "supported",
            "positives": len(positives),
            "groups": len(groups),
            "threshold": chosen,
        }
    return result


def _artifact_hashes(folder: Path) -> dict[str, str]:
    return {
        p.name: sha256_file(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }


def _parameter_hash(model: Any, *, encoder: bool) -> str:
    digest = hashlib.sha256()
    found = False
    for name, tensor in sorted(model.state_dict().items()):
        is_encoder = name.startswith("deberta.")
        if is_encoder != encoder:
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if not found:
        raise ValueError("missing expected model parameters")
    return digest.hexdigest()


def verify_artifact(folder: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["schema_version"] != ARTIFACT_VERSION
            or manifest["status"] != "candidate_exploratory_uncalibrated"
            or manifest["labels"] != list(LABELS)
            or manifest["arm"] not in ("A1", "A2", "A3")
            or not manifest["base_artifact_sha256"]
            or manifest["experiment_kind"]
            not in ("real_supported", "synthetic_supported_exploratory")
            or manifest["checkpoint_rule"] not in ("minimum_dev_bce", "fixed_final_epoch")
            or manifest["config"]["experiment_kind"] != manifest["experiment_kind"]
            or manifest["config"]["checkpoint_rule"] != manifest["checkpoint_rule"]
            or manifest["fit_gates"]["eligible_for_selection"] is not False
            or manifest["production_ready"] is not False
            or manifest["export_allowed"] is not False
            or manifest["thresholds"] is not None
        ):
            raise ValueError("wrong artifact contract")
        Config.model_validate(manifest["config"])
        hashes = manifest["artifact_sha256"]
        if not {"model.safetensors", "config.json", "tokenizer_config.json"} <= hashes.keys():
            raise ValueError("missing assets")
        if hashes != _artifact_hashes(folder):
            raise ValueError("artifact hashes changed")
        return cast("dict[str, Any]", manifest)
    except Exception:
        raise ValueError("invalid research artifact") from None


def train(
    train_path: Path,
    dev_path: Path,
    output: Path,
    base: Path,
    base_hashes: dict[str, str],
    config: Config,
    tokenizer_dir: Path | None = None,
    tokenizer_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    train_rows, train_hash = load_rows(train_path, "train")
    dev_rows, dev_hash = load_rows(dev_path, "dev_select")
    policy, reference = validate_splits({"train": train_rows, "dev_select": dev_rows})
    real_pos_weights, real_counts = positive_weights(train_rows)
    counts: list[dict[str, float]]
    if config.experiment_kind == "real_supported":
        counts = [{key: float(value) for key, value in entry.items()} for entry in real_counts]
        assert_train_support(counts)
        pos_weights = real_pos_weights
    else:
        _, counts = effective_positive_weights(train_rows)
        pos_weights = [10.0] * 4
        assert_train_support(counts)
    if any(not any(row.loss_mask) for row in train_rows):
        raise ValueError("training row has no trusted labels")
    if config.arm == "A1" and any(row.source == "synthetic" for row in train_rows):
        raise ValueError("A1 requires real training rows")
    if not {"config.json", "model.safetensors"} <= base_hashes.keys() or any(
        (Path(name).name != name or len(digest) != 64 or sha256_file(base / name) != digest)
        for name, digest in base_hashes.items()
    ):
        raise ValueError("base artifact hash mismatch")
    tokenizer_dir = tokenizer_dir or base
    tokenizer_hashes = tokenizer_hashes or base_hashes
    if "tokenizer_config.json" not in tokenizer_hashes or any(
        Path(name).name != name or len(digest) != 64 or sha256_file(tokenizer_dir / name) != digest
        for name, digest in tokenizer_hashes.items()
    ):
        raise ValueError("tokenizer artifact hash mismatch")
    if output.exists():
        raise FileExistsError("output exists")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    safetensors = importlib.import_module("safetensors.torch")
    if config.require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    device = "mps" if config.require_mps else "cpu"
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.seed)
    with contextlib.suppress(ImportError):
        importlib.import_module("numpy").random.seed(config.seed)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(base),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        num_labels=4,
        ignore_mismatched_sizes=True,
    ).to(device=device, dtype=torch.float32)
    if model.config.model_type != "deberta-v2" or not hasattr(model, "pooler"):
        raise ValueError("expected DeBERTa V2 sequence classification")
    head_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key.startswith(("pooler.", "classifier."))
    }
    if not head_state:
        raise ValueError("classification head not found")
    initial_head_sha256 = hashlib.sha256(safetensors.save(head_state)).hexdigest()
    initial_encoder_sha256 = _parameter_hash(model, encoder=True)
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None or any(
        max(row.input_ids) >= model.config.vocab_size for row in train_rows + dev_rows
    ):
        raise ValueError("tokenizer mismatch")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith("bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
    all_windows = [
        update_windows(train_rows, config, epoch) for epoch in range(1, config.epochs + 1)
    ]
    order_hashes = [
        hashlib.sha256(
            json.dumps(
                [
                    [
                        [
                            {"id": row.id, "weight": row.weight, "loss_mask": row.loss_mask}
                            for row in batch
                        ]
                        for batch in window
                    ]
                    for window in windows
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for windows in all_windows
    ]
    total_updates = sum(map(len, all_windows))
    warmup = math.ceil(total_updates * config.warmup_ratio)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup, total_updates)
    output.mkdir(mode=0o700, parents=True)
    best = float("inf")
    minimum_observed_dev_bce = float("inf")
    best_epoch = 0
    updates = 0
    epochs: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    gradient_norm_max = 0.0
    gradient_norm_min = float("inf")
    pw = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    for epoch, windows in enumerate(all_windows, 1):
        started = time.monotonic()
        model.train()
        for microbatches in windows:
            all_rows = [row for batch in microbatches for row in batch]
            exposure.append(
                {
                    "epoch": epoch,
                    "windows": len(all_rows),
                    "tokens": sum(len(row.input_ids) for row in all_rows),
                    "trusted_by_head": [
                        sum(row.loss_mask[i] for row in all_rows) for i in range(4)
                    ],
                    "synthetic_weighted_fraction_by_head": synthetic_fraction(all_rows),
                    "synthetic_effective_loss_fraction_by_head": synthetic_effective_fraction(
                        all_rows, pos_weights
                    ),
                }
            )
            denominator = sum(row.weight * sum(row.loss_mask) for row in all_rows)
            if denominator <= 0:
                raise ValueError("empty trusted update")
            optimizer.zero_grad(set_to_none=True)
            for batch in microbatches:
                ids, attention, focus, labels, masks, weights = _batch(batch, pad_id, device, torch)
                logits = _logits(model, ids, attention, focus, config.arm, torch)
                loss = masked_loss(logits, labels, masks, weights, pw, denominator, torch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.clip_norm, error_if_nonfinite=True
            )
            norm = float(gradient_norm.detach().item())
            if not math.isfinite(norm):
                raise RuntimeError("nonfinite gradient norm")
            gradient_norm_min = min(gradient_norm_min, norm)
            gradient_norm_max = max(gradient_norm_max, norm)
            optimizer.step()
            scheduler.step()
            updates += 1
            if updates % 5 == 0 or updates == total_updates:
                print(
                    json.dumps(
                        {
                            "phase": "optimizer_step",
                            "updates": updates,
                            "total_updates": total_updates,
                            "epoch": epoch,
                            "gradient_norm_before_clip": norm,
                        }
                    ),
                    flush=True,
                )
        dev_bce, dev_records = _score(
            dev_rows, model, config.arm, pad_id, device, torch, config.microbatch
        )
        train_epoch_bce, _ = _score(
            train_rows, model, config.arm, pad_id, device, torch, config.microbatch
        )
        epochs.append(
            {
                "epoch": epoch,
                "dev_masked_bce": dev_bce,
                "train_masked_bce_unweighted": train_epoch_bce,
                "updates": updates,
            }
        )
        epochs[-1]["seconds"] = time.monotonic() - started
        print(json.dumps(epochs[-1], allow_nan=False), flush=True)
        minimum_observed_dev_bce = min(minimum_observed_dev_bce, dev_bce)
        select_checkpoint = (
            epoch == config.epochs
            if config.checkpoint_rule == "fixed_final_epoch"
            else dev_bce < best
        )
        if select_checkpoint:
            best, best_epoch = dev_bce, epoch
            safetensors.save_file(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(output / "model.safetensors"),
            )
            os.chmod(output / "model.safetensors", 0o600)
            model.config.save_pretrained(output)
            tokenizer.save_pretrained(output)
            for asset in output.iterdir():
                asset.chmod(0o600)
            _replace_private(output / "dev_scores.json", dev_records)
    # Reload the selected epoch to prove scoring uses the saved checkpoint.
    state = safetensors.load_file(str(output / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    final_encoder_sha256 = _parameter_hash(model, encoder=True)
    final_head_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key.startswith(("pooler.", "classifier."))
    }
    final_head_sha256 = hashlib.sha256(safetensors.save(final_head_state)).hexdigest()
    if final_encoder_sha256 == initial_encoder_sha256 or final_head_sha256 == initial_head_sha256:
        raise RuntimeError("encoder or classification head did not update")
    _, reloaded_dev_records = _score(
        dev_rows, model, config.arm, pad_id, device, torch, config.microbatch
    )
    selected_dev_records = json.loads((output / "dev_scores.json").read_text(encoding="utf-8"))
    if [record["id"] for record in reloaded_dev_records] != [
        record["id"] for record in selected_dev_records
    ]:
        raise RuntimeError("reloaded dev row order mismatch")
    reload_max_abs_diff = max(
        abs(a - b)
        for old, new in zip(selected_dev_records, reloaded_dev_records, strict=True)
        for a, b in zip(old["scores"], new["scores"], strict=True)
    )
    if reload_max_abs_diff > 1e-5:
        raise RuntimeError("reloaded checkpoint predictions differ")
    train_bce, train_records = _score(
        train_rows, model, config.arm, pad_id, device, torch, config.microbatch
    )
    _write_private(output / "train_scores.json", train_records)
    if sha256_file(train_path) != train_hash or sha256_file(dev_path) != dev_hash:
        raise ValueError("dataset changed during training")
    if any(sha256_file(base / name) != digest for name, digest in base_hashes.items()):
        raise ValueError("base artifact changed during training")
    if any(
        sha256_file(tokenizer_dir / name) != digest for name, digest in tokenizer_hashes.items()
    ):
        raise ValueError("tokenizer artifact changed during training")
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "status": "candidate_exploratory_uncalibrated",
        "production_ready": False,
        "export_allowed": False,
        "thresholds": None,
        "labels": list(LABELS),
        "arm": config.arm,
        "public_experiment_name": "E-sequence"
        if config.arm == "A2" and config.experiment_kind == "synthetic_supported_exploratory"
        else "E-focus"
        if config.arm == "A3" and config.experiment_kind == "synthetic_supported_exploratory"
        else config.arm,
        "experiment_kind": config.experiment_kind,
        "checkpoint_rule": config.checkpoint_rule,
        "config": config.model_dump(),
        "policy_sha256": policy,
        "reference_sha256": reference,
        "dataset_sha256": {"train": train_hash, "dev_select": dev_hash},
        "base_artifact_sha256": base_hashes,
        "tokenizer_artifact_sha256": tokenizer_hashes,
        "code_sha256": sha256_file(Path(__file__)),
        "seed": config.seed,
        "device": device,
        "float_dtype": "float32",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_nonencoder_head_sha256": initial_head_sha256,
        "selected_nonencoder_head_sha256": final_head_sha256,
        "initial_encoder_sha256": initial_encoder_sha256,
        "selected_encoder_sha256": final_encoder_sha256,
        "reload_prediction_max_abs_diff": reload_max_abs_diff,
        "gradient_norm_before_clip_min": gradient_norm_min,
        "gradient_norm_before_clip_max": gradient_norm_max,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "determinism_limit": "seeded order and initialization; MPS not bitwise deterministic",
        "updates": updates,
        "best_epoch": best_epoch,
        "best_dev_masked_bce": best,
        "minimum_observed_dev_bce": minimum_observed_dev_bce,
        "train_masked_bce": train_bce,
        "train_mask_counts": counts,
        "real_train_mask_counts": real_counts,
        "positive_weights": pos_weights,
        "positive_weight_rationale": "fixed_cap_10_exploratory_no_real_positive_support"
        if config.experiment_kind == "synthetic_supported_exploratory"
        else "real_trusted_train_negative_positive_ratio_capped_1_to_10",
        "synthetic_weighted_fraction_by_head": dict(
            zip(LABELS, synthetic_fraction(train_rows), strict=True)
        ),
        "synthetic_effective_loss_fraction_by_head": dict(
            zip(LABELS, synthetic_effective_fraction(train_rows, pos_weights), strict=True)
        ),
        "synthetic_positive_fraction_by_head": dict(
            zip(LABELS, synthetic_positive_fraction(train_rows), strict=True)
        ),
        "zero_real_positive_heads": [
            LABELS[i] for i, c in enumerate(real_counts) if c["positive"] == 0
        ],
        "fit_gates": {
            "eligible_for_selection": False,
            "reason": "synthetic_supported_exploratory"
            if config.experiment_kind == "synthetic_supported_exploratory"
            else "development_evidence_not_assessed",
        },
        "dev_mask_counts": positive_weights(dev_rows)[1],
        "epochs": epochs,
        "epoch_order_sha256": order_hashes,
        "exposure_by_update": exposure,
        "artifact_sha256": _artifact_hashes(output),
    }
    _write_private(output / "manifest.json", manifest)
    verify_artifact(output)
    return manifest


def predict(folder: Path, rows: list[Row], *, require_mps: bool = True) -> list[list[float]]:
    manifest = verify_artifact(folder)
    if not rows or any(
        r.policy_sha256 != manifest["policy_sha256"]
        or r.reference_sha256 != manifest["reference_sha256"]
        for r in rows
    ):
        raise ValueError("prediction row contract mismatch")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    safetensors = importlib.import_module("safetensors.torch")
    if require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    device = "mps" if require_mps else "cpu"
    model = transformers.AutoModelForSequenceClassification.from_config(
        transformers.AutoConfig.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False
        )
    )
    model.load_state_dict(
        safetensors.load_file(str(folder / "model.safetensors"), device="cpu"), strict=True
    )
    model.to(device=device, dtype=torch.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False
    )
    if verify_artifact(folder) != manifest:
        raise ValueError("artifact changed during load")
    return [
        record["scores"]
        for record in _score(
            rows,
            model,
            manifest["arm"],
            tokenizer.pad_token_id,
            device,
            torch,
            4,
            require_trusted=False,
        )[1]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fitting = sub.add_parser("train")
    for name in ("train", "dev", "output", "base", "config", "base-hashes"):
        fitting.add_argument("--" + name, type=Path, required=True)
    fitting.add_argument("--tokenizer-dir", type=Path)
    fitting.add_argument("--tokenizer-hashes", type=Path)
    scoring = sub.add_parser("predict")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--split", choices=SPLITS, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--allow-cpu", action="store_true")
    try:
        args = parser.parse_args()
        if args.command == "train":
            result = train(
                args.train,
                args.dev,
                args.output,
                args.base,
                json.loads(args.base_hashes.read_text(encoding="utf-8")),
                Config.model_validate_json(args.config.read_bytes()),
                tokenizer_dir=args.tokenizer_dir,
                tokenizer_hashes=json.loads(args.tokenizer_hashes.read_text(encoding="utf-8"))
                if args.tokenizer_hashes
                else None,
            )
            print(json.dumps({"status": result["status"], "best_epoch": result["best_epoch"]}))
        else:
            rows, _ = load_rows(args.input, args.split)
            scores = predict(args.model, rows, require_mps=not args.allow_cpu)
            _write_private(
                args.output,
                [{"id": row.id, "scores": score} for row, score in zip(rows, scores, strict=True)],
            )
            print(json.dumps({"predictions": len(rows)}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
