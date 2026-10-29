"""Persistent local detection and deterministic whole-sentence redaction.

Errors return no document text. Artifact integrity and complete target coverage
are checked independently of the classifier's statistical quality; deployment
approval still requires the independent evaluation gate.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import LABELS
from .corpus import example_text, sentence_spans, windows
from .training import _batch, _encode, sha256_file, target_text

if TYPE_CHECKING:
    from collections.abc import Sequence


_AMENDMENT_VERSION = 1
# Amendment 2 keeps the same ratified per-label budget and the same gate rule;
# it only changes how each retained label's threshold is selected, so it must
# also carry the out-of-fold evidence that justified the selection.
_AMENDMENT_VERSIONS = frozenset({_AMENDMENT_VERSION, 2})
_GATE_RULE_VERSION = 2


class RedactionError(RuntimeError):
    """Safe failure: callers must stop processing, never forward the input."""


class RedactionSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int
    labels: tuple[str, ...]


class RedactionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(repr=False)
    counts: dict[str, int]
    spans: tuple[RedactionSpan, ...]
    scanned_windows: int
    scanned_sentences: int


def _removal_labels(manifest: dict[str, Any], selection: dict[str, Any]) -> frozenset[str]:
    """Check a ratified amendment and return the labels allowed to remove text.

    A label the project owner excluded from removal keeps a recorded threshold for
    reporting only; it can never fire, so the amendment must state both sides of
    the split and its own per-label utility evidence.
    """
    amendment = manifest.get("amendment")
    version = amendment.get("version") if isinstance(amendment, dict) else None
    if (
        not isinstance(amendment, dict)
        or isinstance(version, bool)
        or version not in _AMENDMENT_VERSIONS
    ):
        raise ValueError("Amended gate rule without a recorded amendment")
    removal = amendment.get("removal_labels")
    thresholds = amendment.get("thresholds")
    gate = amendment.get("per_label_gate")
    if (
        not isinstance(removal, list)
        or not removal
        or len(set(removal)) != len(removal)
        or not set(removal) < set(LABELS)
        or amendment.get("excluded_labels") != [item for item in LABELS if item not in removal]
        or not isinstance(thresholds, dict)
        or set(thresholds) != set(removal)
        or not isinstance(gate, dict)
        or set(gate) != set(removal)
    ):
        raise ValueError("Invalid amendment removal set")
    for label in removal:
        entry, rate, limit = gate[label], None, None
        if isinstance(entry, dict):
            rate, limit = entry.get("false_positive_rate"), entry.get("maximum_false_positive_rate")
        if (
            not isinstance(entry, dict)
            or entry.get("passed") is not True
            or isinstance(rate, bool)
            or isinstance(limit, bool)
            or not isinstance(rate, (int, float))
            or not isinstance(limit, (int, float))
            or not 0 <= rate <= limit <= 0.05
            or thresholds[label] != selection["per_label"][label]["threshold"]
        ):
            raise ValueError("Amendment does not satisfy its recorded per-label gate")
    if version == 2:
        _check_out_of_fold_evidence(amendment.get("out_of_fold_selection"), removal)
    return frozenset(removal)


def _check_out_of_fold_evidence(evidence: Any, removal: list[str]) -> None:
    """Amendment 2 selects on held-out folds, so the held-out numbers must travel.

    Without them a v2 artifact could record a threshold chosen on the same rows
    the budget is measured on, which is exactly the optimism amendment 2 fixes.
    """
    if not isinstance(evidence, dict) or set(evidence) != set(removal):
        raise ValueError("Amendment 2 requires out-of-fold evidence for every retained label")
    for label in removal:
        entry = evidence[label]
        recall = entry.get("recall_ci_lower") if isinstance(entry, dict) else None
        rate = entry.get("standalone_removal_rate") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or isinstance(recall, bool)
            or isinstance(rate, bool)
            or not isinstance(recall, (int, float))
            or not isinstance(rate, (int, float))
            or not 0 <= recall <= 1
            or not 0 <= rate <= 0.05
        ):
            raise ValueError("Amendment 2 out-of-fold evidence is missing or out of budget")


def _verified_artifact(model_dir: Path) -> tuple[dict[str, Any], tuple[float | None, ...]]:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "trained_pending_independent_test"
        or manifest["labels"] != list(LABELS)
        or manifest.get("dtype") != "float32"
    ):
        raise ValueError("Invalid model manifest")
    # New research artifacts record both sensitivity and utility gates. Legacy
    # artifacts lack this field; their per-label feasibility is still checked.
    # Rule version 2 replaces the single OR-removal gate with the ratified
    # per-label budget, so an artifact carrying it has no OR-gate key and an
    # unamended reader fails closed instead of removing on every label.
    gate = manifest.get("development_gate")
    amended = isinstance(gate, dict) and gate.get("gate_rule_version") == _GATE_RULE_VERSION
    if gate is not None:
        version = gate.get("gate_rule_version", 1) if isinstance(gate, dict) else None
        utility_key = {
            1: "validation_utility_passed",
            _GATE_RULE_VERSION: "per_label_utility_passed",
        }.get(version if isinstance(version, int) and not isinstance(version, bool) else -1)
        if (
            not isinstance(gate, dict)
            or utility_key is None
            or gate.get("passed") is not True
            or gate.get("thresholds_feasible") is not True
            or gate.get(utility_key) is not True
        ):
            raise ValueError("Model has not passed the recorded development gates")
    if "amendment" in manifest and not amended:
        raise ValueError("Amendment recorded without the ratified gate rule")
    hashes = manifest["artifact_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing artifact hashes")
    required = {"config.json", "tokenizer_config.json", "thresholds.json"}
    if not required <= hashes.keys() or not any(name.endswith(".safetensors") for name in hashes):
        raise ValueError("Missing model artifacts")
    if not {"tokenizer.json", "spm.model"} & hashes.keys():
        raise ValueError("Missing tokenizer data")
    for name, expected in hashes.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("Invalid artifact hash entry")
        path = model_dir / name
        if path.is_symlink() or not hmac.compare_digest(sha256_file(path), expected):
            raise ValueError("Artifact integrity failure")
    # Hugging Face can select among several tokenizer/model files. Every file
    # it may read must be covered, including optional or sharded assets.
    for path in model_dir.iterdir():
        if path.name == "manifest.json":
            continue
        if path.suffix in (".json", ".safetensors", ".model", ".txt") and path.name not in hashes:
            raise ValueError("Unverified model or tokenizer asset")
    selection = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    if selection != manifest["thresholds"]:
        raise ValueError("Threshold file differs from manifest")
    if selection["labels"] != list(LABELS) or selection["selected_on"] != "validation":
        raise ValueError("Invalid threshold labels or selection split")
    thresholds = selection["thresholds"]
    if (
        not isinstance(thresholds, list)
        or len(thresholds) != len(LABELS)
        or set(selection["per_label"]) != set(LABELS)
    ):
        raise ValueError("Invalid threshold dimensions")
    for label, threshold in zip(LABELS, thresholds, strict=True):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            or selection["per_label"][label]["threshold"] != threshold
            or selection["per_label"][label]["feasible"] is not True
        ):
            raise ValueError("Invalid or unsupported threshold")
    removal = _removal_labels(manifest, selection) if amended else frozenset(LABELS)
    max_length = manifest["hyperparameters"]["max_length"]
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 16:
        raise ValueError("Invalid token budget")
    return manifest, tuple(
        float(value) if label in removal else None
        for label, value in zip(LABELS, thresholds, strict=True)
    )


def _load_runtime(model_dir: Path, device: str) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True
    ).to(device=device, dtype=torch.float32)
    if model.config.id2label != dict(enumerate(LABELS)) or model.config.label2id != {
        label: index for index, label in enumerate(LABELS)
    }:
        raise ValueError("Incompatible model label order")
    model.eval()
    return torch, tokenizer, model


def _checked_windows(text: str) -> tuple[list[tuple[int, int, int, int]], int]:
    candidates = windows(text)
    sentences = set(sentence_spans(text))
    coverage = []
    represented = set()
    for start, end, sentence_start, sentence_end in candidates:
        if (
            not 0 <= sentence_start <= start < end <= sentence_end <= len(text)
            or (sentence_start, sentence_end) not in sentences
            or not text[start:end].strip()
        ):
            raise ValueError("Invalid target bounds")
        represented.add((sentence_start, sentence_end))
        coverage.append((start, end))
    if represented != sentences:
        raise ValueError("Uncovered sentence")
    cursor = 0
    for start, end in sorted(coverage):
        if start > cursor and text[cursor:start].strip():
            raise ValueError("Uncovered document text")
        cursor = max(cursor, end)
    if text[cursor:].strip():
        raise ValueError("Uncovered document tail")
    return candidates, len(sentences)


def subdivide_window(
    text: str,
    tokenizer: Any,
    max_length: int,
    *,
    start: int,
    end: int,
    sentence_start: int,
    sentence_end: int,
) -> list[tuple[int, int, int, int]]:
    """Fit one source window without truncating targets or changing sentence bounds.

    Pass the original document text, not a normalized/stripped frame. Returned
    offsets index that same document. Fitting windows remain unchanged; oversized
    targets are bisected with up to 32 characters of overlap on each side.
    Context is still selected by _encode when each child is scored. This helper
    accepts/returns no labels: derivative annotation frames need fresh labels.
    """
    try:
        if (
            not isinstance(text, str)
            or type(max_length) is not int
            or max_length < 1
            or any(type(value) is not int for value in (start, end, sentence_start, sentence_end))
            or not 0 <= sentence_start <= start < end <= sentence_end <= len(text)
            or not text[start:end].strip()
        ):
            raise ValueError("Invalid adaptive window bounds")
        special = tokenizer.num_special_tokens_to_add(pair=False)
        if type(special) is not int or special < 0:
            raise ValueError("Invalid special-token accounting")
        maximum_depth = 4 * (end - start).bit_length()
        pending = [(start, end, 0)]
        fitted: set[tuple[int, int, int, int]] = set()
        while pending:
            left, right, depth = pending.pop()
            original = text[left:right]
            marked = f"[TARGET]{original.strip()}[/TARGET]"
            target_text(marked)
            count = len(tokenizer.encode(marked, add_special_tokens=False)) + special
            if count <= max_length:
                fitted.add((left, right, sentence_start, sentence_end))
                continue
            active_left = left + len(original) - len(original.lstrip())
            active_right = right - (len(original) - len(original.rstrip()))
            width = active_right - active_left
            if width <= 1 or depth >= maximum_depth:
                raise ValueError("An indivisible target cannot fit the token budget")
            middle = (active_left + active_right) // 2
            overlap = min(32, width // 8)
            left_end = min(active_right - 1, middle + overlap)
            right_start = max(active_left + 1, middle - overlap)
            # Both children lose an active boundary character, cover their
            # parent's entire interval, and shrink geometrically even when
            # whitespace padding is much larger than the active target.
            if not left < right_start <= left_end < right:
                raise ValueError("Adaptive subdivision did not make progress")
            pending.extend(((right_start, right, depth + 1), (left, left_end, depth + 1)))
        return sorted(fitted)
    except Exception:
        raise ValueError("Adaptive target subdivision failed") from None


def _merge_spans(spans: Sequence[RedactionSpan]) -> tuple[RedactionSpan, ...]:
    merged: list[RedactionSpan] = []
    for span in sorted(spans, key=lambda value: (value.start, value.end)):
        if merged and span.start <= merged[-1].end:
            previous = merged.pop()
            labels = set(previous.labels) | set(span.labels)
            merged.append(
                RedactionSpan(
                    start=previous.start,
                    end=max(previous.end, span.end),
                    labels=tuple(label for label in LABELS if label in labels),
                )
            )
        else:
            merged.append(span)
    return tuple(merged)


class LocalRedactor:
    """Load one verified local model, then reuse it across document requests."""

    def __init__(
        self, model_dir: Path, *, device: Literal["cpu", "mps"] = "cpu", batch_size: int = 8
    ) -> None:
        try:
            if device not in ("cpu", "mps") or isinstance(batch_size, bool) or batch_size < 1:
                raise ValueError("Invalid runtime settings")
            model_dir = Path(model_dir)
            manifest_path = model_dir / "manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest, self.thresholds = _verified_artifact(model_dir)
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("Model manifest changed during verification")
            self.max_length: int = manifest["hyperparameters"]["max_length"]
            self.device = device
            self.batch_size = batch_size
            self._torch, self._tokenizer, self._model = _load_runtime(model_dir, device)
            # Loading reads assets separately; reject replacement during startup,
            # including a different, internally valid model or threshold set.
            if (
                _verified_artifact(model_dir) != (manifest, self.thresholds)
                or manifest_path.read_bytes() != manifest_bytes
            ):
                raise ValueError("Model artifacts changed during loading")
            self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._lock = threading.Lock()
        except Exception:
            raise RedactionError("Local redactor initialization failed") from None

    def _score(self, examples: Sequence[str]) -> list[list[float]]:
        encoded = [_encode(self._tokenizer, example, self.max_length) for example in examples]
        if any(not row["input_ids"] or len(row["input_ids"]) > self.max_length for row in encoded):
            raise ValueError("Invalid encoded target")
        with self._torch.inference_mode():
            logits = self._model(**_batch(self._tokenizer, encoded, self.device)).logits
            if tuple(logits.shape) != (len(examples), len(LABELS)):
                raise ValueError("Invalid prediction dimensions")
            if not bool(self._torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite model output")
            scores: list[list[float]] = logits.sigmoid().cpu().tolist()
        return scores

    def redact(self, text: str) -> RedactionResult:
        """Scan every target, then replace flagged sentences with [omissis]."""
        try:
            if not isinstance(text, str):
                raise ValueError("Text must be a string")
            candidates, sentence_count = _checked_windows(text)
            detected: dict[tuple[int, int], set[str]] = {}
            # Bound concurrent GPU/CPU memory per persistent instance. There is
            # no output until all batches have been scored and checked.
            with self._lock:
                candidates = [
                    child
                    for start, end, sentence_start, sentence_end in candidates
                    for child in subdivide_window(
                        text,
                        self._tokenizer,
                        self.max_length,
                        start=start,
                        end=end,
                        sentence_start=sentence_start,
                        sentence_end=sentence_end,
                    )
                ]
                for offset in range(0, len(candidates), self.batch_size):
                    batch = candidates[offset : offset + self.batch_size]
                    examples = [example_text(text, start, end) for start, end, _, _ in batch]
                    predictions = self._score(examples)
                    if len(predictions) != len(batch):
                        raise ValueError("Missing predictions")
                    for (_, _, start, end), scores in zip(batch, predictions, strict=True):
                        if len(scores) != len(LABELS) or any(
                            isinstance(score, bool)
                            or not isinstance(score, (int, float))
                            or not math.isfinite(score)
                            or not 0 <= score <= 1
                            for score in scores
                        ):
                            raise ValueError("Invalid probabilities")
                        labels = {
                            label
                            for label, score, threshold in zip(
                                LABELS, scores, self.thresholds, strict=True
                            )
                            if threshold is not None and score >= threshold
                        }
                        if labels:
                            detected.setdefault((start, end), set()).update(labels)
            counts = {
                label: sum(label in labels for labels in detected.values()) for label in LABELS
            }
            spans = []
            for (start, end), labels in detected.items():
                # Sentence splitting includes separator whitespace. Keep those
                # separators byte-for-byte around the replacement marker.
                original = text[start:end]
                trimmed_start = start + len(original) - len(original.lstrip())
                trimmed_end = end - (len(original) - len(original.rstrip()))
                spans.append(
                    RedactionSpan(
                        start=trimmed_start,
                        end=trimmed_end,
                        labels=tuple(label for label in LABELS if label in labels),
                    )
                )
            merged = _merge_spans(spans)
            pieces: list[str] = []
            cursor = 0
            for span in merged:
                pieces.extend((text[cursor : span.start], "[omissis]"))
                cursor = span.end
            pieces.append(text[cursor:])
            return RedactionResult(
                text="".join(pieces),
                counts=counts,
                spans=merged,
                scanned_windows=len(candidates),
                scanned_sentences=sentence_count,
            )
        except Exception:
            raise RedactionError("Local redaction failed; document must not be forwarded") from None
