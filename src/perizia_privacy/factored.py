"""Factored sentence student (phase 0): wider input, policy factors as heads.

v1 sees 200 characters around a window, reads ``[TARGET]`` as plain text and
pools [CLS] only, while its teacher labels with two sentences on each side.
This student closes that gap:

* input: up to two whole sentences on each side (at most ``SIDE_TOKENS`` each),
  ``[TARGET]``/``[/TARGET]`` as real special tokens, and person mentions
  (``[nome]``, omissis, esecutato/debitore, sig./sig.ra, titled names) wrapped
  in ``[PERSONA]``/``[/PERSONA]``. The person marks are additive features, never
  a filter: every target is scored whether or not it mentions anyone;
* pooling: the [CLS] state concatenated with the ``[TARGET]`` token state;
* outputs: one REMOVE head per category is the decision; topic per category,
  subject (6-way) and assertion (3-way) are auxiliary heads, loss-masked
  wherever a row does not know them.

:func:`document_targets` and :func:`encode_target` are the single encode path
for training, evaluation and :class:`FactoredRedactor`, so what is measured is
what would run. v1 (:mod:`.redactor`) and v2 are untouched; this is a separate
model version.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .corpus import sentence_spans, windows
from .redactor import (
    RedactionError,
    RedactionResult,
    RedactionSpan,
    _merge_spans,
    subdivide_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

STUDENT = "factored-v1"
HEADS = ("salute", "penale", "minori", "famiglia")
SUBJECTS = ("living", "masked_party", "deceased", "company", "hypothetical", "none")
ASSERTIONS = ("affirmed", "negated", "hypothetical")
#: Output layout: remove[0:4] | topic[4:8] | subject[8:14] | assertion[14:17].
REMOVE = slice(0, 4)
TOPIC = slice(4, 8)
SUBJECT = slice(8, 14)
ASSERTION = slice(14, 17)
OUTPUTS = 17

TARGET_OPEN, TARGET_CLOSE = "[TARGET]", "[/TARGET]"
PERSON_OPEN, PERSON_CLOSE = "[PERSONA]", "[/PERSONA]"
SPECIAL_TOKENS = (TARGET_OPEN, TARGET_CLOSE, PERSON_OPEN, PERSON_CLOSE)
CONTEXT_SENTENCES = 2
SIDE_TOKENS = 150
MAX_LENGTH = 512
#: [CLS] + [TARGET] + target + [/TARGET] + [SEP] fit beside two full sides.
TARGET_TOKENS = MAX_LENGTH - 4 - 2 * SIDE_TOKENS
#: Inference subdivides targets below the cap; person marks need headroom.
SUBDIVIDE_TOKENS = TARGET_TOKENS - 48

_NAME = r"(?:\s+[A-ZÀ-Ý][\w'’]+){1,2}"
PERSON = re.compile(
    r"\[nome\]"
    r"|(?i:\bomissis\b|\besecutat[aeio]\b|\bdebitor[ei]\b|\bdebitrice\b)"
    r"|(?i:\bsig(?:\.ra\b|\.|nor[ae]?\b))" + f"(?:{_NAME})?"
    r"|(?i:\b(?:dott|avv|ing|geom|arch|rag|prof)(?:\.ssa\b|ssa\.|\.))" + _NAME
)


def mark_persons(text: str) -> str:
    """Normalise whitespace, drop stray marker strings, wrap person mentions."""
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    text = " ".join(text.split())
    return PERSON.sub(lambda m: f"{PERSON_OPEN}{m.group()}{PERSON_CLOSE}", text)


def add_special_tokens(tokenizer: Any) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})


def encode_target(
    tokenizer: Any,
    text: str,
    spans: Sequence[tuple[int, int]],
    index: int,
    start: int,
    end: int,
    *,
    strict: bool = False,
) -> tuple[list[int], int]:
    """Token ids and the position of ``[TARGET]`` for one target in its sentence.

    Context is the rest of the target's own sentence plus ``CONTEXT_SENTENCES``
    whole sentences each side, truncated to ``SIDE_TOKENS`` away from the
    target. ``strict`` refuses a target longer than ``TARGET_TOKENS`` instead of
    truncating it, so inference never scores a partial target.
    """
    sentence_start, sentence_end = spans[index]
    if not sentence_start <= start < end <= sentence_end:
        raise ValueError("Target outside its sentence")
    first = spans[max(0, index - CONTEXT_SENTENCES)][0]
    last = spans[min(len(spans) - 1, index + CONTEXT_SENTENCES)][1]
    return encode_pieces(
        tokenizer, text[first:start], text[start:end], text[end:last], strict=strict
    )


def encode_pieces(
    tokenizer: Any, before: str, target: str, after: str, *, strict: bool = False
) -> tuple[list[int], int]:
    """The same encoding from already-cut context, for rows that carry no document."""

    def ids(piece: str) -> list[int]:
        return tokenizer.encode(mark_persons(piece), add_special_tokens=False)

    target_ids = ids(target)
    if len(target_ids) > TARGET_TOKENS:
        if strict:
            raise ValueError("Target exceeds the token budget")
        target_ids = target_ids[:TARGET_TOKENS]
    left = ids(before)[-SIDE_TOKENS:]
    right = ids(after)[:SIDE_TOKENS]
    open_id, close_id = tokenizer.convert_tokens_to_ids([TARGET_OPEN, TARGET_CLOSE])
    return (
        [tokenizer.cls_token_id, *left, open_id, *target_ids, close_id, *right]
        + [tokenizer.sep_token_id],
        1 + len(left),
    )


def document_targets(text: str, tokenizer: Any) -> list[tuple[int, int, int]]:
    """(start, end, sentence index) for every target, covering every sentence.

    Uses v1's windows and its token-fitting subdivision, so coverage is the
    same as the local redactor's.
    """
    spans = sentence_spans(text)
    index = {span: i for i, span in enumerate(spans)}
    out = []
    for start, end, sentence_start, sentence_end in windows(text):
        for child in subdivide_window(
            text,
            tokenizer,
            SUBDIVIDE_TOKENS,
            start=start,
            end=end,
            sentence_start=sentence_start,
            sentence_end=sentence_end,
        ):
            out.append((child[0], child[1], index[(sentence_start, sentence_end)]))
    if {i for _, _, i in out} != set(range(len(spans))):
        raise ValueError("Uncovered sentence")
    return out


def build_model(torch: Any, encoder: Any) -> Any:
    """Wrap an encoder with the pooled multi-head output layer."""

    class FactoredStudent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(0.1)
            self.heads = torch.nn.Linear(2 * encoder.config.hidden_size, OUTPUTS)

        def forward(self, input_ids: Any, attention_mask: Any, target_pos: Any) -> Any:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = torch.cat([hidden[:, 0], hidden[rows, target_pos]], dim=-1)
            return self.heads(self.dropout(pooled))

    return FactoredStudent()


def batch_tensors(
    torch: Any, rows: Sequence[tuple[list[int], int]], pad_id: int, device: str
) -> dict[str, Any]:
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for j, (ids, _) in enumerate(rows):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        attention[j, : len(ids)] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "target_pos": torch.tensor([pos for _, pos in rows], device=device),
    }


def probabilities(torch: Any, logits: Any) -> Any:
    """Sigmoid for remove/topic, softmax for subject and assertion."""
    return torch.cat(
        [
            logits[:, :8].sigmoid(),
            logits[:, SUBJECT].softmax(-1),
            logits[:, ASSERTION].softmax(-1),
        ],
        dim=-1,
    )


def load(model_dir: Path, device: str) -> tuple[Any, Any, Any]:
    """torch, tokenizer and the eval-mode student from a saved directory."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    safetensors = importlib.import_module("safetensors.torch")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = build_model(torch, transformers.AutoModel.from_config(config))
    model.load_state_dict(safetensors.load_file(str(model_dir / "model.safetensors")))
    return torch, tokenizer, model.to(device=device, dtype=torch.float32).eval()


def save(model: Any, tokenizer: Any, out_dir: Path, manifest: dict[str, Any]) -> None:
    """Weights, encoder config, tokenizer and a manifest hashing every file."""
    safetensors = importlib.import_module("safetensors.torch")
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    safetensors.save_file(state, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    hashes = {
        p.name: _sha256(p)
        for p in sorted(out_dir.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = manifest | {"student": STUDENT, "heads": list(HEADS), "artifact_sha256": hashes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_targets(
    torch: Any,
    tokenizer: Any,
    model: Any,
    text: str,
    targets: Sequence[tuple[int, int, int]],
    device: str,
    batch_size: int = 16,
    *,
    strict: bool = False,
) -> list[list[float]]:
    """Probabilities (``OUTPUTS`` per target) in target order."""
    spans = sentence_spans(text)
    encoded = [
        encode_target(tokenizer, text, spans, i, start, end, strict=strict)
        for start, end, i in targets
    ]
    order = sorted(range(len(encoded)), key=lambda k: len(encoded[k][0]))
    out: list[list[float]] = [[] for _ in encoded]
    with torch.inference_mode():
        for k in range(0, len(order), batch_size):
            idx = order[k : k + batch_size]
            logits = model(
                **batch_tensors(torch, [encoded[i] for i in idx], tokenizer.pad_token_id, device)
            )
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite model output")
            for i, row in zip(idx, probabilities(torch, logits).cpu().tolist(), strict=True):
                out[i] = row
    return out


def verified_artifact(model_dir: Path) -> tuple[dict[str, Any], tuple[float | None, ...]]:
    """Manifest and the four remove thresholds, or a refusal."""
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("student") != STUDENT or manifest.get("heads") != list(HEADS):
        raise ValueError("Manifest does not describe a factored student")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or "model.safetensors" not in hashes:
        raise ValueError("Manifest records no weights")
    for name, digest in hashes.items():
        path = model_dir / Path(name).name
        if path.is_symlink() or _sha256(path) != digest:
            raise ValueError("Model artifact changed since training")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, list) or len(thresholds) != len(HEADS):
        raise ValueError("Invalid threshold dimensions")
    for value in thresholds:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Invalid threshold")
    if all(value is None for value in thresholds):
        raise ValueError("No head is retained for removal")
    return manifest, tuple(None if v is None else float(v) for v in thresholds)


class FactoredRedactor:
    """One verified factored student, reused across documents; fails closed."""

    def __init__(
        self, model_dir: Path, *, device: Literal["cpu", "mps"] = "cpu", batch_size: int = 8
    ) -> None:
        try:
            if device not in ("cpu", "mps") or isinstance(batch_size, bool) or batch_size < 1:
                raise ValueError("Invalid runtime settings")
            model_dir = Path(model_dir)
            manifest, self.thresholds = verified_artifact(model_dir)
            self.device, self.batch_size = device, batch_size
            self._torch, self._tokenizer, self._model = load(model_dir, device)
            if verified_artifact(model_dir) != (manifest, self.thresholds):
                raise ValueError("Model artifacts changed during loading")
            self._lock = threading.Lock()
        except Exception:
            raise RedactionError("Local factored redactor initialization failed") from None

    def redact(self, text: str) -> RedactionResult:
        """Score every target, then replace flagged sentences with [omissis]."""
        try:
            if not isinstance(text, str):
                raise ValueError("Text must be a string")
            spans = sentence_spans(text)
            with self._lock:
                targets = document_targets(text, self._tokenizer)
                scores = score_targets(
                    self._torch,
                    self._tokenizer,
                    self._model,
                    text,
                    targets,
                    self.device,
                    self.batch_size,
                    strict=True,
                )
            detected: dict[int, set[str]] = {}
            for (_, _, i), row in zip(targets, scores, strict=True):
                labels = {
                    head
                    for head, score, threshold in zip(
                        HEADS, row[REMOVE], self.thresholds, strict=True
                    )
                    if threshold is not None and score >= threshold
                }
                if labels:
                    detected.setdefault(i, set()).update(labels)
            counts = {head: sum(head in v for v in detected.values()) for head in HEADS}
            pieces_spans = []
            for i, labels in detected.items():
                start, end = spans[i]
                original = text[start:end]
                pieces_spans.append(
                    RedactionSpan(
                        start=start + len(original) - len(original.lstrip()),
                        end=end - (len(original) - len(original.rstrip())),
                        labels=tuple(head for head in HEADS if head in labels),
                    )
                )
            merged = _merge_spans(pieces_spans)
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
                scanned_windows=len(targets),
                scanned_sentences=len(spans),
            )
        except Exception:
            raise RedactionError(
                "Local factored redaction failed; document must not be forwarded"
            ) from None
