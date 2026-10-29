"""Evaluate sentence redaction without exposing document text.

Clopper-Pearson bounds describe binomial counts, not independence of sentences.
Use document-cluster bootstrap intervals alongside them for repeated sentences.
Label provenance is supplied by the caller; model agreement is never called gold.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

LABELS = ("salute", "penale", "minori", "famiglia", "altre_categorie")


class LabelMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tp: int
    fn: int
    fp: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    recall_ci_lower: float | None
    recall_ci_upper: float | None
    # Older saved calibration strata did not calculate precision intervals.
    precision_ci_lower: float | None = None
    precision_ci_upper: float | None = None


class DocumentLeakage(BaseModel):
    """Removal counts over supplied originals, irrespective of clustering.

    Unnecessary removal means removing an all-label-negative sentence. Its
    all-document rate uses ``documents``; its conditional rate uses
    ``documents_with_nonsensitive_sentences``. Absent legacy fields stay unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: int
    sensitive_documents: int
    documents_with_leak: int
    any_leak_rate: float | None
    sensitive_document_leak_rate: float | None
    sensitive_sentences: int
    leaked_sensitive_sentences: int
    nonsensitive_sentences: int
    redacted_nonsensitive_sentences: int
    documents_with_nonsensitive_sentences: int | None = None
    documents_with_unnecessary_removal: int | None = None
    unnecessary_removal_rate_all_documents: float | None = None
    unnecessary_removal_rate_nonsensitive_documents: float | None = None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = LABELS
    label_provenance: str
    sentences: int
    per_label: dict[str, LabelMetrics]
    document_leakage: DocumentLeakage
    interval_method: str = "two-sided 95% Clopper-Pearson; assumes independent trials"


class ThresholdResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float
    feasible: bool
    reason: str
    support: int
    precision: float | None
    recall: float | None
    recall_ci_lower: float | None
    recall_ci_upper: float | None


class ThresholdSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = LABELS
    thresholds: list[float]
    per_label: dict[str, ThresholdResult]
    selected_on: Literal["validation"] = "validation"
    label_provenance: str
    requested_recall: float
    constraint_basis: str


class CalibrationStratum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Literal["validation", "calibration"]
    source: Literal["real", "synthetic"]
    label_provenance: str
    rows: int
    positive_support: int
    metrics: LabelMetrics


class CalibratedThresholdResult(ThresholdResult):
    # Inherited support and metrics always describe REAL validation. Synthetic
    # support never fills an absent real denominator.
    calibration_origin: Literal[
        "real_validation_only",
        "real_validation_and_synthetic_calibration",
        "synthetic_calibration_only",
    ]
    real_validation: CalibrationStratum
    synthetic_calibration: CalibrationStratum | None
    precision_objective: Literal["real_validation", "combined_machine_reference"]
    combined_precision: float | None


class CalibrationThresholdSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = LABELS
    thresholds: list[float]
    per_label: dict[str, CalibratedThresholdResult]
    selected_on: Literal["validation"] = "validation"
    label_provenance: str
    calibration_label_provenance: str
    requested_recall: float
    constraint_basis: str = "observed_recall_separately_on_each_positive_stratum"
    fallback_policy: str = (
        "Below 20 real union positives, require at least 100 separate fictional union positives; "
        "require recall >= requested_recall independently on every positive stratum; "
        "maximize combined precision, then higher threshold. At least 20 real positives uses "
        "real validation only. Synthetic evidence is neither real support nor human gold."
    )
    real_positive_cutoff: int = 20
    synthetic_positive_minimum: int = 100


class ValidationUtilityGate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_on: Literal["validation"] = "validation"
    source: Literal["real"] = "real"
    label_provenance: str
    all_negative_support: int
    unnecessarily_removed: int
    false_positive_rate: float | None
    maximum_false_positive_rate: float = 0.05
    passed: bool
    reason: str
    interpretation: str = (
        "OR-removal among real all-negative union-reference rows; observed utility on the "
        "supplied validation mixture, not production prevalence or independently verified truth"
    )


class BootstrapInterval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float | None
    upper: float | None
    valid_resamples: int


class BootstrapReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_provenance: str
    seed: int
    resamples: int
    # Retain the existing field and constructor input for older consumers.
    documents: int
    original_documents: int = Field(
        validation_alias=AliasChoices("original_documents", "documents")
    )
    clusters: int | None = None
    resampling_unit: Literal["original_document", "document_cluster"] = "original_document"
    recall: dict[str, BootstrapInterval]
    document_any_leak_rate: BootstrapInterval
    interval_method: str = (
        "95% percentile bootstrap resampling whole documents; documents assumed independent; "
        "degenerate intervals cannot rule out unobserved errors"
    )


def _binary_matrix(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(LABELS):
        raise ValueError(f"{name} must have shape (sentences, {len(LABELS)})")
    if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(np.bool_)


def _aligned_predictions(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    truth = _binary_matrix(y_true, "y_true")
    predictions = _binary_matrix(y_pred, "y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return truth, predictions


def _document_groups(
    document_ids: Sequence[str], rows: int, *, name: str = "document_ids"
) -> tuple[NDArray[np.int64], int]:
    if isinstance(document_ids, (str, bytes, Mapping, set, frozenset)) or len(document_ids) != rows:
        raise ValueError(f"{name} must contain one nonempty string per sentence")
    groups: dict[str, int] = {}
    codes: list[int] = []
    for document_id in document_ids:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"{name} must contain one nonempty string per sentence")
        codes.append(groups.setdefault(document_id, len(groups)))
    return np.asarray(codes, dtype=np.int64), len(groups)


def _provenance(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("label_provenance must explicitly describe the label source")


def clopper_pearson(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    """Return a two-sided exact binomial interval; zero trials have no interval."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, (int, np.integer))
        or not isinstance(trials, (int, np.integer))
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if trials == 0:
        return None, None
    # scipy belongs to the research/ML extra; runtime redaction does not import it.
    from scipy.stats import beta

    alpha = (1 - confidence) / 2
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha, successes + 1, trials - successes))
    )
    return lower, upper


def _metrics(tp: int, fn: int, fp: int, tn: int) -> LabelMetrics:
    lower, upper = clopper_pearson(tp, tp + fn)
    precision_lower, precision_upper = clopper_pearson(tp, tp + fp)
    return LabelMetrics(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        recall_ci_lower=lower,
        recall_ci_upper=upper,
        precision_ci_lower=precision_lower,
        precision_ci_upper=precision_upper,
    )


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    document_ids: Sequence[str],
    *,
    label_provenance: str,
) -> EvaluationReport:
    """Measure classification and whole-sentence removal using frozen predictions.

    A sentence flagged under any category is removed in full: a wrong category
    is a classification miss but does not leave that sentence's text exposed.
    Rates use the supplied sample, whose prevalence may differ from the archive.
    No model or threshold selection takes place here, including on final test.
    """
    _provenance(label_provenance)
    truth, predictions = _aligned_predictions(y_true, y_pred)
    groups, documents = _document_groups(document_ids, len(truth))
    per_label = {}
    for column, label in enumerate(LABELS):
        actual, predicted = truth[:, column], predictions[:, column]
        per_label[label] = _metrics(
            int(np.sum(actual & predicted)),
            int(np.sum(actual & ~predicted)),
            int(np.sum(~actual & predicted)),
            int(np.sum(~actual & ~predicted)),
        )
    sensitive = truth.any(axis=1)
    redacted = predictions.any(axis=1)
    leaked = sensitive & ~redacted
    unnecessary = ~sensitive & redacted
    sensitive_documents = int(len(np.unique(groups[sensitive])))
    leaked_documents = int(len(np.unique(groups[leaked])))
    nonsensitive_documents = int(len(np.unique(groups[~sensitive])))
    unnecessary_documents = int(len(np.unique(groups[unnecessary])))
    return EvaluationReport(
        label_provenance=label_provenance,
        sentences=len(truth),
        per_label=per_label,
        document_leakage=DocumentLeakage(
            documents=documents,
            sensitive_documents=sensitive_documents,
            documents_with_leak=leaked_documents,
            any_leak_rate=leaked_documents / documents if documents else None,
            sensitive_document_leak_rate=(
                leaked_documents / sensitive_documents if sensitive_documents else None
            ),
            sensitive_sentences=int(sensitive.sum()),
            leaked_sensitive_sentences=int(leaked.sum()),
            nonsensitive_sentences=int((~sensitive).sum()),
            redacted_nonsensitive_sentences=int(unnecessary.sum()),
            documents_with_nonsensitive_sentences=nonsensitive_documents,
            documents_with_unnecessary_removal=unnecessary_documents,
            unnecessary_removal_rate_all_documents=(
                unnecessary_documents / documents if documents else None
            ),
            unnecessary_removal_rate_nonsensitive_documents=(
                unnecessary_documents / nonsensitive_documents if nonsensitive_documents else None
            ),
        ),
    )


def select_validation_thresholds(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    split: Literal["validation"],
    label_provenance: str,
    requested_recall: float = 0.98,
    use_recall_lower_bound: bool = False,
) -> ThresholdSelection:
    """Maximize validation precision subject to the requested recall constraint.

    Ties prefer higher recall, then a higher threshold. If infeasible, prefer
    maximum recall, then precision, then threshold, and report the failure.
    An absent positive category uses 0.5 and is explicitly infeasible. These
    observed validation constraints do not guarantee future or final-test recall.
    The caller must provide genuinely held-out validation rows; the split guard
    rejects accidental test use but cannot establish provenance independently.
    """
    if split != "validation":
        raise ValueError(
            "Threshold selection requires validation data; test selection is forbidden"
        )
    _provenance(label_provenance)
    if not 0 < requested_recall <= 1:
        raise ValueError("requested_recall must be greater than zero and at most one")
    truth = _binary_matrix(y_true, "y_true")
    scores = np.asarray(probabilities)
    if scores.shape != truth.shape:
        raise ValueError("probabilities and y_true must have identical shapes")
    if scores.dtype.kind not in "iuf" or not np.all(
        np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    ):
        raise ValueError("probabilities must contain finite numeric values in [0, 1]")
    selections: dict[str, ThresholdResult] = {}
    for column, label in enumerate(LABELS):
        actual = truth[:, column]
        support = int(actual.sum())
        if not support:
            metrics = _metrics(
                0, 0, int(np.sum(scores[:, column] >= 0.5)), int(np.sum(scores[:, column] < 0.5))
            )
            selections[label] = ThresholdResult(
                threshold=0.5,
                feasible=False,
                reason="no_validation_positives",
                support=0,
                **metrics.model_dump(
                    include={"precision", "recall", "recall_ci_lower", "recall_ci_upper"}
                ),
            )
            continue
        order = np.argsort(-scores[:, column], kind="stable")
        sorted_scores = scores[order, column]
        cumulative_tp = np.cumsum(actual[order])
        # Only the end of each tied score represents a realizable >= threshold.
        ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
        candidates: list[tuple[float, LabelMetrics, float]] = []
        for end in ends:
            tp = int(cumulative_tp[end])
            fp = int(end + 1 - tp)
            metrics = _metrics(tp, support - tp, fp, len(truth) - support - fp)
            constraint = metrics.recall_ci_lower if use_recall_lower_bound else metrics.recall
            assert constraint is not None
            candidates.append((float(sorted_scores[end]), metrics, constraint))
        feasible = [candidate for candidate in candidates if candidate[2] >= requested_recall]
        if feasible:
            chosen = max(
                feasible, key=lambda item: (item[1].precision or 0, item[1].recall or 0, item[0])
            )
        else:
            chosen = max(candidates, key=lambda item: (item[2], item[1].precision or 0, item[0]))
        threshold, metrics, _ = chosen
        selections[label] = ThresholdResult(
            threshold=threshold,
            feasible=bool(feasible),
            reason="validation_constraint_met"
            if feasible
            else "validation_recall_constraint_infeasible",
            support=support,
            **metrics.model_dump(
                include={"precision", "recall", "recall_ci_lower", "recall_ci_upper"}
            ),
        )
    return ThresholdSelection(
        thresholds=[selections[label].threshold for label in LABELS],
        per_label=selections,
        label_provenance=label_provenance,
        requested_recall=requested_recall,
        constraint_basis=(
            "validation_recall_cp_lower_bound"
            if use_recall_lower_bound
            else "observed_validation_recall"
        ),
    )


def _probability_matrix(values: ArrayLike, truth: NDArray[np.bool_]) -> NDArray[np.float64]:
    scores = np.asarray(values)
    if (
        scores.shape != truth.shape
        or scores.dtype.kind not in "iuf"
        or not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1))
    ):
        raise ValueError("probabilities must be aligned finite numeric values in [0, 1]")
    return scores.astype(np.float64)


def _stratum(
    actual: NDArray[np.bool_],
    scores: NDArray[np.float64],
    threshold: float,
    *,
    split: Literal["validation", "calibration"],
    provenance: str,
) -> CalibrationStratum:
    predicted = scores >= threshold
    return CalibrationStratum(
        split=split,
        source="real" if split == "validation" else "synthetic",
        label_provenance=provenance,
        rows=len(actual),
        positive_support=int(actual.sum()),
        metrics=_metrics(
            int((actual & predicted).sum()),
            int((actual & ~predicted).sum()),
            int((~actual & predicted).sum()),
            int((~actual & ~predicted).sum()),
        ),
    )


def _stratified_threshold(
    real_truth: NDArray[np.bool_],
    real_scores: NDArray[np.float64],
    synthetic_truth: NDArray[np.bool_],
    synthetic_scores: NDArray[np.float64],
    requested_recall: float,
) -> float | None:
    """Enumerate realizable tied-score boundaries in O(n log n), not O(n²)."""
    scores = np.r_[real_scores, synthetic_scores]
    is_real = np.arange(len(scores)) < len(real_scores)
    actual = np.r_[real_truth, synthetic_truth]
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    real_tp = np.cumsum((actual & is_real)[order])
    synthetic_tp = np.cumsum((actual & ~is_real)[order])
    real_support, synthetic_support = int(real_truth.sum()), int(synthetic_truth.sum())
    best: tuple[float, float] | None = None
    ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    for end in ends:
        if real_support and real_tp[end] / real_support < requested_recall:
            continue
        if synthetic_support and synthetic_tp[end] / synthetic_support < requested_recall:
            continue
        precision = float((real_tp[end] + synthetic_tp[end]) / (end + 1))
        candidate = (precision, float(ordered_scores[end]))
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[1]


def select_calibration_thresholds(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    calibration_y_true: ArrayLike,
    calibration_probabilities: ArrayLike,
    *,
    split: Literal["validation"],
    calibration_split: Literal["calibration"],
    label_provenance: str,
    calibration_label_provenance: str,
    requested_recall: float = 0.98,
) -> CalibrationThresholdSelection:
    """Apply the preregistered scarce-category fallback without pooling recalls.

    The caller must supply disjoint real validation and fictional calibration
    union references. Synthetic scores never select an epoch. Scalar metrics
    inherited by each threshold result remain real-only; the combined precision
    used for sparse-category selection is an explicitly separate quantity.
    """
    if split != "validation" or calibration_split != "calibration":
        raise ValueError("Calibration requires separate validation and calibration splits")
    _provenance(label_provenance)
    _provenance(calibration_label_provenance)
    if not 0.98 <= requested_recall <= 1:
        raise ValueError("Separate calibration requires requested_recall between 0.98 and one")
    truth = _binary_matrix(y_true, "y_true")
    synthetic = _binary_matrix(calibration_y_true, "calibration_y_true")
    if not len(truth) or not len(synthetic):
        raise ValueError("Real validation and synthetic calibration must both be nonempty")
    scores = _probability_matrix(probabilities, truth)
    synthetic_scores = _probability_matrix(calibration_probabilities, synthetic)
    baseline = select_validation_thresholds(
        truth,
        scores,
        split="validation",
        label_provenance=label_provenance,
        requested_recall=requested_recall,
    )
    selections: dict[str, CalibratedThresholdResult] = {}
    for column, label in enumerate(LABELS):
        real_support, synthetic_support = (
            int(truth[:, column].sum()),
            int(synthetic[:, column].sum()),
        )
        original = baseline.per_label[label]
        sparse = real_support < 20
        threshold, feasible, reason = original.threshold, original.feasible, original.reason
        if sparse:
            if synthetic_support < 100:
                feasible, reason = False, "insufficient_synthetic_union_positives"
            else:
                chosen = _stratified_threshold(
                    truth[:, column],
                    scores[:, column],
                    synthetic[:, column],
                    synthetic_scores[:, column],
                    requested_recall,
                )
                feasible = chosen is not None
                reason = (
                    "separate_positive_strata_constraints_met"
                    if feasible
                    else "separate_strata_recall_constraint_infeasible"
                )
                if chosen is not None:
                    threshold = chosen
        real = _stratum(
            truth[:, column],
            scores[:, column],
            threshold,
            split="validation",
            provenance=label_provenance,
        )
        auxiliary = (
            _stratum(
                synthetic[:, column],
                synthetic_scores[:, column],
                threshold,
                split="calibration",
                provenance=calibration_label_provenance,
            )
            if sparse
            else None
        )
        combined_precision = None
        if auxiliary is not None:
            tp = real.metrics.tp + auxiliary.metrics.tp
            fp = real.metrics.fp + auxiliary.metrics.fp
            combined_precision = tp / (tp + fp) if tp + fp else None
        selections[label] = CalibratedThresholdResult(
            threshold=threshold,
            feasible=feasible,
            reason=reason,
            support=real_support,
            **real.metrics.model_dump(
                include={"precision", "recall", "recall_ci_lower", "recall_ci_upper"}
            ),
            calibration_origin="real_validation_only"
            if not sparse
            else (
                "real_validation_and_synthetic_calibration"
                if real_support
                else "synthetic_calibration_only"
            ),
            real_validation=real,
            synthetic_calibration=auxiliary,
            precision_objective="combined_machine_reference" if sparse else "real_validation",
            combined_precision=combined_precision,
        )
    return CalibrationThresholdSelection(
        thresholds=[selections[label].threshold for label in LABELS],
        per_label=selections,
        label_provenance=label_provenance,
        calibration_label_provenance=calibration_label_provenance,
        requested_recall=requested_recall,
    )


def validation_utility_gate(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    thresholds: Sequence[float],
    *,
    split: Literal["validation"],
    label_provenance: str,
) -> ValidationUtilityGate:
    """Require <=5% OR-removal among real all-negative union references."""
    if split != "validation":
        raise ValueError("Utility gate requires real validation; test input is forbidden")
    _provenance(label_provenance)
    truth = _binary_matrix(y_true, "y_true")
    scores = _probability_matrix(probabilities, truth)
    limits = np.asarray(thresholds)
    if (
        limits.shape != (len(LABELS),)
        or limits.dtype.kind not in "iuf"
        or not np.all(np.isfinite(limits) & (limits >= 0) & (limits <= 1))
    ):
        raise ValueError("Five finite thresholds in [0, 1] are required")
    negative = ~truth.any(axis=1)
    removed = (scores >= limits).any(axis=1)
    support, false_positives = int(negative.sum()), int((negative & removed).sum())
    rate = false_positives / support if support else None
    passed = rate is not None and rate <= 0.05
    return ValidationUtilityGate(
        label_provenance=label_provenance,
        all_negative_support=support,
        unnecessarily_removed=false_positives,
        false_positive_rate=rate,
        passed=passed,
        reason="validation_utility_gate_passed"
        if passed
        else (
            "no_real_all_negative_support"
            if not support
            else "real_validation_removal_fpr_exceeds_limit"
        ),
    )


def document_cluster_bootstrap(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    document_ids: Sequence[str],
    *,
    label_provenance: str,
    cluster_ids: Sequence[str] | None = None,
    resamples: int = 2000,
    seed: int = 42,
) -> BootstrapReport:
    """Resample independent clusters while measuring leakage per original document.

    Supply cluster_ids aligned with sentences to group near-duplicate originals;
    each original document must belong to one cluster. With cluster_ids omitted,
    each original is its own sampling unit, preserving the existing behavior.
    Every draw retains all original documents and sentences in that cluster.
    Repeated draws have fresh replicate identities, implemented by summing
    counts with multiplicity rather than deduplicating source document IDs.

    Recall is sentence-weighted and leakage is original-document-weighted;
    neither is an average of cluster-level rates. Resamples without positives
    for a label are excluded only for that label and their number is reported.
    Fewer than two clusters cannot yield an interval. A zero-width interval
    cannot establish that the population has no errors.
    """
    _provenance(label_provenance)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    truth, predictions = _aligned_predictions(y_true, y_pred)
    groups, documents = _document_groups(document_ids, len(truth))
    if cluster_ids is None:
        sentence_clusters, clusters = groups, documents
    else:
        sentence_clusters, clusters = _document_groups(cluster_ids, len(truth), name="cluster_ids")
    document_clusters = np.full(documents, -1, dtype=np.int64)
    for document, cluster in zip(groups, sentence_clusters, strict=True):
        if document_clusters[document] not in (-1, cluster):
            raise ValueError("Each original document must map to exactly one cluster_id")
        document_clusters[document] = cluster
    tps = np.zeros((documents, len(LABELS)), dtype=np.int64)
    positives = np.zeros_like(tps)
    np.add.at(tps, groups, truth & predictions)
    np.add.at(positives, groups, truth)
    leaked = np.zeros(documents, dtype=np.bool_)
    np.logical_or.at(leaked, groups, truth.any(axis=1) & ~predictions.any(axis=1))
    cluster_tps = np.zeros((clusters, len(LABELS)), dtype=np.int64)
    cluster_positives = np.zeros_like(cluster_tps)
    cluster_leaks = np.zeros(clusters, dtype=np.int64)
    np.add.at(cluster_tps, document_clusters, tps)
    np.add.at(cluster_positives, document_clusters, positives)
    np.add.at(cluster_leaks, document_clusters, leaked)
    cluster_documents = np.bincount(document_clusters, minlength=clusters)
    recalls: list[list[float]] = [[] for _ in LABELS]
    leak_rates: list[float] = []
    rng = np.random.default_rng(seed)
    if clusters >= 2:
        for _ in range(resamples):
            sample = rng.integers(0, clusters, size=clusters)
            sampled_tp = cluster_tps[sample].sum(axis=0)
            sampled_positive = cluster_positives[sample].sum(axis=0)
            for column in range(len(LABELS)):
                if sampled_positive[column]:
                    recalls[column].append(float(sampled_tp[column] / sampled_positive[column]))
            # A cluster drawn twice contributes both copies of every original,
            # exactly as assigning (draw_index, original_id) replicate IDs.
            leak_rates.append(float(cluster_leaks[sample].sum() / cluster_documents[sample].sum()))

    def interval(values: list[float]) -> BootstrapInterval:
        if not values:
            return BootstrapInterval(lower=None, upper=None, valid_resamples=0)
        lower, upper = np.quantile(values, [0.025, 0.975])
        return BootstrapInterval(
            lower=float(lower), upper=float(upper), valid_resamples=len(values)
        )

    return BootstrapReport(
        label_provenance=label_provenance,
        seed=seed,
        resamples=resamples,
        documents=documents,
        original_documents=documents,
        clusters=clusters,
        resampling_unit="original_document" if cluster_ids is None else "document_cluster",
        recall={label: interval(recalls[column]) for column, label in enumerate(LABELS)},
        document_any_leak_rate=interval(leak_rates),
        interval_method=(
            "95% percentile bootstrap resampling whole document clusters; clusters assumed independent; "
            "original documents and sentences within a cluster remain together; "
            "each repeated draw retains separate original-document replicate identities; "
            "recall is sentence-weighted and leakage is original-document-weighted; "
            "fewer than two clusters yields no interval; "
            "degenerate intervals cannot rule out unobserved errors"
            if cluster_ids is not None
            else BootstrapReport.model_fields["interval_method"].default
        ),
    )
