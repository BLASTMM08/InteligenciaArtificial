"""Custom Naive Bayes classifier that estimates likelihoods via 1D KDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass
class FeatureDensity:
    """Container for a fitted per-feature KDE and the bandwidth actually used."""

    kde: KernelDensity
    bandwidth: float


class KDENaiveBayes(BaseEstimator, ClassifierMixin):
    """Naive Bayes variant that factorizes likelihoods and fits 1D KDEs.

    Parameters
    ----------
    kernel:
        Kernel passed to :class:`sklearn.neighbors.KernelDensity`.
    bandwidth:
        Smoothing parameter. For ``bandwidth_source='silverman'`` the value is
        ignored. For ``bandwidth_source='fixed'`` must be a positive float.
    bandwidth_source:
        Whether the bandwidth is fixed or computed via the Silverman rule.
        Accepted values: ``'fixed'`` or ``'silverman'``.
    min_bandwidth:
        Lower bound applied to all per-feature bandwidths to keep the KDE stable
        when the empirical standard deviation is ~0.
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        bandwidth_source: str = "fixed",
        min_bandwidth: float = 1e-3,
    ) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_source = bandwidth_source
        self.min_bandwidth = min_bandwidth

    def fit(
        self,
        X: np.ndarray,
        y: Iterable[int],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "KDENaiveBayes":
        X = check_array(X, ensure_2d=True, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("sample_weight must have length == n_samples.")

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.class_feature_densities_: Dict[int, List[FeatureDensity]] = {}

        if sample_weight is None:
            class_weight = np.bincount(y_indices)
            total_weight = float(X.shape[0])
        else:
            class_weight = np.zeros_like(self.classes_, dtype=float)
            for idx, cls in enumerate(self.classes_):
                class_weight[idx] = sample_weight[y == cls].sum()
            total_weight = float(sample_weight.sum())

        self.class_log_prior_ = np.log(class_weight / total_weight)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]
            weights_cls = sample_weight[mask] if sample_weight is not None else None
            densities: List[FeatureDensity] = []
            for feat_idx in range(self.n_features_in_):
                column = X_cls[:, feat_idx : feat_idx + 1]  # keep 2D
                bandwidth = self._resolve_bandwidth(column, weights_cls)
                kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
                kde.fit(column, sample_weight=weights_cls)
                densities.append(FeatureDensity(kde=kde, bandwidth=bandwidth))
            self.class_feature_densities_[cls] = densities

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_proba(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "class_feature_densities_")
        X = check_array(X, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected number of features at prediction time.")

        n_samples = X.shape[0]
        joint = np.empty((n_samples, len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            log_likelihood = np.zeros(n_samples)
            for feat_idx, density in enumerate(self.class_feature_densities_[cls]):
                column = X[:, feat_idx : feat_idx + 1]
                log_likelihood += density.kde.score_samples(column)
            joint[:, idx] = self.class_log_prior_[idx] + log_likelihood
        return joint

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _resolve_bandwidth(
        self, column: np.ndarray, weights: Optional[np.ndarray]
    ) -> float:
        if self.bandwidth_source not in {"fixed", "silverman"}:
            raise ValueError(
                "bandwidth_source must be 'fixed' or 'silverman', "
                f"got {self.bandwidth_source!r}"
            )

        if self.bandwidth_source == "fixed":
            if not np.isscalar(self.bandwidth):
                raise ValueError("bandwidth must be a positive scalar when fixed.")
            bw = float(self.bandwidth)
        else:
            bw = self._silverman_rule(column, weights)

        return max(bw, self.min_bandwidth)

    def _silverman_rule(
        self, column: np.ndarray, weights: Optional[np.ndarray]
    ) -> float:
        values = column.ravel()
        if weights is None:
            std = np.std(values, ddof=1)
            n = len(values)
        else:
            # Weighted variance
            w = weights
            w_sum = w.sum()
            mean = np.average(values, weights=w)
            std = np.sqrt(np.average((values - mean) ** 2, weights=w))
            n = int(round(w_sum))
        if n <= 1 or std == 0:
            return self.min_bandwidth
        return 1.06 * std * (n ** (-1 / 5))

    # Convenience accessors ------------------------------------------------ #
    def get_bandwidths(self) -> Dict[int, List[float]]:
        """Return the bandwidth used for each (class, feature) pair."""
        check_is_fitted(self, "class_feature_densities_")
        report: Dict[int, List[float]] = {}
        for cls, densities in self.class_feature_densities_.items():
            report[cls] = [density.bandwidth for density in densities]
        return report

