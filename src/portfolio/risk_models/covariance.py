"""Pluggable Covariance Estimators for Portfolio Construction.

Supports:
- Ledoit-Wolf Shrinkage (constant correlation target)
- Oracle Approximating Shrinkage (OAS)
- Empirical Sample Covariance with Regularization
- Factor-Decomposed Covariance (Systematic + Idiosyncratic)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


class BaseCovarianceEstimator(ABC):
    """Abstract base class for all asset covariance estimators."""

    def __init__(self, eps: float = 1e-6, logger: Optional[logging.Logger] = None) -> None:
        self.eps = eps
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def estimate(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate asset covariance matrix (N x N) from return panel (T x N).

        Parameters
        ----------
        returns_df : pd.DataFrame
            Time series return panel indexed by date with ticker columns.

        Returns
        -------
        pd.DataFrame
            Symmetric, positive semi-definite N x N covariance matrix.
        """
        pass

    def _regularize(self, cov_matrix: np.ndarray, tickers: list[str]) -> pd.DataFrame:
        """Ensure covariance matrix is symmetric and strictly positive-definite."""
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig <= 0:
            cov_matrix += (abs(min_eig) + self.eps) * np.eye(cov_matrix.shape[0])

        return pd.DataFrame(cov_matrix, index=tickers, columns=tickers)


class LedoitWolfCovariance(BaseCovarianceEstimator):
    """Ledoit-Wolf shrinkage covariance estimator towards constant correlation."""

    def estimate(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        clean_returns = returns_df.dropna(how="all").fillna(0.0)
        tickers = list(clean_returns.columns)
        if clean_returns.shape[0] < 2 or clean_returns.shape[1] == 0:
            return pd.DataFrame(np.eye(len(tickers)) * self.eps, index=tickers, columns=tickers)

        try:
            lw = LedoitWolf()
            cov_arr, shrinkage = lw.fit(clean_returns.values).covariance_, lw.shrinkage_
            self.logger.debug("Ledoit-Wolf covariance estimated with shrinkage=%.4f", shrinkage)
            return self._regularize(cov_arr, tickers)
        except Exception as exc:
            self.logger.warning("Ledoit-Wolf failed (%s), falling back to SampleCovariance", exc)
            return SampleCovariance(eps=self.eps, logger=self.logger).estimate(returns_df)


class OASCovariance(BaseCovarianceEstimator):
    """Oracle Approximating Shrinkage (OAS) covariance estimator."""

    def estimate(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        clean_returns = returns_df.dropna(how="all").fillna(0.0)
        tickers = list(clean_returns.columns)
        if clean_returns.shape[0] < 2 or clean_returns.shape[1] == 0:
            return pd.DataFrame(np.eye(len(tickers)) * self.eps, index=tickers, columns=tickers)

        try:
            oas = OAS()
            cov_arr = oas.fit(clean_returns.values).covariance_
            return self._regularize(cov_arr, tickers)
        except Exception as exc:
            self.logger.warning("OAS failed (%s), falling back to SampleCovariance", exc)
            return SampleCovariance(eps=self.eps, logger=self.logger).estimate(returns_df)


class SampleCovariance(BaseCovarianceEstimator):
    """Empirical sample covariance with diagonal regularization."""

    def estimate(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        clean_returns = returns_df.dropna(how="all").fillna(0.0)
        tickers = list(clean_returns.columns)
        if clean_returns.shape[0] < 2 or clean_returns.shape[1] == 0:
            return pd.DataFrame(np.eye(len(tickers)) * self.eps, index=tickers, columns=tickers)

        cov_arr = np.cov(clean_returns.values, rowvar=False)
        if cov_arr.ndim == 0:
            cov_arr = np.array([[cov_arr]])
        return self._regularize(cov_arr, tickers)


class FactorCovariance(BaseCovarianceEstimator):
    """Factor-decomposed covariance estimator: Cov = B * Cov_factor * B^T + D_idiosyncratic."""

    def estimate(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        # Defaults to Ledoit-Wolf unless explicit factor loadings are supplied
        return LedoitWolfCovariance(eps=self.eps, logger=self.logger).estimate(returns_df)

    def estimate_decomposed(
        self,
        returns_df: pd.DataFrame,
        factor_exposures_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Decompose asset covariance into systematic factor covariance and idiosyncratic diagonal matrix.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (Total Covariance, Systematic Covariance, Idiosyncratic Diagonal Matrix)
        """
        clean_returns = returns_df.dropna(how="all").fillna(0.0)
        tickers = list(clean_returns.columns)
        common_tickers = [t for t in tickers if t in factor_exposures_df.index]

        if not common_tickers:
            total_cov = SampleCovariance(eps=self.eps, logger=self.logger).estimate(returns_df)
            return total_cov, total_cov, pd.DataFrame(np.zeros_like(total_cov), index=tickers, columns=tickers)

        B = factor_exposures_df.loc[common_tickers].fillna(0.0).values
        ret_sub = clean_returns[common_tickers].values

        # Estimate factor returns via cross-sectional regression: R_t = B * F_t + eps_t
        try:
            # F_t = (B^T B)^-1 B^T R_t^T
            factor_returns = np.linalg.pinv(B) @ ret_sub.T  # K x T
            factor_cov = np.cov(factor_returns)  # K x K

            systematic_cov = B @ factor_cov @ B.T  # N x N
            residuals = ret_sub.T - (B @ factor_returns)  # N x T
            idiosyncratic_var = np.var(residuals, axis=1)  # N
            D = np.diag(idiosyncratic_var)

            total_cov_arr = systematic_cov + D
            total_cov = self._regularize(total_cov_arr, common_tickers)
            sys_cov_df = pd.DataFrame(systematic_cov, index=common_tickers, columns=common_tickers)
            idio_df = pd.DataFrame(D, index=common_tickers, columns=common_tickers)

            return total_cov, sys_cov_df, idio_df
        except Exception as exc:
            self.logger.warning("Factor covariance decomposition failed (%s); returning sample cov", exc)
            total_cov = SampleCovariance(eps=self.eps, logger=self.logger).estimate(returns_df)
            return total_cov, total_cov, pd.DataFrame(np.zeros_like(total_cov), index=tickers, columns=tickers)


# Registry lookup helper
_ESTIMATOR_REGISTRY: Dict[str, Type[BaseCovarianceEstimator]] = {
    "ledoit_wolf": LedoitWolfCovariance,
    "oas": OASCovariance,
    "sample": SampleCovariance,
    "factor": FactorCovariance,
}


def get_covariance_estimator(
    name: str = "ledoit_wolf",
    eps: float = 1e-6,
    logger: Optional[logging.Logger] = None,
) -> BaseCovarianceEstimator:
    """Factory function for instantiating covariance estimators."""
    key = name.lower().strip()
    if key not in _ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown covariance model '{name}'. Available: {list(_ESTIMATOR_REGISTRY.keys())}")
    return _ESTIMATOR_REGISTRY[key](eps=eps, logger=logger)
