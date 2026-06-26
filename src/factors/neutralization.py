from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    """Winsorize a pandas Series by clipping extreme values.

    Args:
        series: Ticker-indexed factor scores.
        limits: Tuple containing (lower_fraction, upper_fraction) to winsorize.
                E.g. (0.01, 0.01) clips values below 1st percentile and above 99th.

    Returns:
        pd.Series: Winsorized Series.
    """
    if series.empty:
        return series
    
    valid_series = series.dropna()
    if valid_series.empty:
        return series

    lower_pct, upper_pct = limits
    lower_val = valid_series.quantile(lower_pct)
    upper_val = valid_series.quantile(1.0 - upper_pct)

    return series.clip(lower=lower_val, upper=upper_val)


def z_score_series(series: pd.Series) -> pd.Series:
    """Z-score standardize a pandas Series: subtract mean and divide by standard deviation.

    Args:
        series: Ticker-indexed factor scores.

    Returns:
        pd.Series: Standardized Series.
    """
    if series.empty:
        return series

    mean_val = series.mean()
    std_val = series.std()

    if pd.isna(std_val) or std_val == 0:
        # If standard deviation is zero, return series with zeros for valid numbers
        return series.map(lambda x: 0.0 if not pd.isna(x) else np.nan)

    return (series - mean_val) / std_val


def neutralize_factor_scores(
    factor_scores: pd.Series,
    sizes: pd.Series,
    sectors: pd.Series,
    neutralize_size: bool = True,
    neutralize_sector: bool = True,
) -> pd.Series:
    """Neutralize factor scores against Size (log of market cap) and Sector.

    Runs OLS regression: Factor ~ intercept + log_size + sector_dummies
    Returns the OLS residuals as neutralized factor scores.

    Args:
        factor_scores: Series of z-scored factor values indexed by ticker.
        sizes: Series of raw market caps/sizes indexed by ticker.
        sectors: Series of sector names indexed by ticker.
        neutralize_size: Flag to enable size neutralization.
        neutralize_sector: Flag to enable sector dummy neutralization.

    Returns:
        pd.Series: Neutralized factor scores (residuals) with same index.
    """
    if factor_scores.empty:
        return factor_scores

    # Build regression DataFrame
    df = pd.DataFrame({
        "y": factor_scores,
        "size": sizes,
        "sector": sectors
    })

    # Drop missing target values
    df = df.dropna(subset=["y"])
    if df.empty:
        return pd.Series(dtype=float, index=factor_scores.index)

    # Clean Size: convert to log format
    if neutralize_size:
        df["log_size"] = np.log(df["size"].astype(float))
        # If any log_size values are infinite or missing, drop them or fill with median
        df = df[np.isfinite(df["log_size"])]
    
    if df.empty:
        return pd.Series(dtype=float, index=factor_scores.index)

    # Build regressor matrix X
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0

    if neutralize_size:
        X["log_size"] = df["log_size"]

    if neutralize_sector:
        # Create sector dummies
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=True, dtype=float)
        X = pd.concat([X, sector_dummies], axis=1)

    # Check that we have enough degrees of freedom to run OLS
    if len(df) <= len(X.columns):
        logger.warning(
            "Insufficient observations (%d) compared to variables (%d) for neutralization regression. Skipping OLS.",
            len(df), len(X.columns)
        )
        return df["y"]  # Return z-scored factors without adjustments

    try:
        y = df["y"]
        model = sm.OLS(y, X, missing="drop")
        results = model.fit()
        residuals = results.resid
        
        # Re-index to the original series' index, filling dropped rows with NaN
        neutralized = residuals.reindex(factor_scores.index)
        return neutralized
    except Exception as exc:
        logger.error("Error during OLS neutralization: %s. Returning raw values.", exc)
        return factor_scores
