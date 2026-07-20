from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_spx_with_regimes(
    prices_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    labels: Optional[Dict[int, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot market benchmark price overlaid with colored regime background shading.

    Args:
        prices_df: DataFrame containing price series
        regime_df: DataFrame containing 'regime_id' column, indexed by date
        labels: Mapping of regime_id to string label
        save_path: Optional file path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Determine price series
    if "close" in prices_df.columns and "ticker" in prices_df.columns:
        p_df = prices_df.pivot(index="date", columns="ticker", values="close").median(axis=1)
    elif "close" in prices_df.columns:
        p_df = prices_df["close"]
    else:
        p_df = prices_df.select_dtypes(include=[np.number]).median(axis=1)

    p_df = p_df.sort_index()
    reg_series = regime_df["regime_id"].reindex(p_df.index).ffill().bfill()

    # Plot price line
    ax.plot(p_df.index, p_df.values, color="black", linewidth=1.5, label="Market Benchmark")

    # Define color palette for up to 6 regimes
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    unique_states = sorted(reg_series.dropna().unique().astype(int))

    # Shade regimes
    for state in unique_states:
        mask = reg_series == state
        state_label = labels.get(state, f"Regime {state}") if labels else f"Regime {state}"
        color = colors[state % len(colors)]

        # Find contiguous regions
        diff = mask.astype(int).diff().fillna(0)
        starts = p_df.index[diff == 1].tolist()
        ends = p_df.index[diff == -1].tolist()

        if mask.iloc[0]:
            starts.insert(0, p_df.index[0])
        if mask.iloc[-1]:
            ends.append(p_df.index[-1])

        added_label = False
        for start, end in zip(starts, ends):
            lbl = state_label if not added_label else None
            ax.axvspan(start, end, color=color, alpha=0.3, label=lbl)
            added_label = True

    ax.set_title("Market Price & HMM Regime States", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_transition_matrix(
    trans_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot transition probability matrix heatmap.

    Args:
        trans_df: Square DataFrame of transition probabilities
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        trans_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=1,
        square=True,
    )

    ax.set_title("HMM State Transition Matrix $P_{ij}$", fontsize=13, fontweight="bold")
    ax.set_xlabel("To State $j$")
    ax.set_ylabel("From State $i$")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_regime_feature_profiles(
    feature_df: pd.DataFrame,
    state_series: pd.Series,
    labels: Optional[Dict[int, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot standardized feature profile means across regimes.

    Args:
        feature_df: Unscaled feature matrix
        state_series: Series of hard state IDs
        labels: Optional label dictionary
        save_path: Optional output path

    Returns:
        matplotlib Figure
    """
    # Standardize feature matrix for comparison across different units
    scaled_features = (feature_df - feature_df.mean()) / feature_df.std()
    scaled_features["regime_id"] = state_series

    grouped = scaled_features.groupby("regime_id").mean()

    if labels:
        grouped.index = [labels.get(i, f"State {i}") for i in grouped.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.T.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title("Normalized Macro/Market Feature Profiles by Regime State", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Standardized Mean Score (Z-Score)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Regime State", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig
