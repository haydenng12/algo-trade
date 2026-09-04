from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _clean_returns(returns: pd.Series) -> pd.Series:
    values = returns.astype(float).dropna()
    if values.empty:
        return values
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Returns must contain only finite values.")
    return values


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualize sample standard deviation of periodic returns."""
    values = _clean_returns(returns)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) * math.sqrt(periods_per_year))


def _periodic_risk_free_rate(annual_rate: float, periods_per_year: int) -> float:
    if annual_rate <= -1:
        raise ValueError("Annual risk-free rate must be greater than -100%.")
    return (1.0 + float(annual_rate)) ** (1.0 / periods_per_year) - 1.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using periodic excess returns.

    ``risk_free_rate`` is expressed as an annual effective rate and converted
    to an equivalent periodic rate before computing excess returns.
    """
    values = _clean_returns(returns)
    if len(values) < 2:
        return float("nan")

    periodic_rf = _periodic_risk_free_rate(risk_free_rate, periods_per_year)
    excess = values - periodic_rf
    std = float(excess.std(ddof=1))
    if np.isclose(std, 0.0):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation.

    Numerator: mean periodic excess return over the risk-free rate.
    Denominator: root-mean-square shortfall below ``target_return``.
    ``target_return`` is a periodic target, defaulting to 0% per period.
    """
    values = _clean_returns(returns)
    if values.empty:
        return float("nan")

    periodic_rf = _periodic_risk_free_rate(risk_free_rate, periods_per_year)
    excess = values - periodic_rf
    downside = np.minimum(values.to_numpy() - float(target_return), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))

    if np.isclose(downside_deviation, 0.0):
        return float("nan")

    return float(
        excess.mean() * periods_per_year
        / (downside_deviation * math.sqrt(periods_per_year))
    )
