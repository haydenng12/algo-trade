from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.cleaner import validate_ohlcv
from execution.costs import TransactionCostModel
from execution.slippage import SlippageModel


@dataclass
class MultiAssetBacktestResult:
    initial_capital: float
    equity_curve: pd.DataFrame
    target_weights: pd.DataFrame
    holdings: pd.DataFrame

    @property
    def ending_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1])


class MultiAssetTargetWeightEngine:
    """Long-only multi-asset target-weight backtester.

    Target weights decided on day t execute at day t+delay Open. Rebalancing
    sells are processed before buys. Costs and adverse slippage are applied to
    every change in position. Weights must be non-negative and sum to <= 1,
    leaving any residual allocation in cash.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
        execution_delay: int = 1,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if execution_delay < 1:
            raise ValueError("execution_delay must be at least 1.")
        self.initial_capital = float(initial_capital)
        self.cost_model = TransactionCostModel(transaction_cost_bps)
        self.slippage_model = SlippageModel(slippage_bps)
        self.execution_delay = int(execution_delay)

    @staticmethod
    def _align(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        if not data:
            raise ValueError("data cannot be empty.")
        common = None
        cleaned: dict[str, pd.DataFrame] = {}
        for raw_symbol, frame in data.items():
            symbol = raw_symbol.upper().strip()
            if not symbol:
                raise ValueError("symbols cannot be blank.")
            validate_ohlcv(frame)
            cleaned[symbol] = frame
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) < 2:
            raise ValueError("Assets need at least two common dates.")
        common = pd.DatetimeIndex(common).sort_values()
        return {s: f.reindex(common) for s, f in cleaned.items()}, common

    def run(self, data: dict[str, pd.DataFrame], target_weights: pd.DataFrame) -> MultiAssetBacktestResult:
        aligned, dates = self._align(data)
        symbols = sorted(aligned)
        weights = target_weights.reindex(index=dates, columns=symbols).fillna(0.0).astype(float)
        if (weights < -1e-12).any().any():
            raise ValueError("Target weights must be non-negative.")
        if (weights.sum(axis=1) > 1.0 + 1e-9).any():
            raise ValueError("Target weights cannot sum above 1 without leverage support.")

        cash = self.initial_capital
        qty = {s: 0.0 for s in symbols}
        rows: list[dict] = []
        holdings_rows: list[dict] = []
        cumulative_fees = 0.0
        cumulative_slippage = 0.0

        for i, date in enumerate(dates):
            opens = {s: float(aligned[s].loc[date, "Open"]) for s in symbols}
            closes = {s: float(aligned[s].loc[date, "Close"]) for s in symbols}
            day_fees = 0.0
            day_slippage = 0.0
            day_notional = 0.0
            source = i - self.execution_delay

            if source >= 0:
                desired = weights.iloc[source]
                equity_open = cash + sum(qty[s] * opens[s] for s in symbols)
                target_qty = {s: equity_open * float(desired[s]) / opens[s] for s in symbols}

                # Sell reductions first to free cash.
                for s in symbols:
                    delta = target_qty[s] - qty[s]
                    if delta < -1e-12:
                        sell_qty = -delta
                        px = self.slippage_model.execution_price(opens[s], "SELL")
                        notional = sell_qty * px
                        fee = self.cost_model.calculate(notional)
                        slip = sell_qty * (opens[s] - px)
                        cash += notional - fee
                        qty[s] -= sell_qty
                        day_notional += notional
                        day_fees += fee
                        day_slippage += slip

                # Buy increases. If cash is tight, scale each desired buy to affordability.
                for s in symbols:
                    delta = target_qty[s] - qty[s]
                    if delta > 1e-12 and cash > 0:
                        px = self.slippage_model.execution_price(opens[s], "BUY")
                        max_affordable = cash / (px * (1.0 + self.cost_model.rate))
                        buy_qty = min(delta, max_affordable)
                        if buy_qty <= 0:
                            continue
                        notional = buy_qty * px
                        fee = self.cost_model.calculate(notional)
                        slip = buy_qty * (px - opens[s])
                        cash -= notional + fee
                        qty[s] += buy_qty
                        day_notional += notional
                        day_fees += fee
                        day_slippage += slip

            cumulative_fees += day_fees
            cumulative_slippage += day_slippage
            equity = cash + sum(qty[s] * closes[s] for s in symbols)
            rows.append({
                "Date": date,
                "cash": float(cash),
                "market_value": float(equity - cash),
                "equity": float(equity),
                "traded_notional": float(day_notional),
                "transaction_costs": float(day_fees),
                "slippage_costs": float(day_slippage),
                "cumulative_transaction_costs": float(cumulative_fees),
                "cumulative_slippage_costs": float(cumulative_slippage),
            })
            holdings_rows.append({"Date": date, **{s: float(qty[s]) for s in symbols}})

        curve = pd.DataFrame(rows).set_index("Date")
        holdings = pd.DataFrame(holdings_rows).set_index("Date")
        return MultiAssetBacktestResult(
            initial_capital=self.initial_capital,
            equity_curve=curve,
            target_weights=weights,
            holdings=holdings,
        )
