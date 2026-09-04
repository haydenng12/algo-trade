from __future__ import annotations

import pandas as pd

from backtest.results import BacktestResult
from data.cleaner import validate_ohlcv
from execution.costs import TransactionCostModel
from execution.slippage import SlippageModel
from portfolio.portfolio import Portfolio, Position
from portfolio.sizing import FixedFractionSizer
from strategies.base import Strategy


TRADE_COLUMNS = [
    "symbol", "direction", "entry_date", "exit_date",
    "entry_market_price", "exit_market_price",
    "entry_price", "exit_price", "quantity",
    "gross_pnl", "transaction_costs", "slippage_costs", "costs",
    "net_pnl", "return_pct", "holding_period_days",
]


class BacktestEngine:
    """Single-symbol, long/flat daily backtester with Stage 4 execution realism.

    Default timing convention:
      * strategy signal for day t is computed using day-t data
      * with execution_delay=1, that signal executes at day t+1 Open
      * portfolio is marked to market at each day's Close

    Buys receive adverse positive slippage; sells receive adverse negative
    slippage. Transaction costs are charged separately on executed notional.
    The engine resizes buys when necessary so cash never becomes negative.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
        position_fraction: float = 1.0,
        execution_delay: int = 1,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if execution_delay < 1:
            raise ValueError("execution_delay must be at least 1 to prevent look-ahead.")

        self.initial_capital = float(initial_capital)
        self.cost_model = TransactionCostModel(transaction_cost_bps)
        self.slippage_model = SlippageModel(slippage_bps)
        self.sizer = FixedFractionSizer(position_fraction)
        self.execution_delay = int(execution_delay)

    def _affordable_quantity(self, cash: float, desired_notional: float, execution_price: float) -> float:
        """Return a quantity that respects both the target size and cash+fee constraint."""
        if cash <= 0 or desired_notional <= 0:
            return 0.0
        desired_qty = desired_notional / execution_price
        max_qty = cash / (execution_price * (1.0 + self.cost_model.rate))
        return float(max(0.0, min(desired_qty, max_qty)))

    def run(self, symbol: str, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        validate_ohlcv(data)
        if len(data) < 2:
            raise ValueError("Backtest requires at least two market-data rows.")

        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol cannot be blank.")

        signals = strategy.generate_signals(data).reindex(data.index)
        if signals.isna().any():
            raise ValueError("Strategy returned missing signals.")
        if not set(signals.unique()).issubset({-1, 0, 1}):
            raise ValueError("Strategy signals must be in {-1, 0, 1}.")
        if (signals < 0).any():
            raise ValueError("Stage 4 engine is long/flat only; short signals are not supported yet.")

        portfolio = Portfolio(initial_capital=self.initial_capital)
        trades: list[dict] = []
        daily_rows: list[dict] = []
        cumulative_transaction_costs = 0.0
        cumulative_slippage_costs = 0.0

        for i, (date, row) in enumerate(data.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            executed_signal = None
            traded_notional = 0.0
            day_transaction_costs = 0.0
            day_slippage_costs = 0.0

            signal_index = i - self.execution_delay
            if signal_index >= 0:
                executed_signal = int(signals.iloc[signal_index])
                position = portfolio.position
                is_long = position is not None and position.is_open

                if executed_signal == 1 and not is_long:
                    execution_price = self.slippage_model.execution_price(open_price, "BUY")
                    portfolio_value_at_open = portfolio.equity(open_price)
                    desired_notional = self.sizer.target_notional(portfolio_value_at_open)
                    quantity = self._affordable_quantity(
                        cash=float(portfolio.cash),
                        desired_notional=desired_notional,
                        execution_price=execution_price,
                    )

                    if quantity > 0:
                        traded_notional = quantity * execution_price
                        fee = self.cost_model.calculate(traded_notional)
                        slippage_cost = quantity * (execution_price - open_price)
                        total_cash_used = traded_notional + fee

                        portfolio.cash -= total_cash_used
                        if portfolio.cash < -1e-9:
                            raise RuntimeError("Cash constraint violated during buy execution.")
                        if abs(portfolio.cash) < 1e-9:
                            portfolio.cash = 0.0

                        portfolio.position = Position(
                            symbol=symbol,
                            quantity=quantity,
                            entry_market_price=open_price,
                            entry_price=execution_price,
                            entry_date=date,
                            entry_fee=fee,
                            entry_slippage_cost=slippage_cost,
                        )
                        day_transaction_costs += fee
                        day_slippage_costs += slippage_cost

                elif executed_signal == 0 and is_long:
                    assert position is not None
                    assert position.entry_price is not None
                    assert position.entry_market_price is not None

                    execution_price = self.slippage_model.execution_price(open_price, "SELL")
                    traded_notional = position.quantity * execution_price
                    exit_fee = self.cost_model.calculate(traded_notional)
                    exit_slippage = position.quantity * (open_price - execution_price)
                    proceeds = traded_notional - exit_fee
                    portfolio.cash += proceeds

                    gross_pnl = (open_price - position.entry_market_price) * position.quantity
                    transaction_costs = position.entry_fee + exit_fee
                    slippage_costs = position.entry_slippage_cost + exit_slippage
                    total_costs = transaction_costs + slippage_costs
                    net_pnl = gross_pnl - total_costs
                    holding_days = (date - position.entry_date).days
                    invested_market_notional = position.entry_market_price * position.quantity
                    return_pct = net_pnl / invested_market_notional if invested_market_notional > 0 else float("nan")

                    trades.append(
                        {
                            "symbol": symbol,
                            "direction": "LONG",
                            "entry_date": position.entry_date,
                            "exit_date": date,
                            "entry_market_price": position.entry_market_price,
                            "exit_market_price": open_price,
                            "entry_price": position.entry_price,
                            "exit_price": execution_price,
                            "quantity": position.quantity,
                            "gross_pnl": gross_pnl,
                            "transaction_costs": transaction_costs,
                            "slippage_costs": slippage_costs,
                            "costs": total_costs,
                            "net_pnl": net_pnl,
                            "return_pct": return_pct,
                            "holding_period_days": holding_days,
                        }
                    )
                    day_transaction_costs += exit_fee
                    day_slippage_costs += exit_slippage
                    portfolio.position = None

            cumulative_transaction_costs += day_transaction_costs
            cumulative_slippage_costs += day_slippage_costs

            quantity = 0.0 if portfolio.position is None else portfolio.position.quantity
            equity = portfolio.equity(close_price)
            daily_rows.append(
                {
                    "Date": date,
                    "signal": int(signals.loc[date]),
                    "executed_signal": executed_signal,
                    "cash": float(portfolio.cash),
                    "quantity": float(quantity),
                    "market_value": float(quantity * close_price),
                    "equity": float(equity),
                    "traded_notional": float(traded_notional),
                    "transaction_costs": float(day_transaction_costs),
                    "slippage_costs": float(day_slippage_costs),
                    "cumulative_transaction_costs": float(cumulative_transaction_costs),
                    "cumulative_slippage_costs": float(cumulative_slippage_costs),
                }
            )

        equity_curve = pd.DataFrame(daily_rows).set_index("Date")
        equity_curve.index = pd.DatetimeIndex(equity_curve.index, name="Date")
        trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)

        return BacktestResult(
            symbol=symbol,
            initial_capital=self.initial_capital,
            signals=signals.copy(),
            equity_curve=equity_curve,
            trades=trade_frame,
        )
