from __future__ import annotations

import pandas as pd

from backtest.results import BacktestResult
from data.cleaner import validate_ohlcv
from portfolio.portfolio import Portfolio, Position
from strategies.base import Strategy


TRADE_COLUMNS = [
    "symbol", "direction", "entry_date", "exit_date", "entry_price",
    "exit_price", "quantity", "gross_pnl", "costs", "net_pnl",
    "return_pct", "holding_period_days",
]


class BacktestEngine:
    """Stage 2 single-symbol, long/flat daily backtester.

    Timing convention:
      * strategy signal for day t is computed using day-t data
      * that signal becomes executable at day t+1 Open
      * portfolio is marked to market at each day's Close

    Stage 2 deliberately has zero fees/slippage and allows fractional shares.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        self.initial_capital = float(initial_capital)

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
            raise ValueError("Stage 2 engine is long/flat only; short signals are not supported yet.")

        portfolio = Portfolio(initial_capital=self.initial_capital)
        trades: list[dict] = []
        daily_rows: list[dict] = []

        for i, (date, row) in enumerate(data.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            executed_signal = None

            # Critical anti-look-ahead rule: today's holdings respond only to
            # yesterday's completed signal, never today's signal.
            if i > 0:
                executed_signal = int(signals.iloc[i - 1])
                position = portfolio.position
                is_long = position is not None and position.is_open

                if executed_signal == 1 and not is_long:
                    quantity = portfolio.cash / open_price
                    portfolio.position = Position(
                        symbol=symbol,
                        quantity=float(quantity),
                        entry_price=open_price,
                        entry_date=date,
                    )
                    portfolio.cash = 0.0

                elif executed_signal == 0 and is_long:
                    assert position is not None and position.entry_price is not None
                    proceeds = position.quantity * open_price
                    gross_pnl = (open_price - position.entry_price) * position.quantity
                    holding_days = (date - position.entry_date).days
                    trades.append(
                        {
                            "symbol": symbol,
                            "direction": "LONG",
                            "entry_date": position.entry_date,
                            "exit_date": date,
                            "entry_price": position.entry_price,
                            "exit_price": open_price,
                            "quantity": position.quantity,
                            "gross_pnl": gross_pnl,
                            "costs": 0.0,
                            "net_pnl": gross_pnl,
                            "return_pct": open_price / position.entry_price - 1.0,
                            "holding_period_days": holding_days,
                        }
                    )
                    portfolio.cash += proceeds
                    portfolio.position = None

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
