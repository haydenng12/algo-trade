from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analytics.drawdown import maximum_drawdown
from analytics.performance import cagr, equity_returns, total_return
from analytics.risk import annualized_volatility, sharpe_ratio
from dashboard.charts import cost_sensitivity_figure, drawdown_figure, normalized_equity_figure, weights_figure
from dashboard.research import cost_sensitivity_table, moving_average_parameter_grid
from dashboard.presentation import benchmark_comparison_table, configuration_rows, interpretation_text
from dashboard.runner import StrategyConfig, run_strategy
from data.loader import MarketDataLoader


st.set_page_config(page_title="Quant Backtester", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stMetricLabel"] {color: #9aa4b2;}
    div[data-testid="stMetricValue"] {font-size: 1.75rem;}
    .benchmark-note {color: #8b949e; font-size: 0.76rem; margin-top: -0.4rem;}
    .config-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 0.7rem 0.9rem; margin-bottom: 0.45rem;
    }
    .config-label {color: #8b949e; font-size: 0.76rem;}
    .config-value {color: #e6edf3; font-weight: 600; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=3600)
def load_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    frame, _ = MarketDataLoader.from_yahoo(symbol, start, end)
    return frame


def benchmark_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity_returns(equity)
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0
    years = days / 365.25
    return {
        "total_return": total_return(float(equity.iloc[0]), float(equity.iloc[-1])),
        "cagr": cagr(float(equity.iloc[0]), float(equity.iloc[-1]), years),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "maximum_drawdown": maximum_drawdown(equity),
    }


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def strategy_controls(name: str, universe_size: int) -> StrategyConfig:
    if name == "Moving Average Trend":
        fast = st.sidebar.number_input("Fast SMA", 2, 250, 20)
        slow = st.sidebar.number_input("Slow SMA", 3, 500, 50)
        if fast >= slow:
            st.sidebar.error("Fast SMA must be smaller than slow SMA.")
        return StrategyConfig(name, {"fast_window": fast, "slow_window": slow})
    if name == "Z-Score Mean Reversion":
        lookback = st.sidebar.number_input("Z-score lookback", 2, 250, 20)
        entry = st.sidebar.number_input("Entry Z-score", -5.0, -0.1, -1.5, 0.1)
        exit_z = st.sidebar.number_input("Exit Z-score", -4.0, 4.0, 0.0, 0.1)
        if entry >= exit_z:
            st.sidebar.error("Entry Z-score must be below exit Z-score.")
        return StrategyConfig(name, {"lookback": lookback, "entry_z": entry, "exit_z": exit_z})
    if name == "RSI Mean Reversion":
        lookback = st.sidebar.number_input("RSI lookback", 2, 100, 14)
        oversold = st.sidebar.slider("Oversold threshold", 1, 49, 30)
        exit_level = st.sidebar.slider("Exit RSI", 2, 100, 50)
        if oversold >= exit_level:
            st.sidebar.error("Oversold threshold must be below exit RSI.")
        return StrategyConfig(name, {"lookback": lookback, "oversold": oversold, "exit_level": exit_level})
    if name == "Cross-Sectional Momentum":
        lookback = st.sidebar.number_input("Momentum lookback (days)", 2, 504, 126)
        rebalance = st.sidebar.number_input("Rebalance every N days", 1, 126, 21)
        top_n = st.sidebar.number_input("Assets held", 1, max(1, universe_size), min(2, universe_size))
        return StrategyConfig(name, {"lookback": lookback, "rebalance_every": rebalance, "top_n": top_n})
    fast = st.sidebar.number_input("Fast SMA", 2, 250, 20, key="vol_fast")
    slow = st.sidebar.number_input("Slow SMA", 3, 500, 50, key="vol_slow")
    target = st.sidebar.slider("Target annual volatility", 0.02, 0.30, 0.10, 0.01)
    vol_lookback = st.sidebar.number_input("Volatility lookback", 2, 126, 20)
    if fast >= slow:
        st.sidebar.error("Fast SMA must be smaller than slow SMA.")
    return StrategyConfig(name, {
        "fast_window": fast,
        "slow_window": slow,
        "target_volatility": target,
        "vol_lookback": vol_lookback,
    })


st.title("Quantitative Strategy Research Dashboard")
st.caption("A modular daily-bar backtesting and research platform with next-period execution, trading friction, benchmarking, and risk analytics.")

st.sidebar.header("Backtest Configuration")
strategy_name = st.sidebar.selectbox(
    "Strategy",
    [
        "Moving Average Trend",
        "Z-Score Mean Reversion",
        "RSI Mean Reversion",
        "Cross-Sectional Momentum",
        "Volatility-Targeted Trend",
    ],
)
primary = st.sidebar.text_input("Primary symbol", "SPY").upper().strip()
universe_text = st.sidebar.text_input("Momentum universe", "SPY, QQQ, IWM, EFA")
universe = list(dict.fromkeys([s.strip().upper() for s in universe_text.split(",") if s.strip()]))
if primary and primary not in universe:
    universe.insert(0, primary)

start_date = st.sidebar.date_input("Start date", date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", date(2026, 1, 1))
capital = st.sidebar.number_input("Starting capital ($)", 1_000.0, 10_000_000.0, 100_000.0, 10_000.0)
fees = st.sidebar.number_input("Transaction fees (bps)", 0.0, 100.0, 5.0, 1.0)
slippage = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)
config = strategy_controls(strategy_name, len(universe))
run_clicked = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()
if not primary:
    st.error("Enter a primary ticker.")
    st.stop()
if strategy_name == "Cross-Sectional Momentum" and len(universe) < 2:
    st.error("Cross-sectional momentum requires at least two tickers.")
    st.stop()

if run_clicked or "dashboard_run" not in st.session_state:
    symbols_to_load = universe if strategy_name == "Cross-Sectional Momentum" else [primary]
    try:
        with st.spinner("Loading market data and running the backtest..."):
            frames = {s: load_symbol(s, str(start_date), str(end_date)) for s in symbols_to_load}
            run = run_strategy(
                frames,
                primary,
                config,
                initial_capital=capital,
                fees_bps=fees,
                slippage_bps=slippage,
            )
        st.session_state["dashboard_run"] = run
        st.session_state["dashboard_frames"] = frames
        st.session_state["dashboard_config"] = config
        st.session_state["dashboard_inputs"] = {
            "primary": primary, "capital": capital, "fees": fees, "slippage": slippage,
        }
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

run = st.session_state["dashboard_run"]
frames = st.session_state["dashboard_frames"]
active_config = st.session_state["dashboard_config"]
inputs = st.session_state["dashboard_inputs"]
benchmark = benchmark_metrics(run.benchmark_equity.dropna())

st.subheader(run.name)
st.caption(
    f"{run.equity.index[0].date()} → {run.equity.index[-1].date()} · "
    f"${inputs['capital']:,.0f} initial capital · {inputs['fees']:.1f} bps fees · "
    f"{inputs['slippage']:.1f} bps slippage · signals execute next trading day"
)

cols = st.columns(6)
metric_pairs = [
    ("Total Return", pct(float(run.summary["total_return"])), pct(benchmark["total_return"])),
    ("CAGR", pct(float(run.summary["cagr"])), pct(benchmark["cagr"])),
    ("Volatility", pct(float(run.summary["annualized_volatility"])), pct(benchmark["annualized_volatility"])),
    ("Sharpe", num(float(run.summary["sharpe_ratio"])), num(benchmark["sharpe_ratio"])),
    ("Sortino", num(float(run.summary["sortino_ratio"])), None),
    ("Max Drawdown", pct(float(run.summary["maximum_drawdown"])), pct(benchmark["maximum_drawdown"])),
]
for col, (label, value, bench_value) in zip(cols, metric_pairs):
    col.metric(label, value)
    if bench_value is not None:
        col.markdown(f'<div class="benchmark-note">Benchmark: {bench_value}</div>', unsafe_allow_html=True)

overview, risk_tab, research_tab = st.tabs(["Overview", "Risk & Trading", "Research"])

with overview:
    st.info(interpretation_text(run.summary, benchmark), icon="↔️")
    left, right = st.columns([2.25, 1], gap="large")
    with left:
        fig = normalized_equity_figure(run.equity, run.benchmark_equity, run.name, run.benchmark_name)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Benchmark")
        st.caption(run.benchmark_name)
        comparison = benchmark_comparison_table(run.summary, benchmark)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("#### Current Configuration")
        for label, value in configuration_rows(active_config.name, active_config.params):
            st.markdown(
                f'<div class="config-card"><div class="config-label">{label}</div>'
                f'<div class="config-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

with risk_tab:
    fig = drawdown_figure(run.drawdown)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    cost1, cost2, cost3 = st.columns(3)
    total_fees = float(run.costs.get("transaction_costs", pd.Series(dtype=float)).sum())
    total_slippage = float(run.costs.get("slippage_costs", pd.Series(dtype=float)).sum())
    total_notional = float(run.costs.get("traded_notional", pd.Series(dtype=float)).sum())
    cost1.metric("Transaction Fees", f"${total_fees:,.2f}")
    cost2.metric("Estimated Slippage", f"${total_slippage:,.2f}")
    cost3.metric("Traded Notional", f"${total_notional:,.0f}")

    if run.trades is not None:
        st.markdown("#### Completed Trades")
        if run.trades.empty:
            st.info("No completed trades in this period.")
        else:
            trade_view = run.trades.copy()
            for c in ["entry_date", "exit_date"]:
                if c in trade_view:
                    trade_view[c] = pd.to_datetime(trade_view[c]).dt.date
            st.dataframe(trade_view, use_container_width=True, hide_index=True)
    if run.target_weights is not None:
        st.markdown("#### Portfolio Allocation")
        fig = weights_figure(run.target_weights)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.dataframe(run.target_weights.tail(20).style.format("{:.1%}"), use_container_width=True)

with research_tab:
    st.markdown("#### Cost Sensitivity")
    st.caption("Reruns the exact active strategy while changing fees and slippage together. Everything else stays fixed.")
    levels = st.multiselect("Cost levels (bps each)", [0.0, 2.5, 5.0, 10.0, 25.0, 50.0], default=[0.0, 5.0, 10.0, 25.0])
    if levels:
        sensitivity = cost_sensitivity_table(
            frames,
            inputs["primary"],
            active_config,
            sorted(levels),
            initial_capital=inputs["capital"],
        )
        fig = cost_sensitivity_figure(sensitivity)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        display = sensitivity.copy()
        display["total_return"] = display["total_return"].map(lambda x: f"{x:.2%}")
        display["maximum_drawdown"] = display["maximum_drawdown"].map(lambda x: f"{x:.2%}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    if active_config.name == "Moving Average Trend":
        st.divider()
        st.markdown("#### Moving-Average Parameter Surface")
        st.caption("Exploratory full-period analysis only. Do not treat the best row here as out-of-sample evidence.")
        fast_values = st.multiselect("Fast windows", [5, 10, 20, 30, 40, 50], default=[10, 20, 40])
        slow_values = st.multiselect("Slow windows", [30, 50, 75, 100, 150, 200], default=[50, 100, 200])
        if fast_values and slow_values:
            grid = moving_average_parameter_grid(
                inputs["primary"], frames[inputs["primary"]], fast_values, slow_values,
                initial_capital=inputs["capital"], fees_bps=inputs["fees"], slippage_bps=inputs["slippage"],
            )
            st.dataframe(
                grid.style.format({
                    "total_return": "{:.2%}", "cagr": "{:.2%}",
                    "sharpe_ratio": "{:.2f}", "maximum_drawdown": "{:.2%}",
                }).background_gradient(subset=["sharpe_ratio"]),
                use_container_width=True,
                hide_index=True,
            )

st.caption("Stage 7 · Research dashboard. Historical backtests are not predictions of future performance.")
