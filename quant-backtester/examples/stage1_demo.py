from __future__ import annotations

import argparse

from data.inspection import market_data_summary
from data.loader import MarketDataLoader
from features.returns import simple_returns
from visualization.charts import plot_price_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a cleaned OHLCV CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data, cleaning = MarketDataLoader.from_csv(args.csv_path)
    print("Cleaning report:", cleaning)
    print("Data summary:", market_data_summary(data))
    print("\nFirst rows with simple returns:")
    preview = data.copy()
    preview["Simple Return"] = simple_returns(data)
    print(preview.head())

    if args.plot:
        import matplotlib.pyplot as plt

        plot_price_history(data, symbol=args.symbol)
        plt.show()


if __name__ == "__main__":
    main()
