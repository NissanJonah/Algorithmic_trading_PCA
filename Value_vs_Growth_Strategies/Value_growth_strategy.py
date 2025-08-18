import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
period = "20y"
lookback_mode = "rolling"  # "rolling" or "since_start"
lookback_days = 252  # only used if lookback_mode == "rolling"
cost_bps = 2  # transaction cost per trade side (basis points, e.g. 2 = 0.02%)

# --------------------------
# Data
# --------------------------
tickers = ["VUG", "VTV", "SPY"]
data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"].dropna()
ret = data.pct_change()

# --------------------------
# Build signal: Growth > Value ?
# --------------------------
if lookback_mode == "since_start":
    vug_norm = data["VUG"] / data["VUG"].iloc[0]
    vtv_norm = data["VTV"] / data["VTV"].iloc[0]
elif lookback_mode == "rolling":
    # Normalize relative to rolling start of window to reduce start-date bias
    vug_norm = data["VUG"] / data["VUG"].rolling(lookback_days).apply(lambda x: x[0], raw=False)
    vtv_norm = data["VTV"] / data["VTV"].rolling(lookback_days).apply(lambda x: x[0], raw=False)

condition = vug_norm > vtv_norm

# --------------------------
# Positions: long SPY if Growth > Value, else cash
# --------------------------
positions = pd.Series(0, index=data.index)
positions[condition] = 1
positions = positions.shift(1).fillna(0)  # trade on next day

# --------------------------
# Strategy returns
# --------------------------
spy_ret = ret["SPY"]
strat_ret = positions * spy_ret

# Apply transaction costs when position changes
trade_changes = positions.diff().abs()
strat_ret -= trade_changes * (cost_bps / 10000.0)

# Cumulative returns
strat_cum = (1 + strat_ret).cumprod()
spy_cum = (1 + spy_ret).cumprod()


# --------------------------
# Performance metrics
# --------------------------
def performance_report(strat_ret, strat_cum, spy_cum, name="Strategy"):
    total_days = len(strat_ret)
    cagr = (strat_cum.iloc[-1]) ** (252 / total_days) - 1
    vol = strat_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    dd = (strat_cum / strat_cum.cummax() - 1).min()

    trades = trade_changes.sum()
    win_rate = (strat_ret[strat_ret != 0] > 0).mean() * 100

    print(f"\n{name} Results:")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"Volatility: {vol * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {dd * 100:.2f}%")
    print(f"Number of trades: {int(trades)}")
    print(f"Win rate (days >0): {win_rate:.2f}%")
    print(f"Final multiple: {strat_cum.iloc[-1]:.2f}x")
    print(f"SPY Final multiple: {spy_cum.iloc[-1]:.2f}x")


performance_report(strat_ret, strat_cum, spy_cum, name="Growth>Value Long-Only Strategy")

# --------------------------
# Plot results
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(strat_cum.index, strat_cum, label="Strategy")
plt.plot(spy_cum.index, spy_cum, label="SPY Buy & Hold")
plt.title("Cumulative Returns")
plt.ylabel("Growth of $1")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.show()
