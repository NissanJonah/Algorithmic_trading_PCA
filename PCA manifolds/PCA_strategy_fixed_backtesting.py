# full_manifold_network_pca_strategy_enhanced.py
# Full pipeline implementing manifold + network-weighted PCA strategy with weekly rebalancing and additional analytics.
# Requirements: numpy, pandas, yfinance, scipy, statsmodels, matplotlib, scikit-learn
# Run in a Python 3.8+ environment.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------
# Configuration / Parameters
# ---------------------------
stocks = [
    'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
    ' COP', 'EOG', 'DVN', 'APA',
    'MPC', 'PSX', 'VLO', 'PBF', 'DK',
    'KMI', 'WMB', 'OKE', 'ET', 'ENB',
    'SLB', 'HAL', 'BKR', 'FTI', 'NOV',
    'FANG', 'HES', 'CTRA'
]

factors_tickers = [
    'XLE', 'XOP', 'OIH', 'VDE', 'IXC',
    'CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F',
    'ICLN', 'TAN', 'FAN', 'PBW', 'QCLN',
    'CRAK', 'PXE', 'FCG', 'MLPX', 'AMLP',
    'FENY', 'OILK', 'USO', 'BNO', 'UNG',
    '^SP500-15', '^DJUSEN', '^XOI', '^OSX',
    'ENOR', 'ENZL', 'KWT', 'GEX', 'URA',
    'RSPG', '^TNX', '^VIX', 'COAL', 'URA',
    'XES', 'IEO', 'PXI', 'TIP', 'GLD'
]

# Which factors we treat as "levels" (changes rather than returns)
level_factors = ['^VIX', '^TNX']

# Lookback & rebalancing
LOOKBACK_DAYS = 252   # rolling window for PCA/regressions
REBALANCE_FREQ = 'W-FRI'  # weekly on Friday
CORR_THRESHOLD = 0.6  # adjacency threshold for network
N_PCS = 6  # number of principal components to use

# *** CAPITAL *** use dollars
CAPITAL = 10000.0  # money put in
LONG_TARGET_SHARE = 0.65  # 65% long, 35% short
BETA_TOL = 0.10  # allowed beta tolerance (in absolute terms)
MAX_LONG_POS = 0.15  # max long as fraction of capital
MAX_SHORT_POS = 0.12  # max short as fraction of capital
MAX_SHORT_CONC_AS_FRACTION_OF_SHORTS = 0.25  # max single short as fraction of total shorts
CORRELATION_NETWORK_THRESHOLD = CORR_THRESHOLD

# Backtest dates
end_date = '2025-08-17'
start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=365 * 6)).strftime('%Y-%m-%d')  # buffer

# Market proxy for beta
MARKET_PROXY = 'XLE'

# ---------------------------
# Helper functions
# ---------------------------

def safe_select_close(df):
    """Take yfinance result and extract a single 'Close' price-level DataFrame regardless of MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        top_levels = list(df.columns.levels[0])
        if 'Close' in top_levels:
            out = df['Close'].copy()
        elif 'Adj Close' in top_levels:
            out = df['Adj Close'].copy()
        else:
            # fallback: collapse to tickers (last level)
            out = df.copy()
            out.columns = [c[-1] for c in df.columns]
    else:
        out = df.copy()
    out.columns.name = None
    return out

def compute_eigenvector_centrality(corr_mat, threshold=0.6):
    """Construct adjacency by thresholding absolute correlation, then compute principal eigenvector (centrality)."""
    A = (np.abs(corr_mat) >= threshold).astype(float)
    np.fill_diagonal(A.values, 1.0)
    vals, vecs = np.linalg.eig(A.values)
    idx = np.nanargmax(vals.real)
    v = vecs[:, idx].real
    v = np.abs(v)
    if v.sum() == 0:
        # fallback uniform
        v = np.ones_like(v)
    v = v / v.sum()
    return pd.Series(v, index=corr_mat.index)

def pca_from_returns(returns_df, n_components=6):
    """Return eigenvectors (loadings) Q (stocks x components), eigenvalues, and PC time series (T x k)."""
    R = returns_df.corr().fillna(0)
    vals, vecs = np.linalg.eigh(R.values)  # ascending
    order = np.argsort(vals)[::-1]
    vals = vals[order][:n_components]
    vecs = vecs[:, order][:, :n_components]
    Q = pd.DataFrame(vecs, index=R.index, columns=[f'PC{i+1}' for i in range(len(vals))])
    PC_ts = pd.DataFrame(returns_df.values.dot(Q.values), index=returns_df.index, columns=Q.columns)
    return Q, vals, PC_ts

def regress_pc_on_factors(PC_series, factor_df, factors_list):
    """
    Regress PC_series on factor changes/returns.
    Returns alpha, betas (Series indexed by factor names), R2
    """
    X_cols = {}
    for f in factors_list:
        if f in level_factors:
            x = factor_df[f].diff().rename(f)
        else:
            x = factor_df[f].pct_change().rename(f)
        X_cols[f] = x
    X = pd.concat(X_cols.values(), axis=1).dropna()
    y = PC_series.reindex(X.index).dropna()
    X = X.loc[y.index]
    if len(y) < 10:
        return 0.0, pd.Series(0.0, index=factors_list), 0.0
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    alpha = model.params['const']
    betas = model.params.drop('const')
    return alpha, betas.reindex(factors_list).fillna(0.0), model.rsquared

def optimize_portfolio(expected_adj_r, betas_vec, capital=10000.0):
    """
    Optimize with variables v_plus (long allocations) and v_minus (short allocations).
    w = v_plus - v_minus
    Objective: maximize expected_adj_r dot w  (minimize negative)
    Constraints implemented as in your specification (65/35, position limits, beta tolerance, short concentration).
    All allocations are dollar amounts.
    """
    expected_adj_r = expected_adj_r.copy().fillna(0.0)
    n = len(expected_adj_r)
    if n == 0:
        return pd.Series([], index=expected_adj_r.index)
    exp_r = expected_adj_r.values.astype(float)
    b = betas_vec.reindex(expected_adj_r.index).fillna(0.0).values.astype(float)

    # bounds
    long_bounds = [(0.0, MAX_LONG_POS * capital)] * n
    short_bounds = []
    for er in exp_r:
        if er < -1e-12:
            short_bounds.append((0.0, MAX_SHORT_POS * capital))
        else:
            short_bounds.append((0.0, 0.0))  # disallow short
    bounds = long_bounds + short_bounds

    # initial guess
    vplus0 = np.zeros(n)
    vminus0 = np.zeros(n)
    eligible_longs = [i for i in range(n) if exp_r[i] >= 0]
    eligible_shorts = [i for i in range(n) if exp_r[i] < 0]
    total_long_cap = LONG_TARGET_SHARE * capital
    total_short_cap = capital - total_long_cap
    if eligible_longs:
        share = total_long_cap / len(eligible_longs)
        for i in eligible_longs:
            vplus0[i] = min(share, MAX_LONG_POS * capital)
    if eligible_shorts:
        share = total_short_cap / len(eligible_shorts)
        for i in eligible_shorts:
            vminus0[i] = min(share, MAX_SHORT_POS * capital)
    x0 = np.concatenate([vplus0, vminus0])

    def obj(x):
        v_plus = x[:n]
        v_minus = x[n:]
        w = v_plus - v_minus
        return -float(np.dot(exp_r, w))

    cons = []
    # capital constraint
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x) - capital})
    # long/short ratio constraint
    def long_short_ratio_constr(x):
        v_plus = x[:n]
        v_minus = x[n:]
        long = v_plus.sum()
        short = v_minus.sum()
        return long - LONG_TARGET_SHARE * (long + short)
    cons.append({'type': 'eq', 'fun': long_short_ratio_constr})
    # beta tolerance -> two inequalities: BETA_TOL - w·b >=0 and BETA_TOL + w·b >=0
    cons.append({'type': 'ineq', 'fun': lambda x: BETA_TOL - np.dot((x[:n] - x[n:]), b)})
    cons.append({'type': 'ineq', 'fun': lambda x: BETA_TOL + np.dot((x[:n] - x[n:]), b)})

    # short concentration: each v_minus_i <= max_single_short_abs
    total_short_cap = (1 - LONG_TARGET_SHARE) * capital
    max_single_short_abs = MAX_SHORT_CONC_AS_FRACTION_OF_SHORTS * total_short_cap
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': (lambda idx: (lambda x: max_single_short_abs - x[n + idx]))(i)})

    # Solve
    result = minimize(obj, x0, bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-9})
    if not result.success:
        # fallback: proportional allocation from x0
        v_plus = vplus0
        v_minus = vminus0
    else:
        v_plus = result.x[:n]
        v_minus = result.x[n:]
    w = pd.Series(v_plus - v_minus, index=expected_adj_r.index)
    w[np.abs(w) < 1e-12] = 0.0
    return w

# ---------------------------
# Data download & preprocessing
# ---------------------------
print("Downloading price data (this may take a bit)...")
all_tickers = list(set(stocks + factors_tickers))
raw_prices = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)

# split into stock_data and factor_data by available tickers
stock_data_raw = raw_prices[[c for c in raw_prices.columns if (isinstance(c, str) and c in stocks) or (isinstance(c, tuple) and c[-1] in stocks)]].copy()
factor_data_raw = raw_prices[[c for c in raw_prices.columns if (isinstance(c, str) and c in factors_tickers) or (isinstance(c, tuple) and c[-1] in factors_tickers)]].copy()

stock_data = safe_select_close(stock_data_raw)
factor_data = safe_select_close(factor_data_raw)

missing_stocks = [s for s in stocks if s not in stock_data.columns]
if missing_stocks:
    print("Warning - the following stocks are missing from data and will be removed:", missing_stocks)
    stocks = [s for s in stocks if s in stock_data.columns]
    stock_data = stock_data[stocks]

missing_factors = [f for f in factors_tickers if f not in factor_data.columns]
if missing_factors:
    print("Warning - the following factors missing and will be ignored:", missing_factors)
    factors_tickers = [f for f in factors_tickers if f in factor_data.columns]
    factor_data = factor_data[factors_tickers]

# compute daily returns for stocks and keep factor price-levels for computing diffs/returns as needed
stock_returns_all = stock_data.pct_change().dropna(how='all')
# Align the index across stocks and factors (intersection)
common_index = stock_returns_all.index.intersection(factor_data.index)
stock_data = stock_data.reindex(common_index)
factor_data = factor_data.reindex(common_index)
stock_returns_all = stock_data.pct_change().dropna(how='all')
print("Data ready. Stocks:", len(stocks), "Date range:", stock_data.index.min(), "to", stock_data.index.max())

# ---------------------------
# Backtest: weekly rolling window
# ---------------------------
weekly_index = stock_data.resample(REBALANCE_FREQ).last().index
# Only keep weeks where we have at least LOOKBACK_DAYS prior
start_idx = np.searchsorted(weekly_index, stock_data.index[0] + pd.Timedelta(days=LOOKBACK_DAYS))
weekly_index = weekly_index[start_idx:]

out_of_sample_returns = []  # realized portfolio returns (dollars normalized to CAPITAL)
weights_history = []
dates_history = []
trade_log = []  # To store trade details
pc0_pred_history = []  # Predicted PC0 values
pc0_actual_history = []  # Actual PC0 values
pc0_r2_history = []  # R² values for PC0 predictions

# diagnostics histories
avg_long_expected = []
avg_short_expected = []
avg_long_weight = []
avg_short_weight = []
centrality_history = []
long_pnl_history = []
short_pnl_history = []
total_pnl_history = []

print(f"Starting weekly backtest on {len(weekly_index)} weeks. This may take a minute...")

for t_idx, week_end in enumerate(weekly_index):
    # find nearest trading day <= week_end
    if week_end not in stock_returns_all.index:
        try:
            week_end = stock_returns_all.index[stock_returns_all.index.get_indexer([week_end], method='pad')[0]]
        except Exception:
            continue
    window_end = week_end
    window_start_idx = stock_returns_all.index.get_loc(window_end) - (LOOKBACK_DAYS - 1)
    if window_start_idx < 0:
        continue
    window_start = stock_returns_all.index[window_start_idx]
    window_returns = stock_returns_all.loc[window_start:window_end].dropna(axis=1, how='all')

    valid_cols = window_returns.columns[window_returns.notna().sum() >= int(0.6 * LOOKBACK_DAYS)]
    window_returns = window_returns[valid_cols]
    if window_returns.shape[1] < 4:
        continue

    # 1) centrality
    corr_mat = window_returns.corr().fillna(0.0)
    centrality = compute_eigenvector_centrality(corr_mat, threshold=CORRELATION_NETWORK_THRESHOLD)
    centrality = centrality.reindex(window_returns.columns).fillna(0.0)

    # 2) PCA on window
    Q, eigvals, PC_ts = pca_from_returns(window_returns, n_components=N_PCS)
    pc_std = PC_ts.std(ddof=1)

    # 3) regress PCs on a chosen subset of factors
    regression_factors = [f for f in factor_data.columns if f in level_factors or f in ['XLE', 'VDE', 'USO', 'BZ=F', 'CL=F', '^VIX', '^TNX']]
    regression_factors = [f for f in regression_factors if f in factor_data.columns]
    factor_window = factor_data.loc[window_start:window_end, regression_factors].copy()

    pc_alphas = {}
    pc_betas = {}
    pc_r2 = {}
    for pc in PC_ts.columns:
        alpha, betas, r2 = regress_pc_on_factors(PC_ts[pc], factor_window, regression_factors)
        pc_alphas[pc] = alpha
        pc_betas[pc] = betas
        pc_r2[pc] = r2

    # 4) forecast factor weekly change using last weekly change as a naive forecast
    factor_weekly = factor_data[regression_factors].resample(REBALANCE_FREQ).last().loc[:week_end]
    if len(factor_weekly) < 2:
        continue
    factor_weekly_changes = pd.DataFrame(index=factor_weekly.index)
    for f in regression_factors:
        if f in level_factors:
            factor_weekly_changes[f] = factor_weekly[f].diff()
        else:
            factor_weekly_changes[f] = factor_weekly[f].pct_change()
    delta_factors_forecast = factor_weekly_changes.iloc[-1].fillna(0.0)

    # 5) Predict ΔPC_j = α_j + β_j dot Δfactor_forecast -> then percent movement = ΔPC * sigma_PC
    pc_pred_moves = {}
    for pc in PC_ts.columns:
        alpha = pc_alphas.get(pc, 0.0)
        betas = pc_betas.get(pc, pd.Series(0.0, index=regression_factors))
        delta_pc = alpha + np.dot(betas.reindex(delta_factors_forecast.index).fillna(0.0).values,
                                  delta_factors_forecast.values)
        pc_pred_pct = delta_pc * pc_std.get(pc, 0.0)
        pc_pred_moves[pc] = pc_pred_pct
    pc_pred_moves = pd.Series(pc_pred_moves)

    # Store PC0 predicted and actual for regression plot
    if 'PC1' in PC_ts.columns:
        pc0_pred = pc_pred_moves.get('PC1', 0.0)
        pc0_actual = PC_ts['PC1'].iloc[-1] if not PC_ts['PC1'].empty else 0.0
        pc0_pred_history.append(pc0_pred)
        pc0_actual_history.append(pc0_actual)
        # Compute R² for this week's prediction
        if len(PC_ts['PC1']) > 5:
            lr = LinearRegression()
            lr.fit(PC_ts['PC1'].values[:-1].reshape(-1, 1), PC_ts['PC1'].shift(-1).values[:-1])
            pred = lr.predict([[pc0_actual]])
            r2 = r2_score([PC_ts['PC1'].iloc[-1]], [pred[0]]) if not np.isnan(pred[0]) else 0.0
            pc0_r2_history.append(r2)
        else:
            pc0_r2_history.append(0.0)

    # 6) Convert to stock-level expected returns: r_i = sum_j Q_ij * PC_j_movement
    Q_aligned = Q.reindex(window_returns.columns).fillna(0.0)
    pc_vector = pc_pred_moves.reindex(Q_aligned.columns).fillna(0.0).values
    r_from_pcs = pd.Series(Q_aligned.values.dot(pc_vector), index=Q_aligned.index)

    # 7) Weight by centrality
    expected_adj_r = r_from_pcs * centrality.reindex(r_from_pcs.index).fillna(0.0)

    # 8) Beta calc relative to MARKET_PROXY (XLE)
    if MARKET_PROXY not in factor_data.columns:
        raise ValueError(f"Market proxy {MARKET_PROXY} not in downloaded factors.")
    market_series = factor_data[MARKET_PROXY].loc[window_start:window_end]
    market_returns = market_series.pct_change().loc[window_returns.index].dropna()
    aligned_sr, aligned_mr = window_returns.align(market_returns, join='inner', axis=0)
    aligned_sr = aligned_sr.dropna(axis=1, how='all')
    if len(aligned_mr) <= 1:
        betas = pd.Series(0.0, index=expected_adj_r.index)
    else:
        betas = aligned_sr.apply(lambda x: np.cov(x, aligned_mr)[0, 1] / np.var(aligned_mr) if len(aligned_mr) > 1 else 0.0)
    common_universe = expected_adj_r.index.intersection(betas.index)
    expected_adj_r = expected_adj_r.reindex(common_universe).fillna(0.0)
    betas = betas.reindex(common_universe).fillna(0.0)

    # 9) Optimization -> dollar weights (sum absolute allocations = capital split long/short internally)
    w = optimize_portfolio(expected_adj_r, betas, capital=CAPITAL)

    # Diagnostics
    longs = w[w > 0]
    shorts = -w[w < 0]
    avg_long_expected.append(expected_adj_r.loc[longs.index].mean() if len(longs) else 0.0)
    avg_short_expected.append(expected_adj_r.loc[shorts.index].mean() if len(shorts) else 0.0)
    avg_long_weight.append(longs.sum() if len(longs) else 0.0)
    avg_short_weight.append(shorts.sum() if len(shorts) else 0.0)

    # 10) Compute next-week realized return (dollar PnL) and log trades
    next_idx = t_idx + 1
    if next_idx >= len(weekly_index):
        break
    next_week_end = weekly_index[next_idx]
    if next_week_end not in stock_data.index:
        next_week_end = stock_data.index[stock_data.index.get_indexer([next_week_end], method='pad')[0]]

    price_t = stock_data.loc[window_end, common_universe]
    price_t1 = stock_data.loc[next_week_end, common_universe]
    realized = (price_t1 / price_t - 1).fillna(0.0)

    # w currently indexed by expected_adj_r.index which is common_universe; ensure alignment
    w = w.reindex(common_universe).fillna(0.0)

    # realized pnl in dollars: sum(w * realized)
    realized_pnl = float(np.dot(w.values, realized.values))
    # longs and shorts breakdown
    realized_long_pnl = float((w[w > 0].values * realized[w > 0].values).sum()) if (w > 0).any() else 0.0
    realized_short_pnl_signed = float((w[w < 0].values * realized[w < 0].values).sum()) if (w < 0).any() else 0.0
    # Present short profit as positive if profitable:
    realized_short_profit = -realized_short_pnl_signed

    # Log trades for profitable ones
    for stock in w.index:
        if w[stock] != 0:
            shares = w[stock] / price_t[stock] if price_t[stock] != 0 else 0
            position_value = abs(w[stock])
            profit = shares * (price_t1[stock] - price_t[stock]) if w[stock] > 0 else -shares * (price_t1[stock] - price_t[stock])
            # if profit > 0:  # Only log profitable trades
            trade_log.append({
                'Date Enter': window_end,
                'Date Exit': next_week_end,
                'Stock': stock,
                'Position': 'Long' if w[stock] > 0 else 'Short',
                'Entry Price': price_t[stock],
                'Exit Price': price_t1[stock],
                'Shares': shares,
                'Initial Value': position_value,
                'Profit': profit,
                'Total Value': position_value + profit
            })

    # Append histories only if we have a valid portfolio and next week's data
    centrality_history.append(centrality.reindex(stocks).fillna(0.0))
    out_of_sample_returns.append(realized_pnl / CAPITAL)  # store normalized return for stats
    long_pnl_history.append(realized_long_pnl)
    short_pnl_history.append(realized_short_profit)
    total_pnl_history.append(realized_pnl)
    weights_history.append(w)
    dates_history.append(next_week_end)

# ---------------------------
# Results & Diagnostics
# ---------------------------
if len(out_of_sample_returns) == 0:
    raise RuntimeError("No out-of-sample weeks were produced; check data and lookback settings.")

port_rets = pd.Series(out_of_sample_returns, index=dates_history)  # normalized to CAPITAL (fractional)
cum = (1 + port_rets).cumprod()
portfolio_values = CAPITAL * cum  # dollar portfolio values

annualized_return = (1 + port_rets.mean()) ** 52 - 1
annualized_vol = port_rets.std() * np.sqrt(52)
sharpe = annualized_return / (annualized_vol if annualized_vol > 0 else np.nan)
total_return = portfolio_values.iloc[-1] - CAPITAL
total_return_pct = total_return / CAPITAL
rolling_max = portfolio_values.cummax()
drawdown = (portfolio_values - rolling_max) / rolling_max
max_dd = drawdown.min()

# aggregated long/short PnL
total_long_pnl = np.nansum(long_pnl_history)
total_short_pnl = np.nansum(short_pnl_history)
total_pnl = np.nansum(total_pnl_history)

# Build weights DataFrame
weights_df = pd.DataFrame(weights_history, index=dates_history).fillna(0.0).sort_index()

# Build centrality DataFrame
centrality_df = pd.DataFrame(centrality_history, index=dates_history).fillna(0.0).sort_index()

# average hold period (weeks -> days)
def contiguous_runs_length(seq, match_val):
    runs = []
    cur = 0
    for v in seq:
        if v == match_val:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs

long_runs = []
short_runs = []
for col in weights_df.columns:
    sgn = np.sign(weights_df[col].values)
    long_runs += contiguous_runs_length(sgn, 1)
    short_runs += contiguous_runs_length(sgn, -1)
all_runs = long_runs + short_runs
avg_hold_weeks = np.nan
if len(all_runs) > 0:
    avg_hold_weeks = np.mean(all_runs)
avg_hold_days = avg_hold_weeks * 7 if not np.isnan(avg_hold_weeks) else np.nan

# Print summary
print("==== Backtest summary ====")
print("Weeks:", len(port_rets))
print(f"Money put in: ${CAPITAL:,.2f}")
print(f"Total profit: ${total_return:,.2f} ({total_return_pct:.2%})")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized vol: {annualized_vol:.2%}")
print(f"Sharpe (rfr=0): {sharpe:.2f}")
print(f"Max drawdown: {max_dd:.2%}")
print(f"Total long PnL (dollars): ${total_long_pnl:,.2f}")
print(f"Total short PnL (dollars, positive means profitable): ${total_short_pnl:,.2f}")
print(f"Average hold period: {avg_hold_days:.1f} days (approx)")
print(f"Rebalancing frequency: weekly ({REBALANCE_FREQ})")
print("==========================")

# Print profitable trades
print("\n==== Profitable Trades ====")
trade_df = pd.DataFrame(trade_log)
if not trade_df.empty:
    print(trade_df[['Date Enter', 'Date Exit', 'Stock', 'Position', 'Entry Price', 'Exit Price', 'Shares', 'Initial Value', 'Profit', 'Total Value']].to_string(index=False))
else:
    print("No profitable trades recorded.")
print("==========================")

# show last weights
print("\nSample weights (last rebalance) - top exposures:")
print(weights_df.iloc[-1].sort_values(ascending=False).head(20))

# ---------------------------
# Extra analytics & plots
# ---------------------------

# 1) cumulative returns with drawdown
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values.index, portfolio_values.values, label='Portfolio value ($)')
plt.fill_between(drawdown.index, portfolio_values.min(), portfolio_values.max(),
                 where=drawdown < 0, color='red', alpha=0.05, label='Drawdown')
plt.title('Portfolio Value and Drawdown')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# 2) PCA on full sample (global) for plotting PCs over time
returns_for_global = stock_returns_all.dropna(how='any')
if returns_for_global.shape[0] >= 20 and returns_for_global.shape[1] >= 2:
    Q_full, eigvals_full, PC_ts_full = pca_from_returns(returns_for_global, n_components=min(N_PCS, returns_for_global.shape[1]-1))
    # plot first 3 PCs
    to_plot = min(3, PC_ts_full.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(to_plot):
        plt.plot(PC_ts_full.index, PC_ts_full.iloc[:, i], label=f'PC{i+1}')
    plt.title('Global PCA Components Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    PC_ts_full = None
    print("Not enough data for global PCA plotting.")

# 3) PC0 regression overlay
if len(pc0_pred_history) > 0 and len(pc0_actual_history) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(dates_history, pc0_pred_history, label='Predicted PC0', color='red')
    plt.plot(dates_history, pc0_actual_history, label='Actual PC0', color='blue')
    plt.title('PC0 Predicted vs Actual Over Time')
    plt.xlabel('Date')
    plt.ylabel('PC0 Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute overall R²
    if len(pc0_pred_history) > 1:
        r2_overall = r2_score(pc0_actual_history, pc0_pred_history)
        print(f"Overall R² between predicted and actual PC0: {r2_overall:.4f}")

# 4) R² over time
if len(pc0_r2_history) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(dates_history, pc0_r2_history, label='PC0 Prediction R²')
    plt.title('PC0 Prediction R² Over Time')
    plt.xlabel('Date')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5) Buy/Short signals over profit line
cum_pnl = np.cumsum(total_pnl_history)
new_longs = ((weights_df > 0) & (weights_df.shift(1).fillna(0.0) <= 0)).sum(axis=1)
new_shorts = ((weights_df < 0) & (weights_df.shift(1).fillna(0.0) >= 0)).sum(axis=1)

plt.figure(figsize=(12, 6))
plt.plot(dates_history, cum_pnl, label='Cumulative Profit ($)', color='black')
plt.scatter(dates_history, cum_pnl * (new_longs > 0), color='green', label='New Longs', marker='^', s=100)
plt.scatter(dates_history, cum_pnl * (new_shorts > 0), color='red', label='New Shorts', marker='v', s=100)
plt.title('Cumulative Profit with Buy/Short Signals')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit ($)')
plt.legend()
plt.grid(True)
plt.show()

# 6) Portfolio composition over time
if not weights_df.empty:
    # Normalize weights for plotting (stacked area plot)
    long_weights = weights_df.where(weights_df > 0, 0)
    short_weights = -weights_df.where(weights_df < 0, 0)
    # Scale to fit 65% longs, 35% shorts
    long_weights_scaled = long_weights / long_weights.max().max() * 0.65 if long_weights.max().max() != 0 else long_weights
    short_weights_scaled = short_weights / short_weights.max().max() * 0.35 if short_weights.max().max() != 0 else short_weights

    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot longs (positive, above midline at 0)
    ax.stackplot(dates_history, long_weights_scaled.T, labels=long_weights.columns, alpha=0.8)
    # Plot shorts (negative, below midline)
    ax.stackplot(dates_history, -short_weights_scaled.T, labels=[f"{s} (Short)" for s in short_weights.columns], alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', label='Midline')
    ax.set_title('Portfolio Composition Over Time (Longs 65%, Shorts 35%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Scaled Position Size')
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.grid(True)
    plt.show()

# 7) centrality over time: show top N stocks by average centrality
if not centrality_df.empty:
    avg_cent = centrality_df.mean().sort_values(ascending=False)
    topN = min(10, len(avg_cent))
    top_stocks_by_cent = list(avg_cent.index[:topN])
    plt.figure(figsize=(12, 6))
    for s in top_stocks_by_cent:
        plt.plot(centrality_df.index, centrality_df[s], label=s)
    plt.title('Centrality over time (top stocks)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 8) weights heatmap for top stocks by avg abs exposure
if not weights_df.empty:
    avg_abs_weight = weights_df.abs().mean().sort_values(ascending=False)
    top_m = min(20, len(avg_abs_weight))
    top_stocks_by_weight = list(avg_abs_weight.index[:top_m])
    weights_sub = weights_df[top_stocks_by_weight]
    plt.figure(figsize=(12, max(4, top_m * 0.25)))
    plt.imshow(weights_sub.T, aspect='auto', cmap='seismic', interpolation='nearest')
    plt.colorbar(label='Dollar weight')
    plt.yticks(range(len(top_stocks_by_weight)), top_stocks_by_weight)
    plt.xticks(range(len(weights_sub.index)), [d.strftime('%Y-%m-%d') for d in weights_sub.index], rotation=90)
    plt.title('Weights heatmap (rows=stocks, cols=weeks)')
    plt.show()

# 9) Long vs Short weekly PnL
plt.figure(figsize=(12, 6))
plt.plot(dates_history, long_pnl_history, label='Long PnL ($)')
plt.plot(dates_history, short_pnl_history, label='Short PnL ($)')
plt.title('Weekly Long vs Short PnL ($)')
plt.legend()
plt.grid(True)
plt.show()

# 10) Weekly total PnL bar
plt.figure(figsize=(12, 5))
plt.bar(dates_history, total_pnl_history)
plt.title('Weekly Total PnL ($)')
plt.grid(True)
plt.show()

# 11) New long / new short counts per week (signal activity)
if not weights_df.empty:
    shifted = weights_df.shift(1).fillna(0.0)
    new_longs = ((weights_df > 0) & (shifted <= 0)).sum(axis=1)
    new_shorts = ((weights_df < 0) & (shifted >= 0)).sum(axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(new_longs.index, new_longs.values, label='New longs (count)')
    plt.plot(new_shorts.index, new_shorts.values, label='New shorts (count)')
    plt.title('New long / short signals per rebalance')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------
# PC -> Factor mapping table (global)
# ---------------------------
print("\nPC -> Factor mapping (global):")
pc_factor_rows = []
if PC_ts_full is not None:
    # build daily factor changes/returns DataFrame (for factors that exist)
    factor_changes_daily = pd.DataFrame(index=factor_data.index)
    for f in factor_data.columns:
        if f in level_factors:
            factor_changes_daily[f] = factor_data[f].diff()
        else:
            factor_changes_daily[f] = factor_data[f].pct_change()
    candidate_factors = list(factor_changes_daily.columns)
    for pc in PC_ts_full.columns:
        best_r2 = -1.0
        best_f = None
        best_lr = None
        best_r2_reg = None
        pc_series = PC_ts_full[pc].dropna()
        for f in candidate_factors:
            fac_series = factor_changes_daily[f].dropna()
            a, b = pc_series.align(fac_series, join='inner')
            if len(a) < 20:
                continue
            # measure R^2 (correlation squared) as baseline
            r2_base = np.corrcoef(a, b)[0, 1] ** 2
            # linear regression
            lr = LinearRegression().fit(b.values.reshape(-1, 1), a.values)
            pred = lr.predict(b.values.reshape(-1, 1))
            r2_reg = r2_score(a.values, pred)
            if r2_base > best_r2:
                best_r2 = r2_base
                best_f = f
                best_lr = lr
                best_r2_reg = r2_reg
        if best_f is not None:
            eq = f"PC = {best_lr.coef_[0]:.6f}*{best_f} + {best_lr.intercept_:.6f}"
            pc_factor_rows.append([pc, best_f, best_r2, eq, best_r2_reg])
        else:
            pc_factor_rows.append([pc, None, 0.0, "n/a", 0.0])

    pc_factor_df = pd.DataFrame(pc_factor_rows, columns=['PC', 'Best factor', 'R^2 (PC vs factor)', 'Regression equation', 'R^2 (reg vs PC)'])
    print(pc_factor_df.to_string(index=False))
else:
    print("Not enough data to compute global PC->factor mapping table.")

# ---------------------------
# Final summary print (compact)
# ---------------------------
print("\n==== Final Key Metrics ====")
print(f"Initial capital: ${CAPITAL:,.2f}")
print(f"Final portfolio value: ${portfolio_values.iloc[-1]:,.2f}")
print(f"Total profit: ${total_return:,.2f} ({total_return_pct:.2%})")
print(f"Long PnL total: ${total_long_pnl:,.2f}")
print(f"Short PnL total: ${total_short_pnl:,.2f}")
print(f"Average hold (days): {avg_hold_days:.1f}")
print(f"Rebalance frequency: weekly ({REBALANCE_FREQ})")
print("==========================\n")

# Optional: save diagnostics to CSV for later analysis
try:
    weights_df.to_csv("weights_history.csv")
    pc_factor_df.to_csv("pc_factor_mapping.csv")
    pd.Series(total_pnl_history, index=dates_history).to_csv("weekly_total_pnl.csv")
    trade_df.to_csv("profitable_trades.csv")
    print("Saved weights_history.csv, pc_factor_mapping.csv, weekly_total_pnl.csv, profitable_trades.csv")
except Exception:
    pass