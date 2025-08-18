# full_manifold_network_pca_strategy_updated.py
# Full pipeline implementing manifold + network-weighted PCA strategy with weekly rebalancing.
# Requirements: numpy, pandas, yfinance, scipy, statsmodels, matplotlib, scikit-learn
# Run in a Python 3.8+ environment.

import warnings
warnings.filterwarnings("ignore")  # turn off warnings as requested

import math
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta

# ---------------------------
# Configuration / Parameters
# ---------------------------
stocks = [
    'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
    'COP', 'EOG', 'DVN', 'APA',
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

# Market proxy for beta2
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
    if X.empty:
        return 0.0, pd.Series(0.0, index=factors_list), 0.0
    y = PC_series.reindex(X.index).dropna()
    X = X.loc[y.index]
    if len(y) < 10:
        return 0.0, pd.Series(0.0, index=factors_list), 0.0
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    alpha = float(model.params['const'])
    betas = model.params.drop('const')
    return alpha, betas.reindex(factors_list).fillna(0.0), float(model.rsquared)

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

# If a ticker missing in either, drop it from our universe (with a warning)
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

out_of_sample_returns = []  # realized portfolio returns (fractional of CAPITAL)
weights_history = []
dates_history = []

# diagnostics histories (only append when an out-of-sample week is recorded)
avg_long_expected = []
avg_short_expected = []
avg_long_weight = []
avg_short_weight = []
centrality_history = []
long_pnl_history = []
short_pnl_history = []
total_pnl_history = []
trade_records = []  # per-trade record list of dicts
pc0_pred_history = []  # predicted PC0 each week (for overlay)
pc0_actual_history = []  # actual PC0 each week (aligned)
pc0_r2_history = []  # r2 between predicted series up to that point and actual (cumulative)

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

    # 9) Optimization -> dollar weights (sum of v_plus + v_minus = CAPITAL)
    w = optimize_portfolio(expected_adj_r, betas, capital=CAPITAL)

    # Diagnostics before OOS compute
    longs = w[w > 0]
    shorts = -w[w < 0]
    avg_long_expected.append(expected_adj_r.loc[longs.index].mean() if len(longs) else 0.0)
    avg_short_expected.append(expected_adj_r.loc[shorts.index].mean() if len(shorts) else 0.0)
    avg_long_weight.append(longs.sum() if len(longs) else 0.0)
    avg_short_weight.append(shorts.sum() if len(shorts) else 0.0)

    # 10) Compute next-week realized return (dollar PnL)
    next_idx = t_idx + 1
    if next_idx >= len(weekly_index):
        # No next-week price -> skip appending diagnostics for this iteration to avoid length mismatches
        continue

    # find next rebalancing date (the next Friday)
    next_week_end = weekly_index[next_idx]
    if next_week_end not in stock_data.index:
        # pad backward to previous available trading day
        try:
            next_week_end = stock_data.index[stock_data.index.get_indexer([next_week_end], method='pad')[0]]
        except Exception:
            # if can't find, skip
            continue

    # compute realized returns for common_universe between window_end and next_week_end
    try:
        price_t = stock_data.loc[window_end, common_universe]
        price_t1 = stock_data.loc[next_week_end, common_universe]
    except KeyError:
        # if some tickers missing at either date, align carefully
        price_t = stock_data.reindex(index=[window_end]).loc[window_end, common_universe]
        price_t1 = stock_data.reindex(index=[next_week_end]).loc[next_week_end, common_universe]

    realized = (price_t1 / price_t - 1).fillna(0.0)

    # w currently indexed by expected_adj_r.index which is common_universe; ensure alignment
    w = w.reindex(common_universe).fillna(0.0)

    # realized pnl in dollars: sum(w * realized)
    realized_pnl = float(np.dot(w.values, realized.values))
    # longs and shorts breakdown
    realized_long_pnl = float((w[w > 0].values * realized[w > 0].values).sum()) if (w > 0).any() else 0.0
    realized_short_pnl_signed = float((w[w < 0].values * realized[w < 0].values).sum()) if (w < 0).any() else 0.0
    realized_short_profit = -realized_short_pnl_signed  # positive if short profited

    # store normalized (fractional) return relative to CAPITAL
    out_of_sample_returns.append(realized_pnl / CAPITAL)
    long_pnl_history.append(realized_long_pnl)
    short_pnl_history.append(realized_short_profit)
    total_pnl_history.append(realized_pnl)
    weights_history.append(w)
    dates_history.append(next_week_end)

    # append centrality history only when we appended a date (keeps lengths aligned)
    centrality_history.append(centrality.reindex(common_universe).reindex(index=common_universe).fillna(0.0))

    # Save per-trade records for this weekly batch (each stock is a trade for one week)
    for stk in common_universe:
        alloc = abs(w.get(stk, 0.0))
        if alloc == 0:
            continue
        side = 'long' if w.get(stk, 0.0) > 0 else 'short'
        entry_price = price_t.get(stk, np.nan)
        exit_price = price_t1.get(stk, np.nan)
        rtn = (exit_price / entry_price - 1) if (side == 'long' and not (math.isnan(entry_price) or math.isnan(exit_price))) else (entry_price / exit_price - 1)
        pnl = alloc * rtn if not (math.isnan(rtn) or math.isnan(alloc)) else 0.0
        trade_records.append({
            'start': window_end,
            'end': next_week_end,
            'stock': stk,
            'side': side,
            'alloc': alloc,
            'entry': entry_price,
            'exit': exit_price,
            'return': rtn,
            'pnl': pnl
        })

    # PC0 weekly prediction & actual capture (for plotting and R^2 over time)
    # compute predicted PC0 value for the NEXT week as scalar pc_pred_moves['PC1'] etc.
    # We align PC_ts last observed (PC_ts is in-sample daily; we use its last value as "actual" for this week_end)
    # For weekly aggregation: take PC_ts resampled by week (last)
    try:
        pc1_weekly_actual = PC_ts['PC1'].resample(REBALANCE_FREQ).last().loc[:week_end].iloc[-1]
    except Exception:
        pc1_weekly_actual = PC_ts['PC1'].iloc[-1] if 'PC1' in PC_ts.columns else 0.0
    # Our predicted percent-move corresponds to pc_pred_moves['PC1'] (percent movement scaled)
    # We'll treat predicted PC value as last_actual + predicted_percent* (historic std)
    last_pc1_value = PC_ts['PC1'].iloc[-1] if 'PC1' in PC_ts.columns else 0.0
    pred_pc1_value = last_pc1_value + pc_pred_moves.get('PC1', 0.0)
    pc0_pred_history.append(pred_pc1_value)
    pc0_actual_history.append(pc1_weekly_actual)

    # compute rolling R2 between predicted and actual series so far
    if len(pc0_pred_history) >= 3:
        arr_pred = np.array(pc0_pred_history)
        arr_actual = np.array(pc0_actual_history)
        try:
            r2v = r2_score(arr_actual, arr_pred)
        except Exception:
            r2v = np.nan
    else:
        r2v = np.nan
    pc0_r2_history.append(r2v)

# ---------------------------
# Results & Diagnostics
# ---------------------------
if len(out_of_sample_returns) == 0:
    raise RuntimeError("No out-of-sample weeks were produced; check data and lookback settings.")

# Build series
port_rets = pd.Series(out_of_sample_returns, index=dates_history)  # normalized to CAPITAL (fraction)
cum = (1 + port_rets).cumprod()
portfolio_values = CAPITAL * cum  # dollar portfolio values

annualized_return = (1 + port_rets.mean()) ** 52 - 1
annualized_vol = port_rets.std() * np.sqrt(52)
sharpe = annualized_return / (annualized_vol if annualized_vol > 0 else np.nan)
total_profit = portfolio_values.iloc[-1] - CAPITAL
total_return_pct = total_profit / CAPITAL
rolling_max = portfolio_values.cummax()
drawdown = (portfolio_values - rolling_max) / rolling_max
max_dd = drawdown.min()

# aggregated long/short PnL (dollars)
total_long_pnl = float(np.nansum(long_pnl_history))
total_short_pnl = float(np.nansum(short_pnl_history))
total_pnl = float(np.nansum(total_pnl_history))

# Build weights DataFrame (index=dates_history). Ensure consistent columns
weights_df = pd.DataFrame(weights_history, index=dates_history).fillna(0.0).sort_index()

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

# Trade-level profitability
trades_df = pd.DataFrame(trade_records)
if not trades_df.empty:
    longs_trades = trades_df[trades_df['side'] == 'long']
    shorts_trades = trades_df[trades_df['side'] == 'short']
    long_profitable = longs_trades[longs_trades['pnl'] > 0]
    short_profitable = shorts_trades[shorts_trades['pnl'] > 0]
    long_perf_pct = len(long_profitable) / len(longs_trades) if len(longs_trades) else np.nan
    short_perf_pct = len(short_profitable) / len(shorts_trades) if len(shorts_trades) else np.nan
    avg_long_trade_pnl = longs_trades['pnl'].mean() if len(longs_trades) else 0.0
    avg_short_trade_pnl = shorts_trades['pnl'].mean() if len(shorts_trades) else 0.0
    total_money_traded = trades_df['alloc'].sum()  # note: sum of per-week allocations (not unique capital)
else:
    long_perf_pct = short_perf_pct = np.nan
    avg_long_trade_pnl = avg_short_trade_pnl = 0.0
    total_money_traded = 0.0

# Print summary
print("==== Backtest summary ====")
print("Weeks (OOS):", len(port_rets))
print(f"Money put in (initial capital): ${CAPITAL:,.2f}")
print(f"Final portfolio value: ${portfolio_values.iloc[-1]:,.2f}")
print(f"Total profit: ${total_profit:,.2f} ({total_return_pct:.2%})")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized vol: {annualized_vol:.2%}")
print(f"Sharpe (rfr=0): {sharpe:.2f}")
print(f"Max drawdown: {max_dd:.2%}")
print(f"Total long PnL (dollars): ${total_long_pnl:,.2f}")
print(f"Total short PnL (dollars, positive means profitable): ${total_short_pnl:,.2f}")
print(f"Average hold period: {avg_hold_days:.1f} days (approx)")
print(f"Rebalancing frequency: weekly ({REBALANCE_FREQ})")
print("==========================")

# show last weights
print("\nSample weights (last rebalance) - top exposures:")
print(weights_df.iloc[-1].sort_values(ascending=False).head(20))

# ---------------------------
# Extra analytics & plots
# ---------------------------

# 1) cumulative returns with drawdown (dollars)
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values.index, portfolio_values.values, label='Portfolio value ($)')
plt.fill_between(portfolio_values.index, portfolio_values.values, rolling_max.values,
                 where=portfolio_values.values < rolling_max.values, color='red', alpha=0.15, label='Drawdown')
plt.title('Portfolio Value and Drawdown')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# 2) cumulative returns (normalized)
plt.figure(figsize=(10, 5))
plt.plot(cum.index, cum.values, label='Cumulative return (index)')
plt.title('Strategy cumulative returns (weekly rebalanced)')
plt.xlabel('Date')
plt.ylabel('Cumulative return (1=initial)')
plt.grid(True)
plt.legend()
plt.show()

# 3) PCA on full sample (global) for plotting PCs over time
returns_for_global = stock_returns_all.dropna(how='any')
PC_ts_full = None
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
    print("Not enough data for global PCA plotting.")

# 4) PC0 regression overlay: weekly-prediction line and actual PC0
if PC_ts_full is not None:
    # Build weekly PC1 actual series
    pc1_weekly_actual = PC_ts_full['PC1'].resample(REBALANCE_FREQ).last()
    # Align weekly predicted (we built pc0_pred_history aligned to dates_history)
    pred_series = pd.Series(pc0_pred_history, index=dates_history) if len(pc0_pred_history) == len(dates_history) else None
    actual_series = pd.Series(pc0_actual_history, index=dates_history) if len(pc0_actual_history) == len(dates_history) else None

    # Plot weekly predicted vs actual PC1
    if pred_series is not None and actual_series is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(actual_series.index, actual_series.values, label='Actual PC1 (weekly last in-sample)')
        plt.plot(pred_series.index, pred_series.values, label='Predicted PC1 (our model)', linestyle='--')
        plt.title('PC1 Weekly: Predicted vs Actual')
        plt.xlabel('Date')
        plt.ylabel('PC1 value')
        plt.legend()
        plt.grid(True)
        plt.show()

        # R^2 overall
        mask = ~np.isnan(pred_series.values) & ~np.isnan(actual_series.values)
        if mask.sum() >= 3:
            r2_overall = r2_score(actual_series.values[mask], pred_series.values[mask])
        else:
            r2_overall = np.nan
        print(f"PC1 predicted vs actual overall R^2: {r2_overall:.4f}")

        # R^2 over time (rolling window)
        window = min(26, len(actual_series))  # half-year rolling
        rolling_r2 = []
        for i in range(len(actual_series)):
            if i + 1 < 5:
                rolling_r2.append(np.nan)
                continue
            start_i = max(0, i + 1 - window)
            a = actual_series.values[start_i:i+1]
            p = pred_series.values[start_i:i+1]
            try:
                rolling_r2.append(r2_score(a, p))
            except Exception:
                rolling_r2.append(np.nan)
        plt.figure(figsize=(12, 4))
        plt.plot(actual_series.index, rolling_r2, label='Rolling R^2 (PC1 predicted vs actual)')
        plt.title('Rolling R^2 for PC1 prediction (window ~{} weeks)'.format(window))
        plt.ylabel('R^2')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("PC1 weekly predicted/actual not available for plotting.")
else:
    print("Not enough global PCA data to plot PC1 regression overlay.")

# 5) centrality over time: show top N stocks by average centrality
if centrality_history and dates_history:
    centrality_df = pd.DataFrame([s.reindex(sorted(set().union(*[list(s.index) for s in centrality_history]))).fillna(0.0) for s in centrality_history], index=dates_history)
    centrality_df = centrality_df.sort_index().fillna(0.0)
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
else:
    print("No centrality history to plot.")

# 6) weights heatmap for top stocks by avg abs exposure
if not weights_df.empty:
    avg_abs_weight = weights_df.abs().mean().sort_values(ascending=False)
    top_m = min(20, len(avg_abs_weight))
    top_stocks_by_weight = list(avg_abs_weight.index[:top_m])
    weights_sub = weights_df[top_stocks_by_weight].sort_index()
    plt.figure(figsize=(14, max(4, top_m * 0.25)))
    im = plt.imshow(weights_sub.T, aspect='auto', cmap='seismic', interpolation='nearest', vmin=-np.max(np.abs(weights_sub.values)), vmax=np.max(np.abs(weights_sub.values)))
    plt.colorbar(im, label='Dollar weight')
    plt.yticks(range(len(top_stocks_by_weight)), top_stocks_by_weight)
    plt.xticks(range(len(weights_sub.index)), [d.strftime('%Y-%m-%d') for d in weights_sub.index], rotation=90)
    plt.title('Weights heatmap (rows=stocks, cols=weeks)')
    plt.tight_layout()
    plt.show()

# 7) Long vs Short weekly PnL overlayed with total PnL line & buys/shorts counts
plt.figure(figsize=(12, 6))
plt.plot(dates_history, np.cumsum(long_pnl_history), label='Cumulative Long PnL ($)')
plt.plot(dates_history, np.cumsum(short_pnl_history), label='Cumulative Short PnL ($)')
plt.plot(dates_history, np.cumsum(total_pnl_history), label='Cumulative Total PnL ($)', linewidth=2, color='black')
plt.title('Cumulative Long/Short/Total PnL ($)')
plt.legend()
plt.grid(True)
plt.show()

# New long / new short counts per week overlaid on cumulative profit
if not weights_df.empty:
    shifted = weights_df.shift(1).fillna(0.0)
    new_longs = ((weights_df > 0) & (shifted <= 0)).sum(axis=1)
    new_shorts = ((weights_df < 0) & (shifted >= 0)).sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(portfolio_values.index, portfolio_values.values, color='black', label='Portfolio ($)')
    ax1.set_ylabel('Portfolio $', color='black')
    ax2 = ax1.twinx()
    ax2.plot(new_longs.index, new_longs.values, label='New longs (count)', color='green', alpha=0.7)
    ax2.plot(new_shorts.index, new_shorts.values, label='New shorts (count)', color='red', alpha=0.7)
    ax2.set_ylabel('New signals (count)')
    ax1.set_title('Portfolio value with new long/short counts per rebalance')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

# 8) Weekly total PnL bar
plt.figure(figsize=(12, 5))
plt.bar(dates_history, total_pnl_history, color='tab:blue', alpha=0.7)
plt.title('Weekly Total PnL ($)')
plt.grid(True)
plt.show()

# 9) Portfolio composition over time (stacked area) - longs (top 65%), shorts (bottom 35%)
if not weights_df.empty:
    # separate into longs and shorts
    longs_df = weights_df.clip(lower=0)
    shorts_df = -weights_df.clip(upper=0)  # positive numbers for shorts amounts
    # Normalize vertical allocation: longs occupy 65% of vertical room, shorts 35%
    # We'll scale them to a common vertical axis where max(long sum) maps to 0.65 and max(short sum) maps to 0.35
    long_totals = longs_df.sum(axis=1)
    short_totals = shorts_df.sum(axis=1)
    max_long = long_totals.max() if long_totals.max() > 0 else 1.0
    max_short = short_totals.max() if short_totals.max() > 0 else 1.0

    # scale factors
    long_scale = 0.65 / max_long
    short_scale = 0.35 / max_short

    scaled_longs = longs_df.multiply(long_scale)
    scaled_shorts = shorts_df.multiply(short_scale)

    # stacked area: plot shorts as positive areas below the middle line, longs above
    midline = 0.5  # center of plot in relative units; map longs above 0.5 and shorts below 0.5
    dates = weights_df.index

    plt.figure(figsize=(14, 6))
    # Plot shorts stacked downward from midline
    bottom = np.full(len(dates), midline)
    for col in scaled_shorts.columns:
        vals = scaled_shorts[col].values
        plt.fill_between(dates, bottom, bottom - vals, label=col, alpha=0.6)
        bottom = bottom - vals
    # Plot longs stacked upward from midline
    top = np.full(len(dates), midline)
    for col in scaled_longs.columns:
        vals = scaled_longs[col].values
        plt.fill_between(dates, top, top + vals, label=col, alpha=0.6)
        top = top + vals

    plt.plot(portfolio_values.index, (portfolio_values - portfolio_values.min()) / (portfolio_values.max() - portfolio_values.min()) * 0.9 + 0.05,
             color='black', linewidth=2, label='Normalized portfolio $ (for reference)')
    plt.title('Portfolio Composition Over Time (stacked stocks). Midline separates longs (above) and shorts (below)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylim(0, 1)
    plt.show()
else:
    print("No weights to show composition plot.")

# 10) PC -> Factor mapping table (global)
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
            r2_base = np.corrcoef(a, b)[0, 1] ** 2
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
print(f"Total profit: ${total_profit:,.2f} ({total_return_pct:.2%})")
print(f"Long PnL total: ${total_long_pnl:,.2f}")
print(f"Short PnL total: ${total_short_pnl:,.2f}")
print(f"Average hold (days): {avg_hold_days:.1f}")
print(f"Rebalance frequency: weekly ({REBALANCE_FREQ})")
print(f"Trades executed (weekly single-week trades): {len(trades_df) if not trades_df.empty else 0}")
print(f"Long profitable trades: {len(long_profitable) if not trades_df.empty else 0} / {len(longs_trades) if not trades_df.empty else 0} ({long_perf_pct:.2%} if available)")
print(f"Short profitable trades: {len(short_profitable) if not trades_df.empty else 0} / {len(shorts_trades) if not trades_df.empty else 0} ({short_perf_pct:.2%} if available)")
print(f"Avg long trade PnL: ${avg_long_trade_pnl:,.2f}")
print(f"Avg short trade PnL: ${avg_short_trade_pnl:,.2f}")
print("==========================\n")

# Optional: save diagnostics to CSV for later analysis
try:
    weights_df.to_csv("weights_history.csv")
    if PC_ts_full is not None:
        pc_factor_df.to_csv("pc_factor_mapping.csv")
    pd.Series(total_pnl_history, index=dates_history).to_csv("weekly_total_pnl.csv")
    if not trades_df.empty:
        trades_df.to_csv("trade_records.csv", index=False)
    print("Saved weights_history.csv, pc_factor_mapping.csv (if available), weekly_total_pnl.csv, trade_records.csv")
except Exception:
    pass

# End of script
