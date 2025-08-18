import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as pulp_value
import matplotlib.pyplot as plt
import warnings
import pulp

warnings.filterwarnings('ignore')


def fetch_data(stocks, factors_tickers, start_date, end_date, max_retries=3):
    """Fetch price data for stocks and factors with retry logic"""
    prices = pd.DataFrame()
    failed_tickers = []

    for ticker in stocks + factors_tickers:
        success = False
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if 'Close' in df.columns and not df['Close'].empty and len(df) > 50:
                    prices[ticker] = df['Close']
                    success = True
                    break
            except:
                pass

        if not success:
            failed_tickers.append(ticker)

    return prices


def fetch_external_factors(start_date, end_date, max_retries=3):
    """Fetch external factors with robust retry and proper alignment"""
    external_data = pd.DataFrame()

    with open("data_fetch_diagnostics.txt", 'w') as f:
        f.write(f"=== Data Fetch for {start_date} to {end_date} ===\n")

        for ticker, name in [('^VIX', 'VIX'), ('^TNX', 'TNX'), ('DX-Y.NYB', 'DXY'), ('GC=F', 'GOLD')]:
            success = False
            for attempt in range(max_retries):
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False, repair=True,
                                     auto_adjust=True)
                    if not df.empty and 'Close' in df.columns and len(df) > 50:
                        # Key fix: Don't reindex to daily, keep original trading day index
                        # Only forward fill missing values within the existing index
                        factor_series = df['Close'].ffill().dropna()

                        if len(factor_series) > 50:  # Ensure we have enough data
                            external_data[name] = factor_series
                            f.write(
                                f"{name}: {len(factor_series)} data points, from {factor_series.index[0]} to {factor_series.index[-1]}\n")
                            success = True
                            break
                    else:
                        f.write(f"{name}: No data retrieved on attempt {attempt + 1}\n")
                except Exception as e:
                    f.write(f"{name}: Error on attempt {attempt + 1} - {e}\n")

            if not success:
                f.write(f"{name}: Failed after {max_retries} attempts, creating empty series\n")
                # Create empty series instead of zero-filled - let alignment happen later
                external_data[name] = pd.Series(dtype=float, name=name)

    return external_data


def prepare_returns_and_factors(prices, stocks, external_factors,
                                lookback_period=252, min_observations=100):
    """Prepare returns and align with external factors - IMPROVED VERSION"""
    valid_stocks = [stock for stock in stocks if
                    stock in prices.columns and len(prices[stock].dropna()) >= min_observations]

    if len(valid_stocks) < 10:
        raise ValueError(f"Insufficient valid stocks: {len(valid_stocks)}")

    stock_prices = prices[valid_stocks].dropna()
    returns = stock_prices.pct_change().dropna().tail(lookback_period)

    # Prepare external factor changes with proper alignment
    factor_data = pd.DataFrame(index=returns.index)

    for factor in external_factors.columns:
        if factor in external_factors.columns and len(external_factors[factor].dropna()) >= min_observations:
            # Align the external factor with returns index first
            aligned_factor = external_factors[factor].reindex(returns.index, method='ffill')

            # Calculate changes based on factor type
            if factor == 'VIX':
                # VIX: use absolute changes (more meaningful for volatility)
                factor_changes = aligned_factor.diff()
            elif factor in ['TNX']:
                # Interest rates: use absolute changes (basis points)
                factor_changes = aligned_factor.diff()
            else:
                # Currency and commodities: use percentage changes
                factor_changes = aligned_factor.pct_change()

            # Only include if we have sufficient non-null data
            if factor_changes.dropna().shape[0] >= min_observations // 2:
                factor_data[f'{factor}_change'] = factor_changes

    # Clean up missing data more carefully
    # First, ensure we have at least some factor data
    if factor_data.empty:
        raise ValueError("No valid external factors after alignment")

    # Remove rows where ALL factor changes are NaN
    factor_data = factor_data.dropna(how='all')

    # Align returns with the cleaned factor data
    returns = returns.reindex(factor_data.index).dropna()

    # Final alignment - keep only common index
    common_index = returns.index.intersection(factor_data.index)
    returns = returns.loc[common_index]
    factor_data = factor_data.loc[common_index]

    # Remove any remaining NaN values
    factor_data = factor_data.fillna(0)  # Fill remaining NaNs with 0 for changes

    if len(returns) < 100:
        raise ValueError(f"Insufficient aligned observations: {len(returns)}")

    print(f"Aligned data: {len(returns)} observations, {len(factor_data.columns)} factors")
    print(f"Factor data columns: {list(factor_data.columns)}")
    print(f"Factor data sample:\n{factor_data.head()}")

    return returns, factor_data, valid_stocks

def compute_betas(returns, market_returns, stocks, min_r_squared=0.05):
    """Calculate beta for each stock relative to market"""
    betas = []
    for stock in stocks:
        try:
            y = returns[stock].dropna()
            X = market_returns.reindex(y.index).dropna()
            common_idx = y.index.intersection(X.index)
            y_aligned = y.loc[common_idx]
            X_aligned = X.loc[common_idx]

            if len(y_aligned) > 30 and np.std(X_aligned) > 1e-6:
                X_with_const = sm.add_constant(X_aligned)
                model = sm.OLS(y_aligned, X_with_const, missing='drop').fit()
                betas.append(float(model.params.iloc[1]) if model.rsquared >= min_r_squared else 1.0)
            else:
                betas.append(1.0)
        except:
            betas.append(1.0)
    return np.array(betas)


def compute_network_centrality(returns, stocks, corr_threshold=0.6):
    """Calculate eigenvector centrality based on correlation network"""
    corr_matrix = returns.corr().fillna(0).values
    adj_matrix = np.abs(corr_matrix) >= corr_threshold
    adj_matrix = adj_matrix.astype(float)
    np.fill_diagonal(adj_matrix, 1.0)

    try:
        eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
        max_eig_idx = np.argmax(np.real(eigenvalues))
        centrality_vector = np.abs(np.real(eigenvectors[:, max_eig_idx]))
    except:
        centrality_vector = np.sum(adj_matrix, axis=1)

    if np.sum(centrality_vector) > 0:
        centrality_vector /= np.sum(centrality_vector)
    else:
        centrality_vector = np.ones(len(stocks)) / len(stocks)

    centrality = dict(zip(stocks, centrality_vector))
    return centrality, centrality_vector


def perform_pca(returns, k=8):
    """Perform PCA with standardized returns"""
    returns_standardized = (returns - returns.mean()) / returns.std()
    corr_matrix = returns_standardized.corr().fillna(0).values
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    sorted_idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[sorted_idx])
    eigenvectors = np.real(eigenvectors[:, sorted_idx])

    k_final = min(k, len(eigenvalues))
    Q = eigenvectors[:, :k_final]
    pc_time_series = returns_standardized.fillna(0).values @ Q
    pc_std = np.std(pc_time_series, axis=0, ddof=1)
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)

    return Q, pc_time_series, pc_std, explained_var_ratio[:k_final]


def debug_factor_alignment(returns, factor_data, pc_time_series, k=8):
    """Debug function to check factor data quality and alignment"""
    print("=== FACTOR ALIGNMENT DEBUG ===")
    print(f"Returns shape: {returns.shape}")
    print(f"Factor data shape: {factor_data.shape}")
    print(f"PC time series shape: {pc_time_series.shape}")
    print(f"Returns index range: {returns.index[0]} to {returns.index[-1]}")
    print(f"Factor data index range: {factor_data.index[0]} to {factor_data.index[-1]}")

    print("\nFactor data statistics:")
    print(factor_data.describe())

    print("\nFactor data null counts:")
    print(factor_data.isnull().sum())

    print("\nFactor data standard deviations:")
    print(factor_data.std())

    # Check if any factors have zero variance
    zero_var_factors = factor_data.columns[factor_data.std() < 1e-8]
    if len(zero_var_factors) > 0:
        print(f"\nWARNING: Zero variance factors: {list(zero_var_factors)}")

    # Check alignment between PC time series and factor data
    print(f"\nPC time series vs factor data length match: {len(pc_time_series) == len(factor_data)}")

    return True


# Add this to your identify_economic_factors function (replace the existing one)
def identify_economic_factors(pc_time_series, factor_data, k, min_r2_threshold=0.05):
    """Map PCs to economic factors with enhanced debugging and alignment checks"""
    factor_names = factor_data.columns.tolist()
    assignments = {}
    betas = {}
    alphas = {}
    r2s = {}
    used_factors = set()

    print(f"\n=== IDENTIFYING FACTORS FOR {k} PCs ===")
    print(f"Available factors: {factor_names}")
    print(f"PC time series shape: {pc_time_series.shape}")
    print(f"Factor data shape: {factor_data.shape}")

    for j in range(k):
        y = pc_time_series[:, j]
        best_factor = 'Residual'
        best_r2 = 0
        best_beta = 0
        best_alpha = 0

        print(f"\nAnalyzing PC{j}:")
        print(f"  PC{j} stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}")

        for factor_name in factor_names:
            if factor_name in used_factors:
                continue
            try:
                factor_series = factor_data[factor_name].values

                # Ensure lengths match
                if len(factor_series) != len(y):
                    print(f"  {factor_name}: Length mismatch {len(factor_series)} vs {len(y)}")
                    continue

                # Check for sufficient variation
                if np.std(factor_series) < 1e-8 or np.std(y) < 1e-8:
                    print(
                        f"  {factor_name}: Insufficient variation (std_factor={np.std(factor_series):.6f}, std_pc={np.std(y):.6f})")
                    continue

                # Remove NaN values pairwise
                mask = ~(np.isnan(factor_series) | np.isnan(y))
                if np.sum(mask) < 50:
                    print(f"  {factor_name}: Insufficient valid pairs after NaN removal ({np.sum(mask)})")
                    continue

                y_clean = y[mask]
                x_clean = factor_series[mask]

                X_with_const = sm.add_constant(x_clean)
                model = sm.OLS(y_clean, X_with_const, missing='drop').fit()
                r2 = model.rsquared

                print(f"  {factor_name}: R²={r2:.4f}, beta={model.params[1]:.4f}, n_obs={len(y_clean)}")

                if r2 > best_r2 and r2 >= min_r2_threshold:
                    best_r2 = r2
                    best_factor = factor_name
                    best_beta = float(model.params[1])
                    best_alpha = float(model.params[0])

            except Exception as e:
                print(f"  {factor_name}: Error - {e}")
                continue

        assignments[j] = best_factor
        betas[j] = best_beta
        alphas[j] = best_alpha
        r2s[j] = best_r2

        if best_factor != 'Residual':
            used_factors.add(best_factor)
            print(f"  → ASSIGNED: {best_factor} (R²={best_r2:.4f})")
        else:
            print(f"  → RESIDUAL (best R²={best_r2:.4f})")

    return assignments, betas, alphas, r2s


def detect_regime_and_predict_factors(factor_data, lookback_periods=[20, 60, 120],
                                      log_file="factor_predictions_diagnostics.txt", current_date=None,
                                      log_last_n_periods=3, period_counter=None, total_periods=None):
    """Predict factor changes with regime detection and streamlined diagnostics"""
    predictions = {}

    # Only log diagnostics for the last n periods
    should_log = period_counter is not None and total_periods is not None and period_counter >= total_periods - log_last_n_periods

    if should_log:
        # Append diagnostics with rebalance date
        with open(log_file, 'a') as f:
            f.write(
                f"\n=== Diagnostics for factor predictions at rebalance date {current_date.strftime('%Y-%m-%d')} ===\n")

            for factor in factor_data.columns:
                factor_series = factor_data[factor].dropna()
                f.write(f"\nFactor: {factor}\n")
                f.write(f"Available data points: {len(factor_series)}\n")

                if len(factor_series) < max(lookback_periods):
                    predictions[factor] = 0.0
                    f.write(f"Insufficient data (< {max(lookback_periods)} periods), setting prediction to 0.0\n")
                    continue

                # Short-term momentum
                short_momentum = factor_series.tail(lookback_periods[0]).mean()
                f.write(f"Short-term momentum ({lookback_periods[0]} days): {short_momentum:.6f}\n")

                # Medium-term momentum
                medium_momentum = factor_series.tail(lookback_periods[1]).mean()
                f.write(f"Medium-term momentum ({lookback_periods[1]} days): {medium_momentum:.6f}\n")

                # Long-term mean reversion
                long_term_mean = factor_series.tail(lookback_periods[2]).mean()
                current_level = factor_series.iloc[-1]
                mean_reversion = (long_term_mean - current_level) * 0.1
                f.write(f"Long-term mean ({lookback_periods[2]} days): {long_term_mean:.6f}\n")
                f.write(f"Current level: {current_level:.6f}\n")
                f.write(f"Mean reversion signal: {mean_reversion:.6f}\n")

                # Volatility regime detection
                recent_vol = factor_series.tail(lookback_periods[0]).std()
                long_vol = factor_series.tail(lookback_periods[2]).std()
                vol_regime_multiplier = 1.0
                f.write(f"Recent volatility ({lookback_periods[0]} days): {recent_vol:.6f}\n")
                f.write(f"Long-term volatility ({lookback_periods[2]} days): {long_vol:.6f}\n")

                if recent_vol > 1.5 * long_vol:
                    vol_regime_multiplier = 0.5
                    f.write("High volatility regime detected, multiplier: 0.5\n")
                elif recent_vol < 0.7 * long_vol:
                    vol_regime_multiplier = 1.2
                    f.write("Low volatility regime detected, multiplier: 1.2\n")
                else:
                    f.write("Neutral volatility regime, multiplier: 1.0\n")

                # Combine signals with regime adjustment
                base_prediction = (short_momentum * 0.4 + medium_momentum * 0.3 + mean_reversion * 0.3)
                f.write(f"Base prediction (before clipping): {base_prediction:.6f}\n")
                prediction = base_prediction * vol_regime_multiplier
                f.write(f"Prediction after volatility adjustment: {prediction:.6f}\n")

                # Relaxed clipping to reduce boundary effects
                predictions[factor] = np.clip(prediction, -0.2, 0.2)
                f.write(f"Final clipped prediction: {predictions[factor]:.6f}\n")

    else:
        # Compute predictions without logging
        for factor in factor_data.columns:
            factor_series = factor_data[factor].dropna()
            if len(factor_series) < max(lookback_periods):
                predictions[factor] = 0.0
                continue

            short_momentum = factor_series.tail(lookback_periods[0]).mean()
            medium_momentum = factor_series.tail(lookback_periods[1]).mean()
            long_term_mean = factor_series.tail(lookback_periods[2]).mean()
            current_level = factor_series.iloc[-1]
            mean_reversion = (long_term_mean - current_level) * 0.1
            recent_vol = factor_series.tail(lookback_periods[0]).std()
            long_vol = factor_series.tail(lookback_periods[2]).std()
            vol_regime_multiplier = 1.0

            if recent_vol > 1.5 * long_vol:
                vol_regime_multiplier = 0.5
            elif recent_vol < 0.7 * long_vol:
                vol_regime_multiplier = 1.2

            base_prediction = (short_momentum * 0.4 + medium_momentum * 0.3 + mean_reversion * 0.3)
            prediction = base_prediction * vol_regime_multiplier
            predictions[factor] = np.clip(prediction, -0.2, 0.2)

    return predictions

def predict_pc_movements(assignments, betas, alphas, predicted_factor_changes, pc_std, k):
    """Predict PC movements based on factor predictions"""
    pc_movements = np.zeros(k)
    for j in range(k):
        factor = assignments[j]
        if factor in predicted_factor_changes and factor != 'Residual':
            delta_factor = predicted_factor_changes[factor]
            raw_pc_change = alphas[j] + betas[j] * delta_factor
            pc_movements[j] = raw_pc_change * pc_std[j] if pc_std[j] > 1e-8 else raw_pc_change
    return pc_movements


def compute_expected_returns(Q, pc_movements_percent, centrality, stocks):
    """Compute expected returns with network centrality weighting"""
    raw_returns = Q @ pc_movements_percent
    c_vector = np.array([centrality[stock] for stock in stocks])
    adjusted_returns = raw_returns * c_vector
    return adjusted_returns

def optimize_portfolio_market_neutral(adjusted_r, betas, stocks, capital=10000.0,
                                     max_position_pct=0.15, long_short_ratio=0.65,
                                     beta_tolerance=0.1):
    """Market neutral portfolio optimization with stricter position limits"""
    prob = LpProblem("Market_Neutral_Portfolio", LpMaximize)

    v_long = {stock: LpVariable(f"v_long_{stock}", lowBound=0, upBound=capital * max_position_pct)
              for stock in stocks}
    v_short = {stock: LpVariable(f"v_short_{stock}", lowBound=0, upBound=capital * max_position_pct * 0.8)
               for stock in stocks}

    prob += lpSum([adjusted_r[i] * (v_long[stocks[i]] - v_short[stocks[i]]) for i in range(len(stocks))])

    prob += lpSum([v_long[stock] for stock in stocks]) == capital * long_short_ratio
    prob += lpSum([v_short[stock] for stock in stocks]) == capital * (1 - long_short_ratio)

    beta_exposure = lpSum([betas[i] * (v_long[stocks[i]] - v_short[stocks[i]]) for i in range(len(stocks))])
    prob += beta_exposure <= capital * beta_tolerance
    prob += beta_exposure >= -capital * beta_tolerance

    # Stricter concentration limit
    for stock in stocks:
        prob += v_long[stock] <= capital * 0.1  # Max 10% per long position
        prob += v_short[stock] <= capital * 0.08  # Max 8% per short position

    for i, stock in enumerate(stocks):
        if adjusted_r[i] >= -0.001:
            prob += v_short[stock] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    positions = {}
    for stock in stocks:
        long_val = pulp_value(v_long[stock]) if v_long[stock].value() else 0.0
        short_val = pulp_value(v_short[stock]) if v_short[stock].value() else 0.0
        net_position = long_val - short_val
        positions[stock] = net_position if abs(net_position) > 0.01 else 0.0

    return positions


def plot_trades_over_time(positions_history, rebalance_dates, portfolio_returns):
    """Plot all buys, sells, and shorts over time with cumulative portfolio returns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot cumulative portfolio returns
    portfolio_cum_returns = np.cumprod(1 + np.array(portfolio_returns)) - 1
    ax1.plot(rebalance_dates[:-1], portfolio_cum_returns, label='Portfolio Cumulative Return', color='blue',
             linewidth=2)
    ax1.set_title('Portfolio Cumulative Returns', fontsize=12)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot trades (long and short positions)
    all_stocks = set()
    for period in positions_history:
        all_stocks.update(period['positions'].keys())
    all_stocks = sorted(list(all_stocks))

    long_positions = []
    short_positions = []
    for period in positions_history:
        positions = period['positions']
        long_vals = [positions.get(stock, 0) if positions.get(stock, 0) > 0 else 0 for stock in all_stocks]
        short_vals = [-positions.get(stock, 0) if positions.get(stock, 0) < 0 else 0 for stock in all_stocks]
        long_positions.append(long_vals)
        short_positions.append(short_vals)

    long_positions = np.array(long_positions)
    short_positions = np.array(short_positions)

    # Stacked bar plot for long and short positions
    colors_long = plt.cm.Greens(np.linspace(0.3, 1, len(all_stocks)))
    colors_short = plt.cm.Reds(np.linspace(0.3, 1, len(all_stocks)))

    for i, stock in enumerate(all_stocks):
        ax2.bar(rebalance_dates[:-1], long_positions[:, i] / 10000.0,
                bottom=np.sum(long_positions[:, :i], axis=1) / 10000.0,
                color=colors_long[i], alpha=0.7, width=7)
        ax2.bar(rebalance_dates[:-1], -short_positions[:, i] / 10000.0,
                bottom=np.sum(-short_positions[:, :i], axis=1) / 10000.0,
                color=colors_short[i], alpha=0.7, width=7)

    ax2.set_title('Long and Short Positions Over Time', fontsize=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Position Size (Capital Fraction)', fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_regime_predictions(factor_data, predicted_changes, rebalance_dates):
    """Plot factor predictions over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    factors_to_plot = list(factor_data.columns)[:4]

    for i, factor in enumerate(factors_to_plot):
        if i < len(axes):
            ax = axes[i]

            # Plot historical factor values
            factor_values = factor_data[factor].values
            ax.plot(range(len(factor_values)), factor_values,
                    label=f'{factor} History', color='blue', alpha=0.7)

            # Plot predictions at rebalancing points
            prediction_values = [predicted_changes.get(factor, 0) for _ in rebalance_dates[:-1]]
            ax2 = ax.twinx()
            ax2.scatter(range(0, len(factor_values), len(factor_values) // len(prediction_values)),
                        prediction_values[:min(len(prediction_values), len(factor_values) // 10 + 1)],
                        color='red', label=f'{factor} Predictions', s=30)

            ax.set_title(f'{factor}: History vs Predictions')
            ax.set_ylabel('Factor Value')
            ax2.set_ylabel('Predicted Change')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def comprehensive_backtest(prices, stocks, external_factors,
                          lookback_period=252, rebalance_freq='W-FRI', k=8,
                          capital=10000.0, total_years=2, min_r2_threshold=0.05):
    """Comprehensive backtest with dynamic factor prediction"""
    # Clear diagnostic file at start
    with open("factor_predictions_diagnostics.txt", 'w') as f:
        f.write("=== Backtest Diagnostics ===\n")

    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=365 * total_years)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    rebalance_dates = rebalance_dates[rebalance_dates.isin(prices.index)]

    if len(rebalance_dates) < 10:
        return None

    portfolio_returns = []
    long_returns = []
    short_returns = []
    portfolio_values = [capital]
    final_assignments = {}
    final_r2s = {}
    final_alphas = {}
    final_betas = {}
    long_positions_count = []
    short_positions_count = []
    avg_long_expected_return = []
    avg_short_expected_return = []
    all_positions_history = []
    all_predictions_history = []
    winning_periods = 0
    losing_periods = 0
    max_drawdown = 0
    peak_value = capital
    final_Q = None
    final_returns = None

    total_periods = len(rebalance_dates) - 1

    for i, current_date in enumerate(rebalance_dates[:-1]):
        try:
            lookback_start = current_date - timedelta(days=int(lookback_period * 1.5))
            hist_prices = prices.loc[lookback_start:current_date]
            hist_external = external_factors.loc[lookback_start:current_date]

            if len(hist_prices) < lookback_period // 2:
                continue

            returns, factor_data, valid_stocks = prepare_returns_and_factors(
                hist_prices, stocks, hist_external, lookback_period
            )
            if len(returns) < 100 or len(valid_stocks) < 10:
                continue

            market_proxy = 'XLE'
            if market_proxy in hist_prices.columns:
                market_returns = hist_prices[market_proxy].pct_change().reindex(returns.index).dropna()
            else:
                market_returns = returns.mean(axis=1)

            betas = compute_betas(returns, market_returns, valid_stocks)
            centrality, centrality_vector = compute_network_centrality(returns, valid_stocks, corr_threshold=0.6)
            Q, pc_time_series, pc_std, _ = perform_pca(returns, k)
            assignments, f_betas, alphas, r2s = identify_economic_factors(
                pc_time_series, factor_data, k, min_r2_threshold
            )
            debug_factor_alignment(returns, factor_data, pc_time_series, k)

            # Pass current_date, period_counter, and total_periods for diagnostics
            predicted_factor_changes = detect_regime_and_predict_factors(
                factor_data, current_date=current_date, period_counter=i, total_periods=total_periods
            )
            all_predictions_history.append({
                'date': current_date,
                'predictions': predicted_factor_changes.copy()
            })

            pc_movements = predict_pc_movements(assignments, f_betas, alphas,
                                               predicted_factor_changes, pc_std, k)
            adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)

            positions = optimize_portfolio_market_neutral(
                adjusted_r, betas, valid_stocks, capital,
                max_position_pct=0.15, long_short_ratio=0.65, beta_tolerance=0.1
            )

            next_date = rebalance_dates[i + 1]
            period_prices = prices.loc[current_date:next_date]
            if len(period_prices) < 2:
                continue

            period_returns = {}
            for stock in valid_stocks:
                if stock in period_prices.columns:
                    start_price = period_prices[stock].iloc[0]
                    end_price = period_prices[stock].iloc[-1]
                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        period_returns[stock] = (end_price - start_price) / start_price
                    else:
                        period_returns[stock] = 0
                else:
                    period_returns[stock] = 0

            portfolio_return = 0
            long_return = 0
            short_return = 0
            long_positions = {k: v for k, v in positions.items() if v > 0}
            short_positions = {k: -v for k, v in positions.items() if v < 0}

            long_capital = sum(long_positions.values())
            short_capital = sum(short_positions.values())

            long_positions_count.append(len(long_positions))
            short_positions_count.append(len(short_positions))

            if long_positions:
                long_exp_ret = np.mean([adjusted_r[valid_stocks.index(stock)] for stock in long_positions.keys()])
                avg_long_expected_return.append(long_exp_ret)

            if short_positions:
                short_exp_ret = np.mean([adjusted_r[valid_stocks.index(stock)] for stock in short_positions.keys()])
                avg_short_expected_return.append(short_exp_ret)

            for stock, position in positions.items():
                if abs(position) > 0.01:
                    stock_return = period_returns.get(stock, 0)
                    if position > 0:
                        long_return += (position / long_capital if long_capital > 0 else 0) * stock_return
                    else:
                        short_return += (-position / short_capital if short_capital > 0 else 0) * (-stock_return)
                    portfolio_return += (position / capital) * stock_return

            portfolio_returns.append(portfolio_return)
            long_returns.append(long_return)
            short_returns.append(short_return)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            all_positions_history.append({
                'date': current_date,
                'positions': positions.copy(),
                'expected_returns': dict(zip(valid_stocks, adjusted_r)),
                'period_returns': period_returns.copy(),
                'portfolio_return': portfolio_return
            })

            if portfolio_return > 0:
                winning_periods += 1
            else:
                losing_periods += 1

            current_value = portfolio_values[-1]
            if current_value > peak_value:
                peak_value = current_value
            else:
                drawdown = (peak_value - current_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

            if i == len(rebalance_dates) - 2:
                final_assignments = assignments
                final_r2s = r2s
                final_alphas = alphas
                final_betas = f_betas
                final_Q = Q
                final_returns = returns

                plot_trades_over_time(all_positions_history, rebalance_dates, portfolio_returns)
                plot_regime_predictions(factor_data, predicted_factor_changes, rebalance_dates)

        except Exception as e:
            print(f"Error in period {i}: {e}")
            continue

    if len(portfolio_returns) < 20:
        return None

    portfolio_returns = np.array(portfolio_returns)
    total_return = (portfolio_values[-1] / capital - 1) * 100
    long_return_pct = (np.prod(1 + np.array(long_returns)) - 1) * 100
    short_return_pct = (np.prod(1 + np.array(short_returns)) - 1) * 100

    risk_free_rate = 0.03
    periods_per_year = 52
    portfolio_returns_annual = np.array(portfolio_returns) * periods_per_year
    excess_returns = portfolio_returns_annual - risk_free_rate / periods_per_year
    sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns_annual) if np.std(portfolio_returns_annual) > 0 else 0

    total_rebalance_periods = len(portfolio_returns)
    win_rate = winning_periods / total_rebalance_periods if total_rebalance_periods > 0 else 0

    portfolio_vol = np.std(portfolio_returns) * np.sqrt(periods_per_year)
    annualized_return = total_return / total_years
    calmar_ratio = annualized_return / (max_drawdown * 100) if max_drawdown > 0 else np.inf

    return {
        'total_return': total_return,
        'long_return_pct': long_return_pct,
        'short_return_pct': short_return_pct,
        'final_value': portfolio_values[-1],
        'assignments': final_assignments,
        'r2s': final_r2s,
        'alphas': final_alphas,
        'betas': final_betas,
        'avg_long_positions': np.mean(long_positions_count) if long_positions_count else 0,
        'avg_short_positions': np.mean(short_positions_count) if short_positions_count else 0,
        'avg_long_expected_return': np.mean(avg_long_expected_return) if avg_long_expected_return else 0,
        'avg_short_expected_return': np.mean(avg_short_expected_return) if avg_short_expected_return else 0,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_rebalance_periods': total_rebalance_periods,
        'winning_periods': winning_periods,
        'losing_periods': losing_periods,
        'portfolio_volatility': portfolio_vol,
        'calmar_ratio': calmar_ratio,
        'annualized_return': annualized_return,
        'predictions_history': all_predictions_history
    }



if __name__ == "__main__":
    # Energy sector stocks (updated with TTE, OXY, CNQ)
    stocks = [
        'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
        'COP', 'EOG', 'OXY', 'CNQ', 'APA',
        'MPC', 'PSX', 'VLO', 'PBF', 'DK',
        'KMI', 'WMB', 'OKE', 'ET', 'ENB',
        'SLB', 'HAL', 'BKR', 'FTI', 'NOV',
        'FANG', 'DVN', 'HES', 'CTRA'
    ]

    # Energy-related factors for beta calculation
    factors_tickers = ['XLE', 'XOP', 'OIH', 'VDE', 'IXC']

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    # Fetch stock and factor data
    prices = fetch_data(stocks, factors_tickers, start_date, end_date)

    # Fetch external economic factors with diagnostics
    external_factors = fetch_external_factors(start_date, end_date)

    if prices.empty or external_factors.empty:
        print("Insufficient data retrieved!")
        exit(1)

    print(f"Retrieved data for {len(prices.columns)} tickers")
    print(f"Retrieved external factors: {list(external_factors.columns)}")

    # Run backtest with dynamic factor prediction
    backtest_results = comprehensive_backtest(
        prices, stocks, external_factors,
        lookback_period=252, rebalance_freq='W-FRI', k=8,
        capital=10000.0, total_years=2, min_r2_threshold=0.05
    )

    if backtest_results:
        print("\n=== DYNAMIC MANIFOLD STRATEGY RESULTS ===")
        print(f"Total Profit: {backtest_results['total_return']:.2f}%")
        print(f"Long Positions Profit: {backtest_results['long_return_pct']:.2f}%")
        print(f"Short Positions Profit: {backtest_results['short_return_pct']:.2f}%")
        print(f"Final Value: ${backtest_results['final_value']:,.2f}")

        print(f"\n=== PORTFOLIO COMPOSITION ===")
        print(f"Average Long Positions: {backtest_results['avg_long_positions']:.1f}")
        print(f"Average Short Positions: {backtest_results['avg_short_positions']:.1f}")
        print(f"Average Long Expected Return: {backtest_results['avg_long_expected_return']:.4f}")
        print(f"Average Short Expected Return: {backtest_results['avg_short_expected_return']:.4f}")

        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Total Rebalance Periods: {backtest_results['total_rebalance_periods']}")
        print(f"Winning Periods: {backtest_results['winning_periods']}")
        print(f"Losing Periods: {backtest_results['losing_periods']}")
        print(f"Portfolio Volatility (Annualized): {backtest_results['portfolio_volatility']:.2%}")
        print(f"Calmar Ratio: {backtest_results['calmar_ratio']:.2f}")
        print(f"Annualized Return: {backtest_results['annualized_return']:.2f}%")

        print("\n=== PC-FACTOR CORRELATIONS ===")
        for j in range(8):
            factor = backtest_results['assignments'].get(j, 'Residual')
            r2 = backtest_results['r2s'].get(j, 0.0)
            alpha = backtest_results['alphas'].get(j, 0.0)
            beta = backtest_results['betas'].get(j, 0.0)
            print(f"PC{j}: {factor}, R² = {r2:.4f}, Equation: PC{j} = {alpha:.4f} + {beta:.4f} × {factor}")

        print("\n=== SAMPLE DYNAMIC PREDICTIONS ===")
        if backtest_results['predictions_history']:
            recent_predictions = backtest_results['predictions_history'][-3:]
            for pred_data in recent_predictions:
                print(f"Date: {pred_data['date'].strftime('%Y-%m-%d')}")
                for factor, change in pred_data['predictions'].items():
                    print(f"  {factor}: {change:.4f}")
                print()
    else:
        print("Backtest failed - insufficient data or other error")