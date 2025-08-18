import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as pulp_value
import matplotlib.pyplot as plt
import warnings
import pulp
from scipy import stats

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


def calculate_predicted_changes(prices, factors_tickers, level_factors, price_factors, external_predictions=None):
    """
    Return predicted changes for factors based on external manifold analysis.
    Predictions are provided as a dictionary of factor changes (e.g., {'^VIX': 10.0, 'XLE': 0.02}).
    Level factors (e.g., VIX, ^TNX) are in absolute changes (points), price factors (e.g., XLE) in percentage changes.
    If no prediction is provided for a factor, defaults to 0.0. Ensures sufficient data for each factor.
    """
    predicted_changes = {}
    external_predictions = external_predictions or {}

    for factor in factors_tickers:
        if factor in prices.columns and len(prices[factor].dropna()) >= 50:
            predicted_changes[factor] = external_predictions.get(factor, 0.0)
        else:
            predicted_changes[factor] = 0.0

    return predicted_changes


def analyze_factor_regime(prices, factor, lookback_days=60):
    """
    Analyze current regime for a specific factor
    Returns regime classification and strength
    """
    if factor not in prices.columns:
        return 'neutral', 0.0

    factor_data = prices[factor].dropna()
    if len(factor_data) < lookback_days:
        return 'neutral', 0.0

    returns = factor_data.pct_change().dropna()
    recent_returns = returns.tail(lookback_days)

    # Calculate regime metrics
    mean_return = recent_returns.mean()
    volatility = recent_returns.std()
    skewness = recent_returns.skew()

    # Classify regime
    if abs(mean_return) < 0.001:
        regime = 'neutral'
    elif mean_return > 0:
        regime = 'uptrend'
    else:
        regime = 'downtrend'

    # Calculate regime strength (0-1)
    strength = min(abs(mean_return) / (volatility + 1e-8), 1.0)

    return regime, strength


def prepare_returns_and_factors(prices, stocks, level_factors, price_factors,
                                lookback_period=252, min_observations=100):
    """Prepare returns and factor data"""
    valid_stocks = [stock for stock in stocks if
                    stock in prices.columns and len(prices[stock].dropna()) >= min_observations]

    if len(valid_stocks) < 10:
        raise ValueError(f"Insufficient valid stocks: {len(valid_stocks)}")

    stock_prices = prices[valid_stocks].dropna()
    returns = stock_prices.pct_change().dropna().tail(lookback_period)
    factor_data = pd.DataFrame(index=returns.index)

    for factor in price_factors:
        if factor in prices.columns and len(prices[factor].dropna()) >= min_observations:
            factor_data[factor] = prices[factor].pct_change().reindex(returns.index)

    for factor in level_factors:
        if factor in prices.columns and len(prices[factor].dropna()) >= min_observations:
            factor_data[factor] = prices[factor].diff().reindex(returns.index)

    factor_data = factor_data.dropna(thresh=len(factor_data) * 0.9, axis=1)
    factor_data = factor_data.dropna()
    returns = returns.reindex(factor_data.index).dropna()

    if len(returns) < 100:
        raise ValueError(f"Insufficient aligned observations: {len(returns)}")

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
    """
    Calculate network centrality with specified threshold

    FROM STRATEGY: "Create binary adjacency matrix A where:
    A_{ij} = 1 if |correlation_{ij}| ≥ 0.6
    A_{ij} = 0 otherwise
    A_{ii} = 1 (diagonal elements)

    Eigenvector Centrality Calculation:
    Solve eigenvalue problem: Av = λv
    Extract eigenvector v₁ corresponding to largest eigenvalue λ₁
    Normalize: c = v₁ / ||v₁|| where c is centrality vector"
    """
    # Step 1: Calculate correlation matrix
    corr_matrix = returns.corr().fillna(0).values

    # Step 2: Create binary adjacency matrix A
    adj_matrix = np.abs(corr_matrix) >= corr_threshold
    adj_matrix = adj_matrix.astype(float)
    np.fill_diagonal(adj_matrix, 1.0)  # A_{ii} = 1

    try:
        # Step 3: Solve eigenvalue problem: Av = λv
        eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
        max_eig_idx = np.argmax(np.real(eigenvalues))

        # Step 4: Extract eigenvector v₁ corresponding to largest eigenvalue λ₁
        centrality_vector = np.abs(np.real(eigenvectors[:, max_eig_idx]))

        # Step 5: Normalize: c = v₁ / ||v₁||
        if np.sum(centrality_vector) > 0:
            centrality_vector /= np.sum(centrality_vector)
        else:
            centrality_vector = np.ones(len(stocks)) / len(stocks)
    except:
        # Fallback to degree centrality
        centrality_vector = np.sum(adj_matrix, axis=1)
        if np.sum(centrality_vector) > 0:
            centrality_vector /= np.sum(centrality_vector)
        else:
            centrality_vector = np.ones(len(stocks)) / len(stocks)

    centrality = dict(zip(stocks, centrality_vector))
    return centrality, centrality_vector


def perform_pca(returns, k=8):
    """
    Perform PCA with standardized returns

    FROM STRATEGY: "Calculate correlation matrix R from return data
    Solve eigenvalue decomposition: R = QΛQ^T
    Q = matrix of eigenvectors (factor loadings)
    Λ = diagonal matrix of eigenvalues (factor variances)
    Extract first k principal components (typically k=5-8)
    Calculate daily PC time series: PC_daily = Returns × Q"
    """
    returns_standardized = (returns - returns.mean()) / returns.std()

    # Calculate correlation matrix R
    corr_matrix = returns_standardized.corr().fillna(0).values

    # Solve eigenvalue decomposition: R = QΛQ^T
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    sorted_idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[sorted_idx])
    eigenvectors = np.real(eigenvectors[:, sorted_idx])

    k_final = min(k, len(eigenvalues))
    Q = eigenvectors[:, :k_final]  # Q = matrix of eigenvectors (factor loadings)

    # Calculate daily PC time series: PC_daily = Returns × Q
    pc_time_series = returns_standardized.fillna(0).values @ Q

    # Calculate σ_{PC_j}: Historical standard deviation of each PC
    pc_std = np.std(pc_time_series, axis=0, ddof=1)
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)

    return Q, pc_time_series, pc_std, explained_var_ratio[:k_final]


def identify_factors_optimized(pc_time_series, factor_data, k, min_r2_threshold=0.05):
    """
    Map PCs to economic factors using linear regression

    FROM STRATEGY: "Run linear regression: PC_j = α_j + β_j × Factor_k + ε_j
    Calculate R² to measure explanatory power

    Linear Regression Mathematics:
    β_j = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²] where x = factor changes, y = PC values
    α_j = ȳ - β_j × x̄
    R² = 1 - (SS_res / SS_tot)"
    """
    factor_names = factor_data.columns.tolist()
    assignments = {}
    betas = {}
    alphas = {}
    r2s = {}
    used_factors = set()

    for j in range(k):
        y = pc_time_series[:, j]
        best_factor = 'Residual'
        best_r2 = 0
        best_beta = 0
        best_alpha = 0

        for factor_name in factor_names:
            if factor_name in used_factors:
                continue
            try:
                factor_series = factor_data[factor_name]
                combined_data = pd.DataFrame({'pc': y, 'factor': factor_series}, index=factor_data.index)
                combined_clean = combined_data.dropna()
                if len(combined_clean) < 50:
                    continue

                y_clean = combined_clean['pc'].values
                x_clean = combined_clean['factor'].values
                if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                    continue

                # FROM STRATEGY: β_j = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]
                x_mean = np.mean(x_clean)
                y_mean = np.mean(y_clean)
                numerator = np.sum((x_clean - x_mean) * (y_clean - y_mean))
                denominator = np.sum((x_clean - x_mean) ** 2)
                beta = numerator / denominator if denominator > 1e-8 else 0

                # FROM STRATEGY: α_j = ȳ - β_j × x̄
                alpha = y_mean - beta * x_mean

                # FROM STRATEGY: R² = 1 - (SS_res / SS_tot)
                y_pred = alpha + beta * x_clean
                ss_res = np.sum((y_clean - y_pred) ** 2)
                ss_tot = np.sum((y_clean - y_mean) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    best_factor = factor_name
                    best_beta = beta
                    best_alpha = alpha
            except:
                continue

        assignments[j] = best_factor
        betas[j] = best_beta
        alphas[j] = best_alpha
        r2s[j] = best_r2
        if best_factor != 'Residual':
            used_factors.add(best_factor)

    return assignments, betas, alphas, r2s


def predict_pc_movements(assignments, betas, alphas, predicted_changes, pc_std, k):
    """
    Predict PC movements

    FROM STRATEGY: "Apply regression equations: ΔPC_j = β_j × Δfactor_k
    Convert to percentage terms: PC_j movement (%) = ΔPC_j × σ_{PC_j}"
    """
    pc_movements = np.zeros(k)
    for j in range(k):
        factor = assignments[j]
        if factor in predicted_changes and factor != 'Residual':
            delta_factor = predicted_changes[factor]
            # FROM STRATEGY: ΔPC_j = β_j × Δfactor_k (plus intercept)
            raw_pc_change = alphas[j] + betas[j] * delta_factor
            # FROM STRATEGY: PC_j movement (%) = ΔPC_j × σ_{PC_j}
            pc_movements[j] = raw_pc_change * pc_std[j] if pc_std[j] > 1e-8 else raw_pc_change
    return pc_movements


def compute_expected_returns(Q, pc_movements_percent, centrality, stocks):
    """
    Compute expected returns with centrality weighting

    FROM STRATEGY: "Mathematical Formula: r_i = Σ_j (Q_{ij} × PC_j movement)
    With Network Weighting: Adjusted return_i = r_i × c_i"
    """
    # FROM STRATEGY: r_i = Σ_j (Q_{ij} × PC_j movement) - matrix multiplication
    raw_returns = Q @ pc_movements_percent

    # FROM STRATEGY: Adjusted return_i = r_i × c_i (element-wise multiplication)
    c_vector = np.array([centrality[stock] for stock in stocks])
    adjusted_returns = raw_returns * c_vector
    return adjusted_returns


def optimize_portfolio_improved_market_neutral(adjusted_r, betas, stocks, capital=10000.0,
                                               max_position_pct=0.15, long_short_ratio=0.65,
                                               beta_tolerance=0.1):
    """
    Improved market neutral portfolio optimization with better short selection

    FROM STRATEGY: "Optimization Problem:
    Maximize: (r ⊙ c) · v  [Note: adjusted_r already contains r ⊙ c]
    Subject to:
    v · b = 0 (beta-neutral constraint)
    Σ|w_i| = $10,000 (capital constraint)

    Only shorting stocks with negative expected returns.
    65%/35% long/short ratio.
    Beta tolerance of 0.1.
    Position limits (15% long, 12% short, 25% short concentration)."
    """
    prob = LpProblem("Improved_Market_Neutral_Portfolio", LpMaximize)

    v_long = {stock: LpVariable(f"v_long_{stock}", lowBound=0, upBound=capital * max_position_pct)
              for stock in stocks}
    v_short = {stock: LpVariable(f"v_short_{stock}", lowBound=0, upBound=capital * max_position_pct * 0.8)
               for stock in stocks}

    # FROM STRATEGY: Maximize: (r ⊙ c) · v [adjusted_r already contains this]
    prob += lpSum([adjusted_r[i] * (v_long[stocks[i]] - v_short[stocks[i]]) for i in range(len(stocks))])

    # FROM STRATEGY: "65%/35% long/short ratio"
    prob += lpSum([v_long[stock] for stock in stocks]) == capital * long_short_ratio
    prob += lpSum([v_short[stock] for stock in stocks]) == capital * (1 - long_short_ratio)

    # FROM STRATEGY: "v · b = 0 (beta-neutral constraint)" with "Beta tolerance of 0.1"
    beta_exposure = lpSum([betas[i] * (v_long[stocks[i]] - v_short[stocks[i]]) for i in range(len(stocks))])
    prob += beta_exposure <= capital * beta_tolerance
    prob += beta_exposure >= -capital * beta_tolerance

    # FROM STRATEGY: "Only shorting stocks with negative expected returns"
    for i, stock in enumerate(stocks):
        if adjusted_r[i] >= -0.001:
            prob += v_short[stock] == 0

    # FROM STRATEGY: "25% short concentration"
    total_short_positions = lpSum([v_short[stock] for stock in stocks if adjusted_r[stocks.index(stock)] < -0.001])
    for stock in stocks:
        if adjusted_r[stocks.index(stock)] < -0.001:
            prob += v_short[stock] <= total_short_positions * 0.25

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    positions = {}
    for stock in stocks:
        long_val = pulp_value(v_long[stock]) if v_long[stock].value() else 0.0
        short_val = pulp_value(v_short[stock]) if v_short[stock].value() else 0.0
        net_position = long_val - short_val
        positions[stock] = net_position if abs(net_position) > 0.01 else 0.0

    return positions


def plot_prediction_analysis(prices, predicted_changes, factors_tickers):
    """Plot prediction analysis showing recent trends and predictions"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    plotted = 0
    for factor in factors_tickers[:9]:  # Plot first 9 factors
        if factor in prices.columns and factor in predicted_changes:
            ax = axes[plotted]

            # Plot recent price history
            factor_data = prices[factor].dropna().tail(60)
            ax.plot(factor_data.index, factor_data.values, 'b-', linewidth=1.5, label='Price History')

            # Show prediction as arrow
            last_price = factor_data.iloc[-1]
            predicted_change = predicted_changes[factor]

            if abs(predicted_change) > 1e-6:  # Only show meaningful predictions
                if factor in ['OVX']:  # Level factors
                    predicted_price = last_price + predicted_change
                else:  # Return factors
                    predicted_price = last_price * (1 + predicted_change)

                ax.annotate('', xy=(factor_data.index[-1], predicted_price),
                            xytext=(factor_data.index[-1], last_price),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))
                ax.text(factor_data.index[-1], predicted_price,
                        f'Pred: {predicted_change:.3f}',
                        ha='right', va='bottom', fontsize=8, color='red')

            ax.set_title(f'{factor}', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3)

            plotted += 1

    # Hide unused subplots
    for i in range(plotted, 9):
        axes[i].set_visible(False)

    plt.suptitle('Factor Trend Analysis and Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()


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


def plot_buys_shorts_over_time(portfolio_returns, long_returns, short_returns, rebalance_dates):
    """Plot cumulative returns of long and short positions over time"""
    plt.figure(figsize=(12, 6))
    plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(long_returns)) - 1,
             label='Long Returns', color='green', linewidth=2)
    plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(short_returns)) - 1,
             label='Short Returns', color='red', linewidth=2)
    plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(portfolio_returns)) - 1,
             label='Portfolio Returns', color='blue', linewidth=2)
    plt.title('Cumulative Returns: Long, Short, and Portfolio', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_centrality_matrix(returns, stocks, corr_threshold=0.6):
    """Plot correlation-based adjacency matrix"""
    corr_matrix = returns.corr().fillna(0).values
    adj_matrix = np.abs(corr_matrix) >= corr_threshold
    adj_matrix = adj_matrix.astype(float)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(adj_matrix, cmap='RdBu_r', interpolation='none', aspect='auto')
    plt.colorbar(im, label='Connection (1 = |corr| ≥ 0.6)')
    plt.xticks(np.arange(len(stocks)), stocks, rotation=45, ha='right')
    plt.yticks(np.arange(len(stocks)), stocks)
    plt.title('Network Centrality Matrix (Stock Correlations)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_pcas_over_time(pc_time_series, assignments, k):
    """Plot PC time series"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for j in range(k):
        ax = axes[j]
        ax.plot(pc_time_series[:, j], color=f'C{j}', linewidth=1.5)
        ax.set_title(f'PC{j}: {assignments.get(j, "Unknown")}', fontsize=10)
        ax.set_xlabel('Time Index', fontsize=8)
        ax.set_ylabel('PC Value', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Principal Component Time Series', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_factor_loadings(Q, stocks, k=8):
    """Plot factor loadings heatmap"""
    plt.figure(figsize=(14, 8))
    im = plt.imshow(Q[:, :k].T, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, label='Loading')
    plt.yticks(np.arange(k), [f'PC{i}' for i in range(k)])
    plt.xticks(np.arange(len(stocks)), stocks, rotation=45, ha='right')
    plt.title('Principal Component Loadings', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe_ratio(portfolio_returns, window=20):
    """Plot rolling Sharpe ratio"""
    returns_series = pd.Series(portfolio_returns)
    rolling_mean = returns_series.rolling(window=window).mean() * 52  # Annualized
    rolling_std = returns_series.rolling(window=window).std() * np.sqrt(52)  # Annualized
    rolling_sharpe = rolling_mean / rolling_std

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Rolling Sharpe Ratio (Window: {window} periods)', fontsize=14)
    plt.xlabel('Rebalancing Period', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown_analysis(portfolio_returns):
    """Plot drawdown analysis"""
    portfolio_values = np.cumprod(1 + np.array(portfolio_returns))
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value
    ax1.plot(portfolio_values, color='blue', linewidth=2, label='Portfolio Value')
    ax1.plot(peak, color='red', linewidth=1, alpha=0.7, label='Running Peak')
    ax1.set_title('Portfolio Value and Running Peak', fontsize=12)
    ax1.set_ylabel('Portfolio Value', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=2)
    ax2.set_title('Drawdown Analysis', fontsize=12)
    ax2.set_xlabel('Rebalancing Period', fontsize=10)
    ax2.set_ylabel('Drawdown', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def comprehensive_backtest(prices, stocks, level_factors, price_factors,
                           lookback_period=252, rebalance_freq='W-FRI', k=8, capital=10000.0,
                           total_years=10, min_r2_threshold=0.05):
    """
    Comprehensive backtest following the exact strategy outline

    FROM STRATEGY: "Expected returns vector: r = [3.2%, 1.8%, 2.1%, ...] (your PC-based predictions)
    Position weights vector: v = [w₁, w₂, w₃, ...]
    total portfolio returns = r dotted with v

    (r element wise multiplication c) dot v = total portfolio returns"
    """
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
    daily_returns = []
    final_Q = None
    final_returns = None

    spot_check_data = []

    for i, current_date in enumerate(rebalance_dates[:-1]):
        try:
            lookback_start = current_date - timedelta(days=int(lookback_period * 4))
            hist_prices = prices.loc[lookback_start:current_date]
            if len(hist_prices) < lookback_period // 2:
                continue

            # Calculate predicted changes based on external manifold analysis
            factors_tickers = level_factors + price_factors
            predicted_changes = calculate_predicted_changes(
                hist_prices, factors_tickers, level_factors, price_factors,
                external_predictions=None  # Let function use defaults (0.0) as per strategy
            )

            all_predictions_history.append({
                'date': current_date,
                'predictions': predicted_changes.copy()
            })

            returns, factor_data, valid_stocks = prepare_returns_and_factors(
                hist_prices, stocks, level_factors, price_factors, lookback_period
            )
            if len(returns) < 100 or len(valid_stocks) < 10:
                continue

            market_returns = factor_data['XLE'] if 'XLE' in factor_data.columns else factor_data.iloc[:, 0]
            betas = compute_betas(returns, market_returns, valid_stocks)
            centrality, centrality_vector = compute_network_centrality(returns, valid_stocks, corr_threshold=0.6)
            Q, pc_time_series, pc_std, _ = perform_pca(returns, k)
            assignments, f_betas, alphas, r2s = identify_factors_optimized(
                pc_time_series, factor_data, k, min_r2_threshold
            )
            pc_movements = predict_pc_movements(assignments, f_betas, alphas, predicted_changes, pc_std, k)

            # FROM STRATEGY: "Expected return from your PC analysis: r
            # Network centrality score: c
            # (r element wise multiplication c) dot v = total portfolio returns"
            adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)

            positions = optimize_portfolio_improved_market_neutral(
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

            # FROM STRATEGY: "total portfolio returns = r dotted with v"
            # where r is expected returns and v is position weights
            portfolio_return = 0
            long_return = 0
            short_return = 0
            long_positions = {k: v for k, v in positions.items() if v > 0}
            short_positions = {k: -v for k, v in positions.items() if v < 0}
            print(
                f'Rebalancing on {current_date} with {len(long_positions)} long positions and {len(short_positions)} short positions')
            print(f'Long positions: {long_positions}')
            print(f'Short positions: {short_positions}')
            print()
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

            daily_returns.append(portfolio_return)

            if i % 10 == 0:
                spot_check_entry = {
                    'period': i,
                    'date': current_date,
                    'next_date': next_date,
                    'sample_positions': dict(list(positions.items())[:3]),
                    'sample_period_returns': dict(list(period_returns.items())[:3]),
                    'sample_predictions': dict(list(predicted_changes.items())[:3]),
                    'portfolio_return': portfolio_return,
                    'portfolio_value': portfolio_values[-1]
                }
                spot_check_data.append(spot_check_entry)

            if i == len(rebalance_dates) - 2:
                final_assignments = assignments
                final_r2s = r2s
                final_alphas = alphas
                final_betas = f_betas
                final_Q = Q
                final_returns = returns

                # Create all plots including prediction analysis
                plot_prediction_analysis(hist_prices, predicted_changes, factors_tickers)
                plot_buys_shorts_over_time(portfolio_returns, long_returns, short_returns, rebalance_dates)
                plot_centrality_matrix(returns, valid_stocks, corr_threshold=0.6)
                plot_pcas_over_time(pc_time_series, assignments, k)
                plot_trades_over_time(all_positions_history, rebalance_dates, portfolio_returns)
                plot_factor_loadings(Q, valid_stocks, k)
                plot_rolling_sharpe_ratio(portfolio_returns, window=20)
                plot_drawdown_analysis(portfolio_returns)

        except Exception as e:
            continue

    if len(portfolio_returns) < 20:
        return None

    total_trading_days = (rebalance_dates[-1] - rebalance_dates[0]).days
    portfolio_returns = np.array(portfolio_returns)
    total_return = (portfolio_values[-1] / capital - 1) * 100
    long_return_pct = (np.prod(1 + np.array(long_returns)) - 1) * 100
    short_return_pct = (np.prod(1 + np.array(short_returns)) - 1) * 100

    risk_free_rate = 0.03
    periods_per_year = 52
    portfolio_returns_annual = np.array(portfolio_returns) * periods_per_year
    excess_returns = portfolio_returns_annual - risk_free_rate / periods_per_year
    sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns_annual) if np.std(
        portfolio_returns_annual) > 0 else 0

    total_rebalance_periods = len(portfolio_returns)
    win_rate = winning_periods / total_rebalance_periods if total_rebalance_periods > 0 else 0

    total_position_changes = 0
    prev_positions = {}
    for period_data in all_positions_history:
        current_positions = period_data['positions']
        for stock in set(list(prev_positions.keys()) + list(current_positions.keys())):
            prev_pos = prev_positions.get(stock, 0)
            curr_pos = current_positions.get(stock, 0)
            if abs(curr_pos - prev_pos) > 0.01:
                total_position_changes += 1
        prev_positions = current_positions.copy()

    portfolio_vol = np.std(portfolio_returns) * np.sqrt(periods_per_year)
    annualized_return = total_return / (total_years)
    calmar_ratio = annualized_return / (max_drawdown * 100) if max_drawdown > 0 else np.inf

    # Calculate prediction accuracy metrics
    prediction_accuracy = {}
    if len(all_predictions_history) > 10:
        # Sample some predictions to analyze
        for factor in factors_tickers[:5]:  # Analyze first 5 factors
            if factor in prices.columns:
                factor_predictions = []
                factor_actuals = []

                for j, pred_data in enumerate(all_predictions_history):
                    if j >= len(all_positions_history) - 1:
                        break

                    pred_date = pred_data['date']
                    next_date = rebalance_dates[j + 1] if j + 1 < len(rebalance_dates) else None

                    if next_date and factor in pred_data['predictions']:
                        pred_change = pred_data['predictions'][factor]

                        # Calculate actual change
                        if factor in prices.columns:
                            pred_price = prices[factor].loc[pred_date]
                            actual_price = prices[factor].loc[next_date]

                            if pd.notna(pred_price) and pd.notna(actual_price) and pred_price > 0:
                                if factor in level_factors:
                                    actual_change = actual_price - pred_price
                                else:
                                    actual_change = (actual_price - pred_price) / pred_price

                                factor_predictions.append(pred_change)
                                factor_actuals.append(actual_change)

                if len(factor_predictions) > 5:
                    # Calculate correlation between predictions and actuals
                    correlation = np.corrcoef(factor_predictions, factor_actuals)[0, 1] if len(
                        factor_predictions) > 1 else 0
                    prediction_accuracy[factor] = {
                        'correlation': correlation,
                        'predictions_count': len(factor_predictions),
                        'avg_predicted': np.mean(factor_predictions),
                        'avg_actual': np.mean(factor_actuals)
                    }

    return {
        'total_return': total_return,
        'long_return_pct': long_return_pct,
        'short_return_pct': short_return_pct,
        'total_trading_days': total_trading_days,
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
        'total_position_changes': total_position_changes,
        'portfolio_volatility': portfolio_vol,
        'calmar_ratio': calmar_ratio,
        'annualized_return': annualized_return,
        'prediction_accuracy': prediction_accuracy,
        'spot_check_data': spot_check_data
    }


if __name__ == "__main__":
    # Updated stock list - energy sector stocks
    stocks = [
        'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
        'COP', 'EOG', 'DVN', 'APA',
        'MPC', 'PSX', 'VLO', 'PBF', 'DK',
        'KMI', 'WMB', 'OKE', 'ET', 'ENB',
        'SLB', 'HAL', 'BKR', 'FTI', 'NOV',
        'FANG',  # Added as a Pioneer proxy
        'HES', 'CTRA'
    ]

    factors_tickers = [
        'XLE', 'XOP', 'OIH', 'VDE', 'IXC',
        'CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F',
        'ICLN', 'TAN', 'FAN', 'PBW', 'QCLN',
        'CRAK', 'PXE', 'FCG', 'MLPX', 'AMLP',
        'FENY', 'OILK', 'USO', 'BNO', 'UNG',
        '^SP500-15', '^DJUSEN', '^XOI', '^OSX',
        'ENOR', 'ENZL', 'KWT', 'GEX', 'URA',
        'RSPG',
        # New factors
        '^TNX', '^VIX', 'COAL', 'URA',
        'XES', 'IEO', 'PXI', 'TIP', 'GLD'
    ]

    level_factors = ['^VIX', '^TNX']
    price_factors = [f for f in factors_tickers if f not in level_factors]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 4)
    prices = fetch_data(stocks, factors_tickers, start_date, end_date)

    if prices.empty:
        print("No price data retrieved!")
        exit(1)

    print("=== RUNNING EXACT STRATEGY IMPLEMENTATION ===")
    print("Following the mathematical framework exactly:")
    print("1. Network centrality: c = v₁ / ||v₁|| where Av = λv")
    print("2. Linear regression: PC_j = α_j + β_j × Factor_k")
    print("3. PC movements: ΔPC_j = β_j × Δfactor_k")
    print("4. Stock returns: r_i = Σ_j (Q_{ij} × PC_j movement)")
    print("5. Network weighting: Adjusted return_i = r_i × c_i")
    print("6. Portfolio optimization: Maximize (r ⊙ c) · v")
    print("7. Position calculation: total portfolio returns = r dotted with v")
    print()

    backtest_results = comprehensive_backtest(
        prices, stocks, level_factors, price_factors,
        lookback_period=252, rebalance_freq='W-FRI', k=8,
        capital=10000.0, total_years=3, min_r2_threshold=0.05
    )

    if backtest_results:
        print("\n=== EXACT STRATEGY RESULTS ===")
        print(
            f"Total Profit: {backtest_results['total_return']:.2f}% over {backtest_results['total_trading_days']} trading days")
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
        print(f"Total Position Changes: {backtest_results['total_position_changes']}")
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

        print("\n=== PREDICTION ACCURACY ANALYSIS ===")
        if 'prediction_accuracy' in backtest_results and backtest_results['prediction_accuracy']:
            for factor, accuracy in backtest_results['prediction_accuracy'].items():
                print(f"{factor}: Correlation = {accuracy['correlation']:.3f}, "
                      f"Predictions = {accuracy['predictions_count']}, "
                      f"Avg Pred = {accuracy['avg_predicted']:.4f}, "
                      f"Avg Actual = {accuracy['avg_actual']:.4f}")
        else:
            print("Prediction accuracy data not available")

        print("\n=== SAMPLE SPOT CHECKS ===")
        for spot_check in backtest_results['spot_check_data'][:3]:
            print(f"Period {spot_check['period']} ({spot_check['date'].strftime('%Y-%m-%d')}):")
            print(f"  Sample Positions: {spot_check['sample_positions']}")
            print(f"  Sample Returns: {spot_check['sample_period_returns']}")
            print(f"  Sample Predictions: {spot_check['sample_predictions']}")
            print(f"  Portfolio Return: {spot_check['portfolio_return']:.4f}")
            print(f"  Portfolio Value: ${spot_check['portfolio_value']:.2f}")
            print()
    else:
        print("Backtest failed!")