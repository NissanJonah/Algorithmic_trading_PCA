import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as pulp_value
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
import warnings
import pulp

warnings.filterwarnings('ignore')

def fetch_data(stocks, factors_tickers, start_date, end_date, max_retries=3):
    """Fetch price data for stocks and factors with retry logic"""
    prices = pd.DataFrame()
    failed_tickers = []

    print(f"Fetching data for {len(stocks + factors_tickers)} tickers...")

    for ticker in stocks + factors_tickers:
        success = False
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if 'Close' in df.columns and not df['Close'].empty and len(df) > 50:
                    prices[ticker] = df['Close']
                    success = True
                    break
            except Exception as e:
                pass

        if not success:
            failed_tickers.append(ticker)

    if failed_tickers:
        print(f"Failed to download: {failed_tickers}")

    return prices


def prepare_returns_and_factors(prices, stocks, level_factors, price_factors,
                                lookback_period=252, min_observations=100):
    """Prepare returns and factor data with enhanced validation"""
    # Filter valid stocks with sufficient data
    valid_stocks = []
    for stock in stocks:
        if (stock in prices.columns and
                not prices[stock].dropna().empty and
                len(prices[stock].dropna()) >= min_observations):
            valid_stocks.append(stock)

    if len(valid_stocks) < 10:
        raise ValueError(f"Insufficient valid stocks: {len(valid_stocks)}")

    # Calculate returns for stocks
    stock_prices = prices[valid_stocks].dropna()
    returns = stock_prices.pct_change().dropna().tail(lookback_period)

    # Prepare factor data
    factor_data = pd.DataFrame(index=returns.index)

    # Price-based factors (use returns)
    for factor in price_factors:
        if factor in prices.columns and len(prices[factor].dropna()) >= min_observations:
            factor_returns = prices[factor].pct_change()
            factor_data[factor] = factor_returns.reindex(returns.index)

    # Level-based factors (use changes)
    for factor in level_factors:
        if factor in prices.columns and len(prices[factor].dropna()) >= min_observations:
            factor_changes = prices[factor].diff()
            factor_data[factor] = factor_changes.reindex(returns.index)

    # Remove factors with too many NaN values
    factor_data = factor_data.dropna(thresh=len(factor_data) * 0.9, axis=1)

    # Align datasets
    factor_data = factor_data.dropna()
    returns = returns.reindex(factor_data.index).dropna()

    if len(returns) < 100:
        raise ValueError(f"Insufficient aligned observations: {len(returns)}")

    print(f"Data prepared: {len(returns)} observations, {len(valid_stocks)} stocks, {len(factor_data.columns)} factors")
    return returns, factor_data, valid_stocks

def compute_betas(returns, market_returns, stocks, min_r_squared=0.05):
    """Calculate beta for each stock relative to market with validation"""
    betas = []
    beta_r2s = []

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

                if model.rsquared >= min_r_squared:
                    betas.append(float(model.params.iloc[1]))
                    beta_r2s.append(model.rsquared)
                else:
                    betas.append(1.0)
                    beta_r2s.append(0.0)
            else:
                betas.append(1.0)
                beta_r2s.append(0.0)
        except:
            betas.append(1.0)
            beta_r2s.append(0.0)

    return np.array(betas)

def compute_network_centrality(returns, stocks, corr_threshold=0.5):
    """Calculate network centrality with improved threshold"""
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
    # Standardize returns for better PCA results
    returns_standardized = (returns - returns.mean()) / returns.std()
    corr_matrix = returns_standardized.corr().fillna(0).values

    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    sorted_idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[sorted_idx])
    eigenvectors = np.real(eigenvectors[:, sorted_idx])

    k_final = min(k, len(eigenvalues))
    Q = eigenvectors[:, :k_final]

    # Calculate PC time series using standardized returns
    pc_time_series = returns_standardized.fillna(0).values @ Q
    pc_std = np.std(pc_time_series, axis=0, ddof=1)

    explained_var_ratio = eigenvalues / np.sum(eigenvalues)

    return Q, pc_time_series, pc_std, explained_var_ratio[:k_final]


def identify_factors_optimized(pc_time_series, factor_data, k, min_r2_threshold=0.15):
    """
    Optimized factor identification with higher R² threshold and better matching
    """
    factor_names = factor_data.columns.tolist()
    assignments = {}
    betas = {}
    alphas = {}
    r2s = {}

    print(f"\n=== FACTOR-PC CORRELATION ANALYSIS ===")
    print(f"Minimum R² threshold: {min_r2_threshold:.3f}")
    print("-" * 60)

    # Track used factors to avoid double assignment
    used_factors = set()

    for j in range(k):
        y = pc_time_series[:, j]
        best_factor = None
        best_r2 = 0
        best_beta = 0
        best_alpha = 0

        print(f"\nPC{j} Factor Analysis:")
        factor_results = []

        for factor_name in factor_names:
            if factor_name in used_factors:
                continue

            try:
                factor_series = factor_data[factor_name]
                combined_data = pd.DataFrame({
                    'pc': y,
                    'factor': factor_series
                }, index=factor_data.index)

                combined_clean = combined_data.dropna()

                if len(combined_clean) < 50:
                    continue

                y_clean = combined_clean['pc'].values
                x_clean = combined_clean['factor'].values

                if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                    continue

                X_with_const = sm.add_constant(x_clean)
                model = sm.OLS(y_clean, X_with_const, missing='drop').fit()

                r2 = model.rsquared
                beta_coef = float(model.params[1])
                alpha_coef = float(model.params[0])

                factor_results.append({
                    'factor': factor_name,
                    'r2': r2,
                    'beta': beta_coef,
                    'alpha': alpha_coef
                })

                if r2 > best_r2:
                    best_r2 = r2
                    best_factor = factor_name
                    best_beta = beta_coef
                    best_alpha = alpha_coef

            except:
                continue

        # Sort and display top candidates
        factor_results.sort(key=lambda x: x['r2'], reverse=True)
        print(f"  Top 5 factor candidates:")
        for i, result in enumerate(factor_results[:5]):
            print(f"    {i + 1}. {result['factor']}: R²={result['r2']:.4f}, β={result['beta']:.4f}")

        # Assignment logic
        if best_r2 >= min_r2_threshold:
            assignments[j] = best_factor
            betas[j] = best_beta
            alphas[j] = best_alpha
            r2s[j] = best_r2
            used_factors.add(best_factor)
            print(f"  ✅ ASSIGNED: {best_factor} (R²={best_r2:.4f})")
        else:
            assignments[j] = 'Residual'
            betas[j] = 0.0
            alphas[j] = 0.0
            r2s[j] = 0.0
            print(f"  ❌ NO ASSIGNMENT: Best R²={best_r2:.4f} < {min_r2_threshold:.4f}")

    return assignments, betas, alphas, r2s


def identify_factors_sequential(pc_time_series, factor_data, k, min_r2_threshold=0.05,
                                max_factors_per_pc=1, verbose=True):
    """
    Sequential PC-factor assignment: go through each PC and find best factor

    Parameters:
    - min_r2_threshold: Minimum R² required for assignment
    - max_factors_per_pc: Maximum factors to assign per PC
    - verbose: Print detailed assignment process
    """
    factor_names = factor_data.columns.tolist()
    assignments = {}
    betas = {}
    alphas = {}
    r2s = {}

    if verbose:
        print(f"\nSequential Factor Assignment (Threshold R² = {min_r2_threshold:.3f})")
        print("-" * 60)

    # Go through each PC sequentially
    for j in range(k):
        y = pc_time_series[:, j]
        best_factors = []

        if verbose:
            print(f"\nPC{j} Factor Analysis:")

        # Calculate R² for all factors
        factor_scores = []

        for factor_name in factor_names:
            try:
                # Get factor data as pandas Series to properly handle index alignment
                factor_series = factor_data[factor_name]

                # Create DataFrame with both PC and factor data
                combined_data = pd.DataFrame({
                    'pc': y,
                    'factor': factor_series
                }, index=factor_data.index)

                # Drop NaN values
                combined_clean = combined_data.dropna()

                # Skip if insufficient data
                if len(combined_clean) < 20:
                    continue

                y_clean = combined_clean['pc'].values
                x_clean = combined_clean['factor'].values

                # Skip if no variation
                if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                    continue

                # Perform regression
                X_with_const = sm.add_constant(x_clean)
                model = sm.OLS(y_clean, X_with_const, missing='drop').fit()

                r2 = model.rsquared
                beta_coef = float(model.params[1])  # Use integer indexing
                alpha_coef = float(model.params[0])  # Use integer indexing

                factor_scores.append({
                    'factor': factor_name,
                    'r2': r2,
                    'beta': beta_coef,
                    'alpha': alpha_coef
                })

            except Exception as e:
                if verbose and 'singular' not in str(e).lower():
                    print(f"    Error with {factor_name}: {str(e)[:30]}")
                continue

        # Sort by R² and assign best factor(s)
        factor_scores.sort(key=lambda x: x['r2'], reverse=True)

        if verbose:
            print(f"    Top factor candidates:")
            for i, score in enumerate(factor_scores[:5]):
                print(f"      {i + 1}. {score['factor']}: R²={score['r2']:.3f}, β={score['beta']:.4f}")

        # Assign best factor if it meets threshold
        if factor_scores and factor_scores[0]['r2'] >= min_r2_threshold:
            best = factor_scores[0]
            assignments[j] = best['factor']
            betas[j] = best['beta']
            alphas[j] = best['alpha']
            r2s[j] = best['r2']

            if verbose:
                print(f"    ✓ ASSIGNED: {best['factor']} (R²={best['r2']:.3f})")
        else:
            # No good factor found
            assignments[j] = 'Residual'
            betas[j] = 0.0
            alphas[j] = 0.0
            r2s[j] = 0.0

            if verbose:
                if factor_scores:
                    best_r2 = factor_scores[0]['r2']
                    print(f"    ✗ NO ASSIGNMENT: Best R²={best_r2:.3f} < {min_r2_threshold:.3f}")
                else:
                    print(f"    ✗ NO ASSIGNMENT: No valid factors found")

    return assignments, betas, alphas, r2s


def print_pc_info(assignments, r2s, alphas, betas, k, detailed=True):
    """
    Print detailed PC information with interpretation

    Parameters:
    - detailed: Include interpretation and economic meaning
    """
    print("\n" + "=" * 70)
    print("PRINCIPAL COMPONENT FACTOR ANALYSIS RESULTS")
    print("=" * 70)

    total_explained_r2 = 0
    strong_assignments = 0

    for j in range(k):
        factor = assignments[j]
        alpha = alphas[j]
        beta = betas[j]
        r2 = r2s[j]

        print(f"\n📊 PC{j} Analysis:")
        print(f"   Associated Factor: {factor}")
        print(f"   R² (Explanatory Power): {r2:.3f}")
        print(f"   Regression Equation: PC{j} = {alpha:.4f} + {beta:.4f} × {factor}")

        if detailed:
            # Economic interpretation
            if r2 >= 0.5:
                interpretation = "🟢 VERY STRONG - Highly predictable"
                strong_assignments += 1
            elif r2 >= 0.3:
                interpretation = "🟡 STRONG - Well explained"
                strong_assignments += 1
            elif r2 >= 0.1:
                interpretation = "🟠 MODERATE - Partially explained"
            elif r2 >= 0.05:
                interpretation = "🔴 WEAK - Limited explanation"
            else:
                interpretation = "⚫ NONE - No significant relationship"

            print(f"   Interpretation: {interpretation}")

            # Beta interpretation
            if abs(beta) > 0.001:
                direction = "increases" if beta > 0 else "decreases"
                print(f"   Impact: 1% change in {factor} → PC{j} {direction} by {abs(beta):.4f}")

        total_explained_r2 += r2

    if detailed:
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Strong Assignments (R² ≥ 0.3): {strong_assignments}/{k}")
        print(f"   Average R²: {total_explained_r2 / k:.3f}")
        print(
            f"   Model Quality: {'Excellent' if strong_assignments / k > 0.6 else 'Good' if strong_assignments / k > 0.3 else 'Needs Improvement'}")


def predict_pc_movements(assignments, betas, alphas, predicted_changes, pc_std, k):
    """Predict PC movements with detailed output"""
    pc_movements = np.zeros(k)

    print(f"\n=== PC MOVEMENT PREDICTIONS ===")

    predicted_count = 0
    for j in range(k):
        factor = assignments[j]

        if factor in predicted_changes and factor != 'Residual':
            delta_factor = predicted_changes[factor]
            beta_coef = betas[j]
            alpha_coef = alphas[j]

            raw_pc_change = alpha_coef + beta_coef * delta_factor
            scaled_movement = raw_pc_change * pc_std[j] if pc_std[j] > 1e-8 else raw_pc_change

            pc_movements[j] = scaled_movement
            predicted_count += 1

            print(f"PC{j} ({factor}): {delta_factor:.4f} → {scaled_movement:.6f}")
        else:
            print(f"PC{j} ({factor}): No prediction available")

    print(f"Predicted {predicted_count}/{k} PCs ({predicted_count / k * 100:.1f}%)")
    return pc_movements, predicted_count / k

def compute_expected_returns(Q, pc_movements_percent, centrality, stocks):
    """Compute expected returns with centrality weighting"""
    raw_returns = Q @ pc_movements_percent
    c_vector = np.array([centrality[stock] for stock in stocks])
    adjusted_returns = raw_returns * c_vector
    return adjusted_returns


def optimize_portfolio(adjusted_r, betas, stocks, capital=10000.0, max_position_pct=0.15):
    """Portfolio optimization with market neutrality"""
    prob = LpProblem("Network_Weighted_Portfolio", LpMaximize)

    v_pos = {stock: LpVariable(f"v_pos_{stock}", lowBound=0, upBound=capital * max_position_pct)
             for stock in stocks}
    v_neg = {stock: LpVariable(f"v_neg_{stock}", lowBound=0, upBound=capital * max_position_pct)
             for stock in stocks}

    # Objective: maximize expected return
    prob += lpSum([adjusted_r[i] * (v_pos[stocks[i]] - v_neg[stocks[i]]) for i in range(len(stocks))])

    # Market neutral constraint
    prob += lpSum([betas[i] * (v_pos[stocks[i]] - v_neg[stocks[i]]) for i in range(len(stocks))]) == 0

    # Capital constraint
    prob += lpSum([v_pos[stock] + v_neg[stock] for stock in stocks]) == capital

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    positions = {}
    for stock in stocks:
        pos_val = pulp_value(v_pos[stock]) if v_pos[stock].value() else 0.0
        neg_val = pulp_value(v_neg[stock]) if v_neg[stock].value() else 0.0
        net_position = pos_val - neg_val
        positions[stock] = net_position if abs(net_position) > 0.01 else 0.0

    return positions


def analyze_portfolio_metrics(returns, positions, betas, adjusted_r, valid_stocks, capital,
                              show_top_positions=10):
    """
    Comprehensive portfolio analysis with configurable output

    Parameters:
    - show_top_positions: Number of top positions to display
    """
    print("\n" + "=" * 70)
    print("PORTFOLIO ANALYSIS & RISK METRICS")
    print("=" * 70)

    # Position analysis
    long_positions = {k: v for k, v in positions.items() if v > 0}
    short_positions = {k: v for k, v in positions.items() if v < 0}

    total_long = sum(long_positions.values())
    total_short = abs(sum(short_positions.values()))
    net_exposure = total_long - total_short
    gross_exposure = total_long + total_short

    print(f"\n💰 POSITION SUMMARY:")
    print(f"   Total Long Positions: ${total_long:,.2f}")
    print(f"   Total Short Positions: ${total_short:,.2f}")
    print(f"   Net Exposure: ${net_exposure:,.2f}")
    print(f"   Gross Exposure: ${gross_exposure:,.2f}")
    print(f"   Capital Utilization: {gross_exposure / capital * 100:.1f}%")

    # Risk analysis
    v_vector = np.array([positions.get(stock, 0) for stock in valid_stocks])
    portfolio_beta = np.dot(betas, v_vector) / capital if capital > 0 else 0
    expected_return_dollar = np.dot(adjusted_r, v_vector)
    expected_return_pct = (expected_return_dollar / capital) * 100 if capital > 0 else 0

    print(f"\n⚖️ RISK METRICS:")
    print(f"   Portfolio Beta: {portfolio_beta:.4f}")
    print(f"   Expected Return: ${expected_return_dollar:.2f}")
    print(f"   Expected Return %: {expected_return_pct:.2f}%")
    print(f"   Market Neutrality: {'✅ Good' if abs(portfolio_beta) < 0.1 else '⚠️ High Beta'}")

    # Top positions
    if long_positions:
        print(f"\n📈 TOP {min(show_top_positions, len(long_positions))} LONG POSITIONS:")
        sorted_long = sorted(long_positions.items(), key=lambda x: x[1], reverse=True)
        for i, (stock, pos) in enumerate(sorted_long[:show_top_positions], 1):
            pct_of_capital = pos / capital * 100
            print(f"   {i:2d}. {stock}: ${pos:,.2f} ({pct_of_capital:.1f}%)")

    if short_positions:
        print(f"\n📉 TOP {min(show_top_positions, len(short_positions))} SHORT POSITIONS:")
        sorted_short = sorted(short_positions.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (stock, pos) in enumerate(sorted_short[:show_top_positions], 1):
            pct_of_capital = abs(pos) / capital * 100
            print(f"   {i:2d}. {stock}: ${pos:,.2f} ({pct_of_capital:.1f}%)")

    return {
        'total_long': total_long,
        'total_short': total_short,
        'net_exposure': net_exposure,
        'gross_exposure': gross_exposure,
        'portfolio_beta': portfolio_beta,
        'expected_return': expected_return_dollar,
        'expected_return_pct': expected_return_pct,
        'positions': positions,
        'num_long': len(long_positions),
        'num_short': len(short_positions)
    }


def plot_visualizations(stocks, centrality_vector, positions, pc_time_series, returns,
                        factor_data, assignments, k=8, plot_top_pcs=6,
                        figsize_large=(15, 10), figsize_medium=(15, 8), save_dpi=300):
    """
    Create comprehensive visualizations with configurable parameters

    Parameters:
    - plot_top_pcs: Number of PCs to plot in time series
    - figsize_large, figsize_medium: Figure sizes
    - save_dpi: Save resolution
    """

    # 1. Network Centrality Plot
    plt.figure(figsize=figsize_medium)
    colors = plt.cm.viridis(centrality_vector / np.max(centrality_vector))
    bars = plt.bar(stocks, centrality_vector, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    plt.title('Stock Network Centrality Scores\n(Higher = More Systemic Influence)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('Centrality Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top bars
    for bar, value in zip(bars, centrality_vector):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('centrality_plot.png', dpi=save_dpi, bbox_inches='tight')
    plt.close()

    # 2. Portfolio Positions
    pos_values = [positions.get(stock, 0) for stock in stocks]
    colors = ['darkgreen' if x > 0 else 'darkred' if x < 0 else 'lightgray' for x in pos_values]

    plt.figure(figsize=figsize_medium)
    bars = plt.bar(stocks, pos_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.title('Portfolio Positions\n(Green: Long, Red: Short, Gray: No Position)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('Position Size ($)', fontsize=12)
    plt.axhline(0, color='black', linewidth=1, alpha=0.8)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on significant positions
    for bar, value in zip(bars, pos_values):
        if abs(value) > 100:  # Only label significant positions
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height + (50 if value > 0 else -50),
                     f'${value:.0f}', ha='center',
                     va='bottom' if value > 0 else 'top', fontsize=9)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('positions_plot.png', dpi=save_dpi, bbox_inches='tight')
    plt.close()

    # 3. PC Time Series (configurable number of plots)
    n_plots = min(plot_top_pcs, k)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    plt.figure(figsize=figsize_large)
    for j in range(n_plots):
        plt.subplot(n_rows, n_cols, j + 1)
        plt.plot(pc_time_series[:, j], alpha=0.8, linewidth=1.5, color=f'C{j}')
        plt.title(f'PC{j}: {assignments.get(j, "Unknown")}', fontsize=11, fontweight='bold')
        plt.xlabel('Time Index', fontsize=10)
        plt.ylabel('PC Value', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(pc_time_series[:, j])
        std_val = np.std(pc_time_series[:, j])
        plt.axhline(mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(0.02, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=8)

    plt.suptitle('Principal Component Time Series Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pc_time_series.png', dpi=save_dpi, bbox_inches='tight')
    plt.close()

    # 4. Rolling R² Analysis (Fixed implementation)
    plt.figure(figsize=figsize_large)
    sector_returns = returns.mean(axis=1)

    window = 30
    plot_pcs = min(5, k)  # Limit to 5 PCs for clarity

    for j in range(plot_pcs):
        rolling_r2_values = []
        dates_r2 = []

        for i in range(window, len(sector_returns)):
            try:
                # Get window data
                sector_window = sector_returns.iloc[i - window:i].values
                pc_window = pc_time_series[i - window:i, j]

                # Skip if insufficient variation
                if np.std(sector_window) < 1e-8 or np.std(pc_window) < 1e-8:
                    rolling_r2_values.append(np.nan)
                else:
                    # Calculate R²
                    X_with_const = sm.add_constant(sector_window)
                    model = sm.OLS(pc_window, X_with_const, missing='drop').fit()
                    rolling_r2_values.append(model.rsquared)

                dates_r2.append(sector_returns.index[i])

            except:
                rolling_r2_values.append(np.nan)
                dates_r2.append(sector_returns.index[i])

        # Plot with error handling
        if len(rolling_r2_values) > 0:
            plt.plot(dates_r2, rolling_r2_values, label=f'PC{j} ({assignments.get(j, "Unknown")})',
                     alpha=0.8, linewidth=2)

    plt.title(f'Rolling R² Analysis (Window = {window} days)\nSector Returns vs Principal Components',
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('R² Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rolling_r2_plot.png', dpi=save_dpi, bbox_inches='tight')
    plt.close()


def comprehensive_backtest(prices, stocks, level_factors, price_factors, predicted_changes,
                           lookback_period=252, rebalance_freq='W-FRI', k=8,
                           capital=10000.0, total_years=2, min_r2_threshold=0.15):
    """Streamlined backtest with key metrics only"""

    end_date = prices.index[-1]
    start_date = end_date - timedelta(days=365 * total_years)

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    rebalance_dates = rebalance_dates[rebalance_dates.isin(prices.index)]

    if len(rebalance_dates) < 10:
        print("❌ Insufficient rebalance dates!")
        return None

    portfolio_returns = []
    portfolio_values = [capital]

    print(f"Running backtest: {len(rebalance_dates) - 1} periods")

    successful_periods = 0

    for i, current_date in enumerate(rebalance_dates[:-1]):
        try:
            # Get historical data
            lookback_start = current_date - timedelta(days=int(lookback_period * 1.5))
            hist_prices = prices.loc[lookback_start:current_date]

            if len(hist_prices) < lookback_period // 2:
                continue

            # Prepare data
            returns, factor_data, valid_stocks = prepare_returns_and_factors(
                hist_prices, stocks, level_factors, price_factors, lookback_period
            )

            if len(returns) < 100 or len(valid_stocks) < 10:
                continue

            # Run strategy components
            market_returns = factor_data['SPY'] if 'SPY' in factor_data.columns else factor_data.iloc[:, 0]
            betas = compute_betas(returns, market_returns, valid_stocks)
            centrality, _ = compute_network_centrality(returns, valid_stocks)
            Q, pc_time_series, pc_std, _ = perform_pca(returns, k)
            assignments, f_betas, alphas, r2s = identify_factors_optimized(
                pc_time_series, factor_data, k, min_r2_threshold
            )

            pc_movements, _ = predict_pc_movements(
                assignments, f_betas, alphas, predicted_changes, pc_std, k
            )
            adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)
            positions = optimize_portfolio(adjusted_r, betas, valid_stocks, capital)

            # Calculate actual returns for next period
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

            # Calculate portfolio return
            portfolio_return = 0
            for stock, position in positions.items():
                if abs(position) > 0.01:
                    stock_return = period_returns.get(stock, 0)
                    contribution = (position / capital) * stock_return
                    portfolio_return += contribution

            portfolio_returns.append(portfolio_return)
            current_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(current_value)
            successful_periods += 1

        except Exception:
            continue

    if len(portfolio_returns) < 20:
        print("❌ Insufficient valid periods for backtest")
        return None

    # Calculate key metrics
    portfolio_returns = np.array(portfolio_returns)
    total_return = (portfolio_values[-1] / capital - 1) * 100

    periods_per_year = 52  # Weekly rebalancing
    annualized_return = ((portfolio_values[-1] / capital) ** (periods_per_year / len(portfolio_returns)) - 1) * 100
    volatility = np.std(portfolio_returns) * np.sqrt(periods_per_year) * 100
    sharpe_ratio = (np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(periods_per_year)) if np.std(
        portfolio_returns) > 0 else 0

    # Drawdown calculation
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100

    winning_periods = np.sum(portfolio_returns > 0)
    win_rate = winning_periods / len(portfolio_returns) * 100

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_periods': len(portfolio_returns),
        'final_value': portfolio_values[-1],
        'portfolio_returns': portfolio_returns
    }


def run_strategy(prices, stocks, level_factors, price_factors, predicted_changes,
                 lookback_period=252, k=8, capital=10000.0, min_r2_threshold=0.15):
    """Main strategy execution with focused output"""

    print("=" * 60)
    print("🚀 NETWORK-WEIGHTED MANIFOLD STRATEGY")
    print("=" * 60)

    # Data preparation
    try:
        returns, factor_data, valid_stocks = prepare_returns_and_factors(
            prices, stocks, level_factors, price_factors, lookback_period
        )
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return None

    # Strategy components
    market_returns = factor_data['SPY'] if 'SPY' in factor_data.columns else factor_data.iloc[:, 0]
    betas = compute_betas(returns, market_returns, valid_stocks)
    centrality, centrality_vector = compute_network_centrality(returns, valid_stocks)
    Q, pc_time_series, pc_std, explained_var = perform_pca(returns, k)

    # Factor identification with detailed output
    assignments, f_betas, alphas, r2s = identify_factors_optimized(
        pc_time_series, factor_data, k, min_r2_threshold
    )

    # PC predictions
    pc_movements, prediction_coverage = predict_pc_movements(
        assignments, f_betas, alphas, predicted_changes, pc_std, k
    )

    # Portfolio construction
    adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)
    positions = optimize_portfolio(adjusted_r, betas, valid_stocks, capital)

    if not positions or all(abs(v) < 0.01 for v in positions.values()):
        print("⚠️ Portfolio optimization produced no meaningful positions!")
        return None

    # Portfolio summary
    long_positions = {k: v for k, v in positions.items() if v > 0}
    short_positions = {k: v for k, v in positions.items() if v < 0}

    print(f"\n=== PORTFOLIO SUMMARY ===")
    print(f"Long positions: {len(long_positions)}")
    print(f"Short positions: {len(short_positions)}")
    print(f"Total long: ${sum(long_positions.values()):,.0f}")
    print(f"Total short: ${abs(sum(short_positions.values())):,.0f}")

    # Factor assignment summary
    print(f"\n=== FACTOR ASSIGNMENT QUALITY ===")
    strong_assignments = sum(1 for r2 in r2s.values() if r2 >= 0.25)
    moderate_assignments = sum(1 for r2 in r2s.values() if 0.15 <= r2 < 0.25)
    weak_assignments = sum(1 for r2 in r2s.values() if 0.05 <= r2 < 0.15)
    no_assignments = sum(1 for r2 in r2s.values() if r2 < 0.05)

    print(f"Strong (R²≥0.25): {strong_assignments}/{k}")
    print(f"Moderate (0.15≤R²<0.25): {moderate_assignments}/{k}")
    print(f"Weak (0.05≤R²<0.15): {weak_assignments}/{k}")
    print(f"None (R²<0.05): {no_assignments}/{k}")
    print(f"Prediction coverage: {prediction_coverage:.1%}")

    return {
        'positions': positions,
        'assignments': assignments,
        'r2s': r2s,
        'prediction_coverage': prediction_coverage
    }


# MAIN EXECUTION SCRIPT
if __name__ == "__main__":
    # Focused stock selection - energy sector for better correlations
    stocks = [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
        'WMB', 'HAL', 'BKR', 'FANG', 'DVN', 'HES', 'APA', 'CTRA', 'EQT', 'OVV'
    ]

    # Key factors with better predictive power
    factors_tickers = [
        'SPY', '^GSPC', '^VIX', '^TNX', '^FVX',
        'CL=F', 'BZ=F', 'NG=F', 'GC=F', 'DX=F',
        'XLE', 'XLF', 'XLI', 'XLU', 'XLK',
        'USO', 'UNG', 'XOP', 'OIH', 'VDE',
        'TLT', 'HYG', 'IWM', 'QQQ', 'IVW', 'IVE'
    ]

    level_factors = ['^VIX', '^TNX', '^FVX']
    price_factors = [f for f in factors_tickers if f not in level_factors]

    # Enhanced predictions
    predicted_changes = {
        'SPY': 0.015, '^VIX': 2.0, '^TNX': 0.20,
        'CL=F': 0.05, 'NG=F': 0.08, 'XLE': 0.035,
        'USO': 0.045, 'XOP': 0.04, 'OIH': 0.038,
        'IVW': 0.008, 'IVE': 0.025, 'IWM': 0.018,
        'XLF': 0.022, 'TLT': -0.015, 'HYG': 0.005
    }

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    print("Fetching market data...")
    prices = fetch_data(stocks, factors_tickers, start_date, end_date)

    if prices.empty:
        print("❌ No price data retrieved!")
        exit(1)

    # Run forward-looking strategy
    strategy_config = {
        'lookback_period': 252,
        'k': 8,
        'capital': 10000.0,
        'min_r2_threshold': 0.15  # Higher threshold for better factor matching
    }

    print(f"Running strategy with R² threshold: {strategy_config['min_r2_threshold']}")

    strategy_results = run_strategy(
        prices, stocks, level_factors, price_factors, predicted_changes, **strategy_config
    )

    if strategy_results is None:
        print("❌ Strategy execution failed!")
        exit(1)
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta
    import statsmodels.api as sm
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as pulp_value
    import matplotlib.pyplot as plt
    import quandl
    import warnings

    warnings.filterwarnings('ignore')


    def fetch_data(stocks, factors_tickers, start_date, end_date, max_retries=3):
        """Fetch price data for stocks and factors with retry logic"""
        prices = pd.DataFrame()
        for ticker in stocks + factors_tickers:
            for attempt in range(max_retries):
                try:
                    if ticker == 'FRED/FEDFUNDS':
                        quandl.ApiConfig.api_key = 'YOUR_QUANDL_API_KEY'
                        df = quandl.get(ticker, start_date=start_date, end_date=end_date)
                        prices[ticker] = df['Value']
                    else:
                        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                        if 'Close' in df.columns and not df['Close'].empty:
                            prices[ticker] = df['Close']
                            break
                except:
                    pass
        return prices


    def prepare_returns_and_factors(prices, stocks, level_factors, price_factors, lookback_period=252,
                                    min_observations=100):
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
                factor_data[factor] = prices[factor].pct_change().reindex(
                    returns.index)  # Use pct_change for consistency

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
        """Calculate network centrality with specified threshold"""
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


    def identify_factors_optimized(pc_time_series, factor_data, k, min_r2_threshold=0.05):
        """Assign each PC to the factor with highest R², avoiding duplicates"""
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
                    X_with_const = sm.add_constant(x_clean)
                    model = sm.OLS(y_clean, X_with_const, missing='drop').fit()
                    r2 = model.rsquared
                    if r2 > best_r2:
                        best_r2 = r2
                        best_factor = factor_name
                        best_beta = float(model.params[1])
                        best_alpha = float(model.params[0])
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
        """Predict PC movements"""
        pc_movements = np.zeros(k)
        for j in range(k):
            factor = assignments[j]
            if factor in predicted_changes and factor != 'Residual':
                delta_factor = predicted_changes[factor]
                raw_pc_change = alphas[j] + betas[j] * delta_factor
                pc_movements[j] = raw_pc_change * pc_std[j] if pc_std[j] > 1e-8 else raw_pc_change
        return pc_movements


    def compute_expected_returns(Q, pc_movements_percent, centrality, stocks):
        """Compute expected returns with centrality weighting"""
        raw_returns = Q @ pc_movements_percent
        c_vector = np.array([centrality[stock] for stock in stocks])
        adjusted_returns = raw_returns * c_vector
        return adjusted_returns


    def optimize_portfolio(adjusted_r, betas, stocks, capital=10000.0, max_position_pct=0.15):
        """Portfolio optimization with market neutrality"""
        prob = LpProblem("Network_Weighted_Portfolio", LpMaximize)
        v_pos = {stock: LpVariable(f"v_pos_{stock}", lowBound=0, upBound=capital * max_position_pct) for stock in
                 stocks}
        v_neg = {stock: LpVariable(f"v_neg_{stock}", lowBound=0, upBound=capital * max_position_pct) for stock in
                 stocks}

        prob += lpSum([adjusted_r[i] * (v_pos[stocks[i]] - v_neg[stocks[i]]) for i in range(len(stocks))])
        prob += lpSum([betas[i] * (v_pos[stocks[i]] - v_neg[stocks[i]]) for i in range(len(stocks))]) == 0
        prob += lpSum([v_pos[stock] + v_neg[stock] for stock in stocks]) == capital
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

        positions = {}
        for stock in stocks:
            pos_val = pulp_value(v_pos[stock]) if v_pos[stock].value() else 0.0
            neg_val = pulp_value(v_neg[stock]) if v_neg[stock].value() else 0.0
            net_position = pos_val - neg_val
            positions[stock] = net_position if abs(net_position) > 0.01 else 0.0
        return positions


    def plot_buys_shorts_over_time(portfolio_returns, long_returns, short_returns, rebalance_dates, save_dpi=300):
        """Plot cumulative returns of long and short positions over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(long_returns)) - 1, label='Long Returns', color='green')
        plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(short_returns)) - 1, label='Short Returns', color='red')
        plt.plot(rebalance_dates[:-1], np.cumprod(1 + np.array(portfolio_returns)) - 1, label='Portfolio Returns',
                 color='blue')
        plt.title('Cumulative Returns: Long, Short, and Portfolio', fontsize=12)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cumulative Return', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('buys_shorts_over_time.png', dpi=save_dpi, bbox_inches='tight')
        plt.close()


    def plot_centrality_matrix(returns, stocks, corr_threshold=0.6, save_dpi=300):
        """Plot correlation-based adjacency matrix"""
        corr_matrix = returns.corr().fillna(0).values
        adj_matrix = np.abs(corr_matrix) >= corr_threshold
        adj_matrix = adj_matrix.astype(float)
        plt.figure(figsize=(10, 8))
        plt.imshow(adj_matrix, cmap='binary', interpolation='none')
        plt.colorbar(label='Connection (1 = |corr| ≥ 0.6)')
        plt.xticks(np.arange(len(stocks)), stocks, rotation=45, ha='right')
        plt.yticks(np.arange(len(stocks)), stocks)
        plt.title('Network Centrality Matrix', fontsize=12)
        plt.tight_layout()
        plt.savefig('centrality_matrix.png', dpi=save_dpi, bbox_inches='tight')
        plt.close()


    def plot_pcas_over_time(pc_time_series, assignments, k, save_dpi=300):
        """Plot PC time series"""
        plt.figure(figsize=(15, 8))
        n_cols = 3
        n_rows = (k + n_cols - 1) // n_cols
        for j in range(k):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.plot(pc_time_series[:, j], color=f'C{j}', linewidth=1.5)
            plt.title(f'PC{j}: {assignments.get(j, "Unknown")}', fontsize=10)
            plt.xlabel('Time Index', fontsize=8)
            plt.ylabel('PC Value', fontsize=8)
            plt.grid(True, alpha=0.3)
        plt.suptitle('Principal Component Time Series', fontsize=12)
        plt.tight_layout()
        plt.savefig('pc_time_series.png', dpi=save_dpi, bbox_inches='tight')
        plt.close()


    def comprehensive_backtest(prices, stocks, level_factors, price_factors, predicted_changes,
                               lookback_period=252, rebalance_freq='W-FRI', k=8, capital=10000.0,
                               total_years=2, min_r2_threshold=0.05):
        """Streamlined backtest with profitability metrics for buys, sells, and shorts"""
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

        for i, current_date in enumerate(rebalance_dates[:-1]):
            try:
                lookback_start = current_date - timedelta(days=int(lookback_period * 1.5))
                hist_prices = prices.loc[lookback_start:current_date]
                if len(hist_prices) < lookback_period // 2:
                    continue

                returns, factor_data, valid_stocks = prepare_returns_and_factors(
                    hist_prices, stocks, level_factors, price_factors, lookback_period
                )
                if len(returns) < 100 or len(valid_stocks) < 10:
                    continue

                market_returns = factor_data['SPY'] if 'SPY' in factor_data.columns else factor_data.iloc[:, 0]
                betas = compute_betas(returns, market_returns, valid_stocks)
                centrality, centrality_vector = compute_network_centrality(returns, valid_stocks, corr_threshold=0.6)
                Q, pc_time_series, pc_std, _ = perform_pca(returns, k)
                assignments, f_betas, alphas, r2s = identify_factors_optimized(
                    pc_time_series, factor_data, k, min_r2_threshold
                )
                pc_movements = predict_pc_movements(assignments, f_betas, alphas, predicted_changes, pc_std, k)
                adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)
                positions = optimize_portfolio(adjusted_r, betas, valid_stocks, capital)

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
                long_capital = sum(v for v in positions.values() if v > 0)
                short_capital = abs(sum(v for v in positions.values() if v < 0))

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

                if i == len(rebalance_dates) - 2:  # Save results from last period for output
                    final_assignments = assignments
                    final_r2s = r2s
                    final_alphas = alphas
                    final_betas = f_betas
                    plot_buys_shorts_over_time(portfolio_returns, long_returns, short_returns, rebalance_dates)
                    plot_centrality_matrix(returns, valid_stocks, corr_threshold=0.6)
                    plot_pcas_over_time(pc_time_series, assignments, k)

            except:
                continue

        if len(portfolio_returns) < 20:
            return None

        total_trading_days = (rebalance_dates[-1] - rebalance_dates[0]).days
        portfolio_returns = np.array(portfolio_returns)
        total_return = (portfolio_values[-1] / capital - 1) * 100
        long_return_pct = (np.prod(1 + np.array(long_returns)) - 1) * 100
        short_return_pct = (np.prod(1 + np.array(short_returns)) - 1) * 100
        annualized_return = ((portfolio_values[-1] / capital) ** (252 / len(portfolio_returns)) - 1) * 100
        volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
        sharpe_ratio = (np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)) if np.std(
            portfolio_returns) > 0 else 0
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100

        return {
            'total_return': total_return,
            'long_return_pct': long_return_pct,
            'short_return_pct': short_return_pct,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trading_days': total_trading_days,
            'final_value': portfolio_values[-1],
            'assignments': final_assignments,
            'r2s': final_r2s,
            'alphas': final_alphas,
            'betas': final_betas
        }


    def run_strategy(prices, stocks, level_factors, price_factors, predicted_changes,
                     lookback_period=252, k=8, capital=10000.0, min_r2_threshold=0.05):
        """Main strategy execution with concise output"""
        try:
            returns, factor_data, valid_stocks = prepare_returns_and_factors(
                prices, stocks, level_factors, price_factors, lookback_period
            )
        except Exception as e:
            print(f"Data preparation failed: {e}")
            return None

        market_returns = factor_data['SPY'] if 'SPY' in factor_data.columns else factor_data.iloc[:, 0]
        betas = compute_betas(returns, market_returns, valid_stocks)
        centrality, centrality_vector = compute_network_centrality(returns, valid_stocks, corr_threshold=0.6)
        Q, pc_time_series, pc_std, _ = perform_pca(returns, k)
        assignments, f_betas, alphas, r2s = identify_factors_optimized(
            pc_time_series, factor_data, k, min_r2_threshold
        )
        pc_movements = predict_pc_movements(assignments, f_betas, alphas, predicted_changes, pc_std, k)
        adjusted_r = compute_expected_returns(Q, pc_movements, centrality, valid_stocks)
        positions = optimize_portfolio(adjusted_r, betas, valid_stocks, capital)

        backtest_results = comprehensive_backtest(
            prices, stocks, level_factors, price_factors, predicted_changes,
            lookback_period=lookback_period, rebalance_freq='W-FRI', k=k,
            capital=capital, total_years=2, min_r2_threshold=min_r2_threshold
        )

        if backtest_results:
            print("\n=== PROFITABILITY REPORT ===")
            print(
                f"Total Profit: {backtest_results['total_return']:.2f}% over {backtest_results['total_trading_days']} trading days")
            print(f"Long Positions Profit: {backtest_results['long_return_pct']:.2f}%")
            print(f"Short Positions Profit: {backtest_results['short_return_pct']:.2f}%")
            print(f"Annualized Return: {backtest_results['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
            print(f"Win Rate: {backtest_results['win_rate']:.1f}%")
            print(f"Volatility: {backtest_results['volatility']:.2f}%")
            print(f"Final Value: ${backtest_results['final_value']:,.2f}")

            print("\n=== PC-FACTOR CORRELATIONS ===")
            for j in range(k):
                factor = backtest_results['assignments'].get(j, 'Residual')
                r2 = backtest_results['r2s'].get(j, 0.0)
                alpha = backtest_results['alphas'].get(j, 0.0)
                beta = backtest_results['betas'].get(j, 0.0)
                print(f"PC{j}: {factor}, R² = {r2:.4f}, Equation: PC{j} = {alpha:.4f} + {beta:.4f} × {factor}")

        return backtest_results


    # MAIN EXECUTION SCRIPT
    if __name__ == "__main__":
        # Expanded stock universe: 20 energy + 10 diverse
        stocks = [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
            'WMB', 'HAL', 'BKR', 'FANG', 'DVN', 'HES', 'APA', 'CTRA', 'EQT', 'OVV',
            'AAPL', 'MSFT', 'GOOGL', 'JPM', 'GS', 'BAC', 'WFC', 'SO', 'DUK', 'NEE'
        ]

        # Expanded factors
        factors_tickers = [
            'SPY', '^GSPC', '^VIX', '^TNX', '^FVX', 'CL=F', 'BZ=F', 'NG=F', 'GC=F', 'DX=F',
            'XLE', 'XLF', 'XLI', 'XLU', 'XLK', 'USO', 'UNG', 'XOP', 'OIH', 'VDE',
            'TLT', 'HYG', 'IWM', 'QQQ', 'IVW', 'IVE', 'FRED/FEDFUNDS', 'VX=F', 'VXX'
        ]
        factors_tickers.append('IVW-IVE')  # Growth-value spread

        level_factors = ['^VIX', '^TNX', '^FVX', 'FRED/FEDFUNDS', 'VXX']
        price_factors = [f for f in factors_tickers if f not in level_factors]

        # Static predicted changes (replace with dynamic model if available)
        predicted_changes = {
            'SPY': 0.015, '^VIX': 2.0, '^TNX': 0.20, 'CL=F': 0.05, 'NG=F': 0.08,
            'XLE': 0.035, 'USO': 0.045, 'XOP': 0.04, 'OIH': 0.038, 'IVW': 0.008,
            'IVE': 0.025, 'IWM': 0.018, 'XLF': 0.022, 'TLT': -0.015, 'HYG': 0.005,
            'FRED/FEDFUNDS': 0.002, 'VX=F': 0.03, 'VXX': 0.025, 'IVW-IVE': 0.01
        }

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        prices = fetch_data(stocks, factors_tickers, start_date, end_date)

        # Compute growth-value spread
        if 'IVW' in prices.columns and 'IVE' in prices.columns:
            prices['IVW-IVE'] = prices['IVW'].pct_change() - prices['IVE'].pct_change()

        if prices.empty:
            print("No price data retrieved!")
            exit(1)

        strategy_config = {
            'lookback_period': 252,
            'k': 8,
            'capital': 10000.0,
            'min_r2_threshold': 0.05
        }

        run_strategy(prices, stocks, level_factors, price_factors, predicted_changes, **strategy_config)
    # Run backtest
    backtest_config = {
        'lookback_period': 252,
        'rebalance_freq': 'W-FRI',
        'k': 8,
        'capital': 10000.0,
        'total_years': 2,
        'min_r2_threshold': 0.15
    }

    print(f"\n{'=' * 60}")
    print("📈 RUNNING BACKTEST")
    print(f"{'=' * 60}")

    backtest_results = comprehensive_backtest(
        prices, stocks, level_factors, price_factors, predicted_changes, **backtest_config
    )

    # Final results
    if backtest_results:
        print(f"\n{'=' * 60}")
        print("🎯 PROFITABILITY REPORT")
        print(f"{'=' * 60}")
        print(f"Total Return: {backtest_results['total_return']:.2f}%")
        print(f"Annualized Return: {backtest_results['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"Volatility: {backtest_results['volatility']:.2f}%")
        print(f"Final Value: ${backtest_results['final_value']:,.2f}")
        print(f"Total Periods: {backtest_results['total_periods']}")

        # Strategy grade
        if backtest_results['sharpe_ratio'] > 1.5:
            grade = "A+"
        elif backtest_results['sharpe_ratio'] > 1.0:
            grade = "A"
        elif backtest_results['sharpe_ratio'] > 0.5:
            grade = "B"
        elif backtest_results['sharpe_ratio'] > 0:
            grade = "C"
        else:
            grade = "D"

        print(f"Strategy Grade: {grade}")
    else:
        print("❌ Backtest failed!")

    print(f"\n{'=' * 60}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'=' * 60}")