import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.optimize as opt
from scipy import linalg
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import lsq_linear
import statsmodels.api as sm
# import logging

warnings.filterwarnings('ignore')


class UltimateDetailedStrategy:
    def __init__(self):
        # 1. Universe & Capital
        self.stocks = None  # S = {s_1, s_2, ..., s_N} from sector or watchlist
        self.C_0 = 10000  # Starting capital: C_0 = 10,000 USD
        self.rebalancing_period = 5  # Weekly (5 trading days)
        self.L = 252  # Lookback period for statistics: L = 252 trading days
        self.transaction_costs = 0.001  # fee per share or percentage per trade

        # Data storage - now holds full dataset
        self.P_full = None  # Full price matrix for entire period
        self.factors_data_full = None  # Full factor data for entire period

        # Current window data (recalculated each week)
        self.P = None  # Price matrix for current window
        self.R = None  # Returns matrix R of shape (T, N)
        self.Sigma = None  # Covariance matrix Σ = (1 / (T - 1)) * R^T · R
        self.V = None  # Loadings matrix V = [v_1, v_2, ..., v_K] shape (N, K)
        self.PC_day = None  # PC time series matrix PC_day = R · V shape (T, K)
        self.factors_data = None  # Factor data for current window
        self.centrality_matrix = None  # Centrality matrix C
        self.u_centrality = None  # First eigenvector of centrality matrix

        # Results storage
        self.portfolio_history = []
        self.performance_metrics = {}

        # Optimization counters
        self.slsqp_success_count = 0
        self.lsq_fallback_count = 0

    def load_full_data(self, stock_symbols, start_date, end_date, factors_dict=None):
        """
        Load full dataset from 252 days before start_date until end_date
        """
        print("Loading full stock and factor data...")
        self.stocks = stock_symbols

        # Calculate actual start date (252 trading days before requested start)
        start_date_dt = pd.to_datetime(start_date)
        # Approximate 252 trading days as 360 calendar days to be safe
        extended_start_date = start_date_dt - pd.Timedelta(days=360)

        # Load stock price data
        stock_data = yf.download(stock_symbols, start=extended_start_date, end=end_date, auto_adjust=True)
        prices = stock_data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(stock_symbols[0])
        self.P_full = prices.dropna()
        print(f"Loaded full price data shape: {self.P_full.shape}")

        # Load factor data
        if factors_dict:
            factor_symbols = list(factors_dict.keys())
            print(f"Loading full factor data for {len(factor_symbols)} symbols...")
            factor_data = yf.download(factor_symbols, start=extended_start_date, end=end_date, auto_adjust=True)
            factors_prices = factor_data['Close']
            if isinstance(factors_prices, pd.Series):
                factors_prices = factors_prices.to_frame(factor_symbols[0])

            # Drop columns with all NaN values and check if data is empty
            factors_prices = factors_prices.dropna(axis=1, how='all')
            if factors_prices.empty:
                print("Warning: No valid factor data loaded. Skipping factor data.")
                self.factors_data_full = None
            else:
                self.factors_data_full = factors_prices.dropna()
                print(f"Loaded full factor data shape: {self.factors_data_full.shape}")
        else:
            self.factors_data_full = None
            print("No factor data provided.")

        # Store the requested backtest start date
        self.backtest_start_date = start_date

    def set_current_window(self, current_date):
        """
        Set the current 252-day lookback window ending at current_date
        """
        current_date = pd.to_datetime(current_date)

        # Get data up to current_date
        available_data = self.P_full[self.P_full.index <= current_date]

        # Take the last L (252) days
        if len(available_data) >= self.L:
            self.P = available_data.tail(self.L)
        else:
            self.P = available_data  # Use all available data if less than L

        # Set corresponding factor data
        if self.factors_data_full is not None:
            available_factor_data = self.factors_data_full[self.factors_data_full.index <= current_date]
            if len(available_factor_data) >= self.L:
                self.factors_data = available_factor_data.tail(self.L)
            else:
                self.factors_data = available_factor_data
        else:
            self.factors_data = None

    def prepare_returns_matrix(self, standardize=False):
        """
        2b. Log Returns Matrix
        Construct returns matrix R of shape (T, N) for current window
        """
        # R[t, s] = log(P_t_s / P_(t-1)_s)
        self.R = np.log(self.P / self.P.shift(1)).dropna()

        # 2c. Optional: Standardization
        if standardize:
            scaler = StandardScaler()
            self.R = pd.DataFrame(scaler.fit_transform(self.R),
                                  index=self.R.index, columns=self.R.columns)

        return self.R

    def compute_covariance_and_pca(self, n_components=None):
        """
        3. Covariance and PCA for current window
        """
        # 3a. Covariance Matrix - Σ = (1 / (T - 1)) * R^T · R, Shape: (N, N)
        T, N = self.R.shape
        self.Sigma = np.cov(self.R.T, ddof=1)  # Shape: (N, N)

        # 3b. Eigen Decomposition - Σ · v_k = λ_k · v_k
        eigenvalues, eigenvectors = linalg.eigh(self.Sigma)
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if n_components is None:
            n_components = min(N, 10)  # Default to 10 components or N if smaller

        # Stack eigenvectors to form loadings matrix V
        # V = [v_1, v_2, ..., v_K] shape (N, K), Each column = PC loadings
        self.V = eigenvectors[:, :n_components]
        self.eigenvalues = eigenvalues[:n_components]

        # 3c. PC Time Series Matrix
        # Project returns onto PCs: PC_day = R · V shape (T, K)
        self.PC_day = self.R.values @ self.V

        return self.V, self.PC_day

    def factor_pc_regression(self, min_r_squared=0.15, max_correlation=0.9, max_vif=5):
        """
        4. Factor-PC Regression for current window
        """
        if self.factors_data is None:
            return None

        # 4a. Candidate Factors - Factors = {F_1, F_2, ..., F_M}
        # Align factor data with returns data
        common_dates = self.R.index.intersection(self.factors_data.index)
        factors_aligned = self.factors_data.loc[common_dates]
        pc_aligned = pd.DataFrame(self.PC_day, index=self.R.index).loc[common_dates]

        # 4b. Delta Factors - Compute weekly factor changes: ΔF_f_t = F_f_t - F_f_(t-1)
        delta_factors = np.log(factors_aligned / factors_aligned.shift(1)).dropna()
        pc_aligned = pc_aligned.loc[delta_factors.index]

        self.pc_regressions = {}

        # 4c. Multi-Factor Regression - For each PC_k:
        for k in range(self.V.shape[1]):
            y = pc_aligned.iloc[:, k].values  # PC_k_t time series
            X = delta_factors.values  # ΔF matrix

            # Filter factors with R² ≥ min_r_squared using statsmodels for p-values
            selected_factors = []
            factor_r2 = []

            for i, factor_name in enumerate(delta_factors.columns):
                X_single = X[:, i].reshape(-1, 1)
                X_single_const = sm.add_constant(X_single)
                reg_single = sm.OLS(y, X_single_const).fit()
                r2_single = reg_single.rsquared
                p_value = reg_single.pvalues[1]  # p-value for the factor coefficient

                if r2_single >= min_r_squared:
                    selected_factors.append(i)
                    factor_r2.append((i, factor_name, r2_single, p_value))

            if not selected_factors:
                continue

            # Store individual R² and p-values
            self.pc_regressions[k] = {
                'individual_r2': {name: {'r2': r2, 'pvalue': pval} for i, name, r2, pval in factor_r2}
            }

            # Collinearity filtering: If two factors have correlation > max_correlation or VIF > max_vif
            X_selected = X[:, selected_factors]

            # Check correlations
            if X_selected.shape[1] > 1:
                corr_matrix = np.corrcoef(X_selected.T)
                high_corr_pairs = np.where((np.abs(corr_matrix) > max_correlation) &
                                           (np.abs(corr_matrix) < 1.0))

                # Remove factors with high correlation (keep the one with higher R²)
                to_remove = set()
                for loc_i, loc_j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                    if loc_i < loc_j:  # Avoid duplicates
                        r2_i = factor_r2[loc_i][2]
                        r2_j = factor_r2[loc_j][2]
                        if r2_i < r2_j:
                            to_remove.add(loc_i)
                        else:
                            to_remove.add(loc_j)

                selected_factors = [f for idx, f in enumerate(selected_factors) if idx not in to_remove]
                X_selected = X[:, selected_factors]

            # Check VIF
            if X_selected.shape[1] > 1:
                try:
                    vif_scores = [variance_inflation_factor(X_selected, i)
                                  for i in range(X_selected.shape[1])]
                    high_vif = [i for i, vif in enumerate(vif_scores) if vif > max_vif]

                    # Remove high VIF factors
                    selected_factors = [selected_factors[i] for i in range(len(selected_factors))
                                        if i not in high_vif]
                    X_selected = X[:, selected_factors]
                except:
                    pass  # Continue if VIF calculation fails

            if X_selected.shape[1] == 0:
                continue

            # Use statsmodels OLS for detailed statistics
            X_selected_const = sm.add_constant(X_selected)
            reg = sm.OLS(y, X_selected_const).fit()
            alpha = reg.params[0]  # Intercept
            beta = reg.params[1:]  # Coefficients
            r_squared = reg.rsquared
            y_hat = reg.fittedvalues
            correlation = np.corrcoef(y, y_hat)[0, 1]

            # Compute final VIF for selected factors
            if X_selected.shape[1] > 1:
                vif_scores = [variance_inflation_factor(X_selected, i) for i in range(X_selected.shape[1])]
            else:
                vif_scores = [1.0]

            # Store results
            factor_names = [delta_factors.columns[i] for i in selected_factors]
            self.pc_regressions[k].update({
                'alpha': alpha,
                'beta': beta,
                'selected_factors': selected_factors,
                'factor_names': factor_names,
                'r_squared': r_squared,
                'correlation': correlation,
                'model': reg,
                'vif': dict(zip(factor_names, vif_scores))
            })

        return self.pc_regressions

    def compute_historic_pc_std(self, lookback_window=None):
        """
        5. Historic PC Standard Deviation for current window
        Over lookback L: σ_PC_k = sqrt( Σ_{i=t-L}^{t-1} (PC_k_i - mean(PC_k))^2 / (L - 1) )
        """
        if lookback_window is None:
            lookback_window = min(self.L, len(self.PC_day))

        # Convert PC_day to DataFrame for easier rolling operations
        PC_df = pd.DataFrame(self.PC_day, index=self.R.index)

        # For current window, we can just compute the std directly since we already have the window
        self.sigma_PC = PC_df.std(ddof=1).values  # σ_PC vector

        return self.sigma_PC

    def calculate_current_factor_changes(self, lookback_days=5):
        """
        Calculate actual factor changes from current window factor data
        Uses the most recent lookback_days to compute ΔF_f_t = log(F_f_t / F_f_(t-lookback_days))
        """
        if self.factors_data is None:
            print("No factor data available")
            return {}

        # Get the most recent data from current window
        recent_data = self.factors_data.tail(lookback_days + 1)

        if len(recent_data) < 2:
            print("Insufficient factor data for change calculation")
            return {}

        # Calculate log returns: ΔF_f_t = log(F_f_t / F_f_(t-lookback_days))
        current_prices = recent_data.iloc[-1]  # Most recent prices
        previous_prices = recent_data.iloc[-(lookback_days + 1)]  # Prices lookback_days ago

        # Calculate log changes
        factor_changes = {}
        for factor in self.factors_data.columns:
            if not np.isnan(current_prices[factor]) and not np.isnan(previous_prices[factor]):
                if previous_prices[factor] > 0:  # Avoid division by zero
                    change = np.log(current_prices[factor] / previous_prices[factor])
                    factor_changes[factor] = change

        return factor_changes

    def calculate_stock_betas(self, benchmark_symbol='XLE', lookback_days=252):
        """
        Calculate stock betas relative to sector benchmark using current window data
        β_i = Cov(R_i, R_benchmark) / Var(R_benchmark)
        """
        if self.factors_data is None or benchmark_symbol not in self.factors_data.columns:
            return np.ones(len(self.stocks))

        # Get benchmark data aligned with stock data from current window
        common_dates = self.R.index.intersection(self.factors_data.index)
        if len(common_dates) < lookback_days // 2:
            return np.ones(len(self.stocks))

        # Calculate benchmark returns
        benchmark_prices = self.factors_data.loc[common_dates, benchmark_symbol]
        benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

        # Align stock returns with benchmark returns
        aligned_stock_returns = self.R.loc[benchmark_returns.index]

        # Calculate betas for each stock
        betas = np.zeros(len(self.stocks))
        benchmark_var = np.var(benchmark_returns, ddof=1)

        if benchmark_var <= 0:
            return np.ones(len(self.stocks))

        for i, stock in enumerate(self.stocks):
            if stock in aligned_stock_returns.columns:
                stock_returns = aligned_stock_returns[stock]
                # β_i = Cov(R_i, R_benchmark) / Var(R_benchmark)
                covariance = np.cov(stock_returns, benchmark_returns, ddof=1)[0, 1]
                betas[i] = covariance / benchmark_var
            else:
                betas[i] = 1.0  # Default beta if stock not found

        return betas

    def predict_pc_movement(self, current_factor_changes):
        """
        6. Predicted PC Movement
        Predict next week's PC movement: ΔPC_pred_k = α_f + Σ (β_f * ΔF_f_current)
        """
        if not hasattr(self, 'pc_regressions') or not self.pc_regressions:
            return np.zeros(self.V.shape[1])

        # Initialize predicted PC movement vector
        self.delta_PC_pred = np.zeros(self.V.shape[1])  # ΔPC_pred = [ΔPC_pred_1, ..., ΔPC_pred_K]

        for k, regression_data in self.pc_regressions.items():
            alpha = regression_data['alpha']  # α_f
            beta = regression_data['beta']  # β_f coefficients
            selected_factors = regression_data['selected_factors']

            # Get current factor changes for selected factors
            current_changes = np.array([current_factor_changes.get(self.factors_data.columns[i], 0)
                                        for i in selected_factors])

            # ΔPC_pred_k = α_f + Σ (β_f * ΔF_f_current)
            self.delta_PC_pred[k] = alpha + np.dot(beta, current_changes)

        # Convert to % movement: PredictedPctChange_PC_k = ΔPC_pred_k * σ_PC_k
        self.predicted_pct_change_PC = self.delta_PC_pred * self.sigma_PC

        return self.delta_PC_pred

    def predict_stock_returns(self):
        """
        7. Predicted Stock Returns
        """
        # 7a. Loadings & Stock Prediction
        pc_weighted = self.delta_PC_pred * self.sigma_PC  # ΔPC_pred ⊙ σ_PC
        self.r_hat = self.V @ self.delta_PC_pred  # Predicted stock returns vector

        return self.r_hat

    def compute_centrality_weighting(self, method='correlation'):
        """
        8. Centrality Weighting - recalculated for current window
        Compute centrality matrix C from stock correlation or network adjacency
        """
        if method == 'correlation':
            # Use correlation matrix as centrality matrix
            self.centrality_matrix = np.corrcoef(self.R.T)  # C from stock correlation
        else:
            # Use covariance matrix
            self.centrality_matrix = self.Sigma

        # Get first eigenvector of C (largest eigenvalue): u_centrality = eigenvector(C_max)
        eigenvalues, eigenvectors = linalg.eigh(self.centrality_matrix)
        max_eigenvalue_idx = np.argmax(eigenvalues)
        self.u_centrality = np.abs(eigenvectors[:, max_eigenvalue_idx])  # Take absolute values

        # Normalize to mean=1 and std=1
        self.u_centrality = self.u_centrality - np.mean(self.u_centrality) + 1  # Shift to mean=1
        self.u_centrality = self.u_centrality / np.std(self.u_centrality, ddof=1)  # Scale to std=1
        self.u_centrality = np.maximum(self.u_centrality, 0)  # Ensure non-negative weights

        # Weight predicted stock returns: r_hat_weighted = r_hat ⊙ u_centrality
        self.r_hat_weighted = self.r_hat * self.u_centrality  # Gives importance-adjusted expected returns

        return self.r_hat_weighted

    def optimize_portfolio(self, current_prices, stock_betas=None):
        N = len(self.r_hat_weighted)
        C_total = self.C_0

        # Validate inputs
        if len(current_prices) != N or np.any(np.isnan(current_prices)) or np.any(np.isinf(current_prices)) or np.any(
                current_prices <= 0):
            return np.zeros(N)
        if np.any(np.isnan(self.r_hat_weighted)) or np.any(np.isinf(self.r_hat_weighted)):
            return np.zeros(N)

        r_hat_scaled = self.r_hat_weighted * 100

        def objective(v):
            return -np.dot(v, r_hat_scaled)

        constraints = []

        # Total allocation constraint: sum(|v_i|) <= C_total
        def total_allocation_constraint(v):
            return C_total - np.sum(np.abs(v))

        constraints.append({'type': 'ineq', 'fun': total_allocation_constraint})

        # Long allocation constraints: 0.55 * C_total <= long_positions <= 0.75 * C_total
        def long_allocation_constraint(v):
            long_positions = np.sum(v[v > 0])
            return long_positions - 0.55 * C_total

        def long_allocation_upper(v):
            return 0.75 * C_total - np.sum(v[v > 0])

        constraints.append({'type': 'ineq', 'fun': long_allocation_constraint})
        constraints.append({'type': 'ineq', 'fun': long_allocation_upper})

        # Short allocation constraints: 0.25 * C_total <= |short_positions| <= 0.45 * C_total
        def short_allocation_constraint(v):
            short_positions = np.abs(np.sum(v[v < 0]))
            return short_positions - 0.25 * C_total

        def short_allocation_upper(v):
            return 0.45 * C_total - np.abs(np.sum(v[v < 0]))

        constraints.append({'type': 'ineq', 'fun': short_allocation_constraint})
        constraints.append({'type': 'ineq', 'fun': short_allocation_upper})

        # Portfolio beta constraint: |portfolio_beta| <= 0.1
        if stock_betas is not None:
            def portfolio_beta_constraint(v):
                portfolio_beta = np.abs(np.dot(v, stock_betas) / C_total)
                return 0.1 - portfolio_beta

            constraints.append({'type': 'ineq', 'fun': portfolio_beta_constraint})

        # Individual position bounds with short-only-negative-returns constraint
        bounds = []
        for i in range(N):
            if r_hat_scaled[i] >= 0:
                # Positive expected returns: can only go long (0 to 15% of capital)
                bound = (0, 0.15 * C_total)
            else:
                # Negative expected returns: can go short (but also allow long positions)
                bound = (-0.12 * C_total, 0.15 * C_total)
            bounds.append(bound)

        # Short concentration constraint: no single short > 25% of total short allocation
        # This is handled via individual bounds above (12% < 25% of 45% short allocation)

        # Initial guess: target allocation based on expected returns
        v0 = np.zeros(N)
        long_stocks = r_hat_scaled >= 0
        short_stocks = r_hat_scaled < 0

        if np.any(long_stocks):
            v0[long_stocks] = (0.65 * C_total) / np.sum(long_stocks)
        if np.any(short_stocks):
            # Only short stocks with negative expected returns
            v0[short_stocks] = -(0.35 * C_total) / np.sum(short_stocks)

        # Adjust for beta constraint if needed
        if stock_betas is not None:
            portfolio_beta = np.dot(v0, stock_betas) / C_total
            if abs(portfolio_beta) > 0.1:
                scaling_factor = 0.1 / abs(portfolio_beta)
                v0 *= scaling_factor

        # First attempt: SLSQP optimization
        try:
            result = opt.minimize(objective, v0, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-8, 'disp': False})
            if result.success and np.sum(np.abs(result.x)) > 1e-6:
                self.optimal_weights = result.x
                self.slsqp_success_count += 1
            else:
                # Fallback to proper least squares approach
                self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, C_total)
                self.lsq_fallback_count += 1
        except Exception:
            # Fallback to proper least squares approach
            self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, C_total)
            self.lsq_fallback_count += 1

        self.share_quantities = self.optimal_weights / current_prices
        return self.optimal_weights

    def _least_squares_fallback(self, r_hat_scaled, bounds, stock_betas, C_total):
        """
        Proper least squares fallback that minimizes ||x - target||² subject to bounds and constraints
        """
        N = len(r_hat_scaled)

        # Create target allocation based on expected returns
        target = np.zeros(N)
        long_stocks = r_hat_scaled >= 0
        short_stocks = r_hat_scaled < 0

        # Allocate based on relative expected returns within each category
        if np.any(long_stocks):
            long_returns = r_hat_scaled[long_stocks]
            # Normalize to sum to target long allocation
            if np.sum(long_returns) > 0:
                long_weights = long_returns / np.sum(long_returns) * (0.65 * C_total)
                target[long_stocks] = long_weights

        if np.any(short_stocks):
            short_returns = np.abs(r_hat_scaled[short_stocks])  # Use absolute values for shorts
            # Normalize to sum to target short allocation
            if np.sum(short_returns) > 0:
                short_weights = -(short_returns / np.sum(short_returns) * (0.35 * C_total))
                target[short_stocks] = short_weights

        # Adjust target for beta constraint if needed
        if stock_betas is not None:
            target_beta = np.dot(target, stock_betas) / C_total
            if abs(target_beta) > 0.1:
                scaling_factor = 0.1 / abs(target_beta)
                target *= scaling_factor

        # Extract bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        # Solve least squares problem: minimize ||x - target||² subject to bounds
        try:
            # Use identity matrix A and target as b for ||x - target||²
            A = np.eye(N)
            res = lsq_linear(A, target, bounds=(lb, ub), method='bvls')

            if res.success and np.sum(np.abs(res.x)) > 1e-6:
                return res.x
            else:
                # Final fallback: return clipped target
                return np.clip(target, lb, ub)

        except Exception:
            # Final fallback: return clipped target
            return np.clip(target, lb, ub)
    def stress_test_transaction_costs(self, cost_levels=[0.001, 0.003, 0.005, 0.01]):
        """
        Test strategy performance under different transaction cost assumptions
        """
        print("\n" + "=" * 70)
        print("TRANSACTION COST STRESS TEST")
        print("=" * 70)

        if not self.portfolio_history:
            print("No portfolio history available")
            return

        results = {}

        print(f"\nOriginal transaction cost: {self.transaction_costs:.1%}")
        print(f"Testing costs: {[f'{c:.1%}' for c in cost_levels]}")

        for cost_level in cost_levels:
            # Recalculate performance with different transaction costs
            total_pnl = 0
            total_costs = 0

            for trade in self.portfolio_history:
                # Get realized P&L from metrics
                ts = trade['timestamp']
                metrics = self.performance_metrics.get(ts, {})
                gross_pnl = metrics.get('realized_pnl', 0)

                # Recalculate transaction cost
                total_trade_value = np.sum(np.abs(trade['weights']))
                new_transaction_cost = total_trade_value * cost_level

                # Net P&L with new costs
                net_pnl = gross_pnl - new_transaction_cost
                total_pnl += net_pnl
                total_costs += new_transaction_cost

            # Calculate performance metrics
            total_return = total_pnl / self.C_0
            num_periods = len(self.portfolio_history)
            annualized_return = (1 + total_return) ** (52 / num_periods) - 1 if num_periods > 0 else 0

            results[cost_level] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'total_costs': total_costs,
                'avg_cost_per_trade': total_costs / num_periods if num_periods > 0 else 0
            }

        # Display results
        print(f"\n{'Cost Level':<12} {'Total Ret':<12} {'Annual Ret':<12} {'Total Costs':<12} {'Avg/Trade':<12}")
        print("-" * 60)
        for cost, metrics in results.items():
            print(f"{cost:<8.1%} {metrics['total_return']:<8.1%} "
                  f"{metrics['annualized_return']:<8.1%} ${metrics['total_costs']:<11,.0f} "
                  f"${metrics['avg_cost_per_trade']:<11,.0f}")

        # Plot impact
        costs = list(results.keys())
        returns = [results[c]['annualized_return'] for c in costs]

        plt.figure(figsize=(10, 6))
        plt.plot([c * 100 for c in costs], [r * 100 for r in returns], marker='o', linewidth=2, markersize=8)
        plt.title('Strategy Performance vs Transaction Costs')
        plt.xlabel('Transaction Cost (%)')
        plt.ylabel('Annualized Return (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return results

    def execute_trades(self, current_prices):
        """
        Execute trades according to the computed v_i
        """
        # Calculate transaction costs
        total_trade_value = np.sum(np.abs(self.optimal_weights))
        transaction_cost = total_trade_value * self.transaction_costs
        expected_returns = self.r_hat_weighted

        # Record trade execution
        trade_record = {
            'timestamp': self.P.index[-1],
            'weights': self.optimal_weights.copy(),
            'shares': self.share_quantities.copy(),
            'prices': current_prices.copy(),
            'transaction_cost': transaction_cost,
            'expected_returns': expected_returns.copy()
        }

        self.portfolio_history.append(trade_record)
        return trade_record

    def record_performance_metrics(self, actual_returns=None):
        """
        Fixed performance metrics calculation for rebalancing strategy
        """
        if not self.portfolio_history:
            return

        latest_trade = self.portfolio_history[-1]

        # Portfolio composition metrics
        long_positions = latest_trade['weights'] > 0
        short_positions = latest_trade['weights'] < 0

        metrics = {
            'portfolio_value': np.sum(np.abs(latest_trade['weights'])),
            'long_exposure': np.sum(latest_trade['weights'][long_positions]),
            'short_exposure': np.abs(np.sum(latest_trade['weights'][short_positions])),
            'num_long_positions': np.sum(long_positions),
            'num_short_positions': np.sum(short_positions),
            'transaction_cost': latest_trade['transaction_cost']
        }

        # Calculate period P&L if actual returns provided
        if actual_returns is not None:
            # For a rebalancing strategy: P&L = sum(weight_i * return_i) for the period
            gross_pnl = np.dot(latest_trade['weights'], actual_returns)
            net_pnl = gross_pnl - latest_trade['transaction_cost']

            metrics['gross_pnl'] = gross_pnl
            metrics['realized_pnl'] = net_pnl  # This is NET P&L after costs
            metrics['expected_pnl'] = np.dot(latest_trade['weights'], latest_trade['expected_returns'])

            # Store actual returns in latest_trade for analysis
            latest_trade['actual_returns'] = actual_returns.copy()

        self.performance_metrics[latest_trade['timestamp']] = metrics
        return metrics

    def compute_profitability_metrics(self, risk_free_rate=0.02, periods_per_year=52):
        """
        Fixed profitability metrics for rebalancing strategy
        """
        if not self.portfolio_history:
            return {}

        # Collect period-by-period P&L (should be net P&L after transaction costs)
        period_pnls = []
        gross_pnls = []
        transaction_costs = []
        portfolio_values = [self.C_0]  # Start with initial capital

        for trade in self.portfolio_history:
            ts = trade['timestamp']
            metrics = self.performance_metrics.get(ts, {})

            net_pnl = metrics.get('realized_pnl', 0)  # Already net of costs
            gross_pnl = metrics.get('gross_pnl', 0)
            cost = trade['transaction_cost']

            period_pnls.append(net_pnl)
            gross_pnls.append(gross_pnl)
            transaction_costs.append(cost)

            # Update portfolio value: previous value + net P&L
            new_portfolio_value = portfolio_values[-1] + net_pnl
            portfolio_values.append(new_portfolio_value)

        if not period_pnls:
            return {}

        # Calculate returns based on portfolio value progression
        initial_capital = self.C_0
        final_portfolio_value = portfolio_values[-1]
        total_return = (final_portfolio_value - initial_capital) / initial_capital

        # Period returns for risk metrics
        period_returns = []
        for i in range(1, len(portfolio_values)):
            period_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            period_returns.append(period_return)

        period_returns = np.array(period_returns)

        # Risk metrics
        mean_return = np.mean(period_returns)
        std_return = np.std(period_returns, ddof=1)

        # Sharpe ratio
        excess_return = mean_return - risk_free_rate / periods_per_year
        sharpe = excess_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0

        # Sortino ratio (downside risk)
        downside_returns = period_returns[period_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else std_return
        sortino = excess_return / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0

        # Maximum drawdown
        portfolio_values_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values_array)
        drawdowns = (running_max - portfolio_values_array) / running_max
        max_drawdown = np.max(drawdowns)

        # Annualized return
        num_periods = len(period_pnls)
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1 if num_periods > 0 else 0

        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf

        # Win/Loss analysis
        winning_periods = [pnl for pnl in period_pnls if pnl > 0]
        losing_periods = [pnl for pnl in period_pnls if pnl < 0]

        avg_win = np.mean(winning_periods) if winning_periods else 0
        avg_loss = np.abs(np.mean(losing_periods)) if losing_periods else 0
        win_rate = len(winning_periods) / len(period_pnls) if period_pnls else 0
        profit_factor = sum(winning_periods) / sum(np.abs(losing_periods)) if losing_periods else np.inf

        # Position-level win rates (this was likely inflated before)
        position_win_rates = self._calculate_position_win_rates()

        # Transaction cost analysis
        total_gross_pnl = sum(gross_pnls)
        total_transaction_costs = sum(transaction_costs)
        cost_drag = total_transaction_costs / abs(total_gross_pnl) if total_gross_pnl != 0 else 0

        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,  # This is period-level win rate
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_periods': num_periods,
            'total_transaction_costs': total_transaction_costs,
            'transaction_cost_drag': cost_drag,
            'final_portfolio_value': final_portfolio_value,
        }

        # Add position-level win rates
        results.update(position_win_rates)

        print(f"\n=== CORRECTED Profitability Metrics ===")
        print(f"Initial Capital: ${self.C_0:,.0f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.0f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Period Win Rate: {win_rate:.1%}")  # This should be much lower
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.1%}")
        print(f"Total Transaction Costs: ${total_transaction_costs:,.0f}")
        print(f"Transaction Cost Drag: {cost_drag:.1%}")

        return results

    def _calculate_position_win_rates(self):
        """
        Calculate win rates at the position level (this was likely inflated before)
        """
        long_wins = []
        short_wins = []
        overall_wins = []

        for trade in self.portfolio_history:
            if 'actual_returns' in trade:
                weights = trade['weights']
                actual_returns = trade['actual_returns']

                # Calculate P&L for each position
                position_pnls = weights * actual_returns

                # Separate long and short positions
                long_positions = weights > 0
                short_positions = weights < 0

                if np.any(long_positions):
                    long_pnl = position_pnls[long_positions]
                    long_win_rate = np.sum(long_pnl > 0) / len(long_pnl)
                    long_wins.append(long_win_rate)

                if np.any(short_positions):
                    short_pnl = position_pnls[short_positions]
                    short_win_rate = np.sum(short_pnl > 0) / len(short_pnl)
                    short_wins.append(short_win_rate)

                # Overall position win rate for this period
                if len(position_pnls) > 0:
                    overall_win_rate = np.sum(position_pnls > 0) / len(position_pnls)
                    overall_wins.append(overall_win_rate)

        return {
            'avg_long_position_win_rate': np.mean(long_wins) if long_wins else 0,
            'avg_short_position_win_rate': np.mean(short_wins) if short_wins else 0,
            'avg_overall_position_win_rate': np.mean(overall_wins) if overall_wins else 0,
        }

    def backtest_strategy(self, start_date, end_date, rebalance_freq=5):
        """
        Fixed backtest with proper portfolio value tracking
        """
        print(f"\n=== Backtesting Strategy from {start_date} to {end_date} ===")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter to backtest period
        backtest_dates = self.P_full[(self.P_full.index >= start_dt) &
                                     (self.P_full.index <= end_dt)].index

        if len(backtest_dates) == 0:
            print("No data available for backtesting period")
            return None

        portfolio_value = self.C_0  # Track actual portfolio value
        portfolio_values = [portfolio_value]
        rebalance_dates = []

        print(f"Starting Portfolio Value: ${portfolio_value:,.0f}")

        # Simulate rebalancing every rebalance_freq days
        for i in range(0, len(backtest_dates), rebalance_freq):
            current_date = backtest_dates[i]
            rebalance_dates.append(current_date)

            # Calculate actual returns for the period (if we have previous trade)
            actual_returns = None
            if len(self.portfolio_history) > 0:
                prev_date_idx = max(0, i - rebalance_freq)
                if prev_date_idx < i:
                    period_data = self.P_full.loc[backtest_dates[prev_date_idx]:current_date]
                    if len(period_data) >= 2:
                        period_returns = (period_data.iloc[-1] / period_data.iloc[0] - 1).values
                        actual_returns = period_returns

            try:
                # Execute rebalancing step with rolling window
                trade_record, metrics = self.weekly_rebalancing_step(current_date, actual_returns)

                # Update portfolio value based on net P&L
                if trade_record is not None and metrics is not None and 'realized_pnl' in metrics:
                    portfolio_value = metrics['realized_pnl']  # Add net P&L
                    portfolio_values.append(portfolio_value)

                    if i < 50:  # Show first few periods
                        print(
                            f"Period {len(portfolio_values) - 1}: ${portfolio_value:,.0f} (P&L: ${metrics['realized_pnl']:+.0f})")
                else:
                    portfolio_values.append(portfolio_value)  # No change

            except Exception as e:
                print(f"Error at {current_date}: {e}")
                portfolio_values.append(portfolio_value)

        # Calculate final performance metrics
        if len(portfolio_values) > 1:
            final_value = portfolio_values[-1]
            total_return = (final_value / self.C_0) - 1
            num_periods = len(portfolio_values) - 1
            annualized_return = (1 + total_return) ** (252 / (num_periods * rebalance_freq)) - 1

            print(f"\nFinal Portfolio Value: ${final_value:,.0f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {annualized_return:.2%}")
        else:
            total_return = 0
            annualized_return = 0
            final_value = self.C_0

        return {
            'portfolio_values': portfolio_values,
            'rebalance_dates': rebalance_dates,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_value': final_value
        }

    def validate_no_lookahead(self):
        """
        Check for data leakage in all trades
        """
        print("\n" + "=" * 60)
        print("DATA LEAKAGE VALIDATION")
        print("=" * 60)

        issues_found = 0
        for i, trade in enumerate(self.portfolio_history):
            trade_date = trade['timestamp']

            # Set the window that was used for this trade
            self.set_current_window(trade_date)

            # Check if window end is after trade date
            window_end = self.P.index[-1]
            if window_end > trade_date:
                print(f"❌ Trade {i + 1} on {trade_date}: Window ends {window_end} (LOOKAHEAD BIAS)")
                issues_found += 1
            else:
                print(f"✅ Trade {i + 1} on {trade_date}: Window ends {window_end} (OK)")

        print(f"\nFound {issues_found} lookahead issues out of {len(self.portfolio_history)} trades")
        return issues_found == 0

    def monte_carlo_test(self, num_simulations=1000):
        """
        Monte Carlo test for statistical significance
        """
        print("\n" + "=" * 60)
        print("MONTE CARLO SIGNIFICANCE TEST")
        print("=" * 60)

        if not self.portfolio_history:
            print("No portfolio history available")
            return None

        # Get actual performance
        actual_pnls = []
        for trade in self.portfolio_history:
            ts = trade['timestamp']
            metrics = self.performance_metrics.get(ts, {})
            pnl = metrics.get('realized_pnl', 0)
            actual_pnls.append(pnl)

        actual_total = sum(actual_pnls)
        actual_return = actual_total / self.C_0

        # Calculate statistics of actual returns
        actual_mean = np.mean(actual_pnls)
        actual_std = np.std(actual_pnls, ddof=1)
        actual_sharpe = (np.mean(actual_pnls) / np.std(actual_pnls)) * np.sqrt(252 / 5) if np.std(
            actual_pnls) > 0 else 0

        print(f"Actual Strategy:")
        print(f"  Total P&L: ${actual_total:,.0f}")
        print(f"  Total Return: {actual_return:.2%}")
        print(f"  Avg Period P&L: ${actual_mean:,.0f}")
        print(f"  Period Std Dev: ${actual_std:,.0f}")
        print(f"  Estimated Sharpe: {actual_sharpe:.2f}")

        # Generate random strategies with same risk characteristics
        random_totals = []
        random_returns = []

        for i in range(num_simulations):
            # Random returns with same mean and std as actual strategy
            random_pnls = np.random.normal(actual_mean, actual_std, len(actual_pnls))
            random_total = sum(random_pnls)
            random_totals.append(random_total)
            random_returns.append(random_total / self.C_0)

        # Calculate p-value
        p_value = np.mean([r >= actual_total for r in random_totals])

        print(f"\nMonte Carlo Results ({num_simulations} simulations):")
        print(f"  Probability of random strategy beating ours: {p_value:.3%}")
        print(
            f"  95% Confidence Interval for random strategy: [{np.percentile(random_returns, 2.5):.2%}, {np.percentile(random_returns, 97.5):.2%}]")

        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(random_returns, bins=50, alpha=0.7, label='Random Strategies')
        plt.axvline(x=actual_return, color='red', linewidth=3, label=f'Actual Strategy ({actual_return:.2%})')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.title(f'Monte Carlo Test: p-value = {p_value:.3%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return {
            'actual_return': actual_return,
            'p_value': p_value,
            'random_returns': random_returns
        }
    def weekly_rebalancing_step(self, current_date, actual_returns=None):
        """
        10. Weekly Rebalancing Steps - now takes current_date and recalculates everything
        """
        # Set the current 252-day window ending at current_date
        self.set_current_window(current_date)

        if len(self.P) < 50:  # Need minimum data
            return None, None

        # Recalculate everything for the current window
        self.prepare_returns_matrix(standardize=False)
        self.compute_covariance_and_pca(n_components=5)
        self.factor_pc_regression(min_r_squared=.1, max_correlation=0.9, max_vif=5)
        self.compute_historic_pc_std()

        # Get current factor changes from the current window
        current_factor_changes = self.calculate_current_factor_changes(lookback_days=5)

        # Get current prices (use last available prices from current window)
        current_prices = self.P.iloc[-1].values

        # Calculate stock betas relative to sector benchmark (XLE) for current window
        stock_betas = self.calculate_stock_betas(benchmark_symbol='XLE', lookback_days=252)

        # Predict PC movements ΔPC_pred
        self.predict_pc_movement(current_factor_changes)

        # Compute r_hat and r_hat_weighted
        self.predict_stock_returns()
        self.compute_centrality_weighting()

        # Solve optimization for weights w_i
        self.optimize_portfolio(current_prices, stock_betas)

        # Execute trades
        trade_record = self.execute_trades(current_prices)

        # Record performance metrics
        metrics = self.record_performance_metrics(actual_returns)

        return trade_record, metrics

    def backtest_strategy(self, start_date, end_date, rebalance_freq=5):
        """
        Comprehensive backtesting with rolling window approach
        """
        print(f"\n=== Backtesting Strategy from {start_date} to {end_date} ===")

        # Get all available dates in the backtest period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter to backtest period
        backtest_dates = self.P_full[(self.P_full.index >= start_dt) &
                                     (self.P_full.index <= end_dt)].index

        if len(backtest_dates) == 0:
            print("No data available for backtesting period")
            return None

        portfolio_values = [self.C_0]  # Start with initial capital
        rebalance_dates = []

        # Simulate rebalancing every rebalance_freq days
        for i in range(0, len(backtest_dates), rebalance_freq):
            current_date = backtest_dates[i]
            rebalance_dates.append(current_date)

            # Calculate actual returns for the period (if we have previous trade)
            actual_returns = None
            if len(self.portfolio_history) > 0:
                # Get returns from last rebalancing date to current date
                prev_date_idx = max(0, i - rebalance_freq)
                if prev_date_idx < i:
                    period_data = self.P_full.loc[backtest_dates[prev_date_idx]:current_date]
                    if len(period_data) >= 2:
                        period_returns = (period_data.iloc[-1] / period_data.iloc[0] - 1).values
                        actual_returns = period_returns

            try:
                # Execute rebalancing step with rolling window
                trade_record, metrics = self.weekly_rebalancing_step(current_date, actual_returns)

                # Calculate portfolio return
                if trade_record is not None and metrics is not None and 'realized_pnl' in metrics:
                    new_portfolio_value = portfolio_values[-1] + metrics['realized_pnl']
                    portfolio_values.append(new_portfolio_value)
                else:
                    portfolio_values.append(portfolio_values[-1])  # No change if no valid trades

            except Exception as e:
                print(f"Error at {current_date}: {e}")
                portfolio_values.append(portfolio_values[-1])

        # Calculate performance metrics
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            num_periods = len(portfolio_values) - 1
            annualized_return = (1 + total_return) ** (252 / (num_periods * rebalance_freq)) - 1
        else:
            total_return = 0
            annualized_return = 0

        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")

        # Plot backtest results
        plt.figure(figsize=(12, 6))
        plt.plot(rebalance_dates[:len(portfolio_values) - 1], portfolio_values[1:],
                 marker='o', label='Strategy Portfolio Value')
        plt.axhline(y=self.C_0, color='r', linestyle='--', label='Initial Capital')
        plt.title('Strategy Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return {
            'portfolio_values': portfolio_values,
            'rebalance_dates': rebalance_dates,
            'total_return': total_return,
            'annualized_return': annualized_return
        }


    def analyze_pc_regressions(self, n_pcs=None):
        if not hasattr(self, 'pc_regressions') or not self.pc_regressions:
            return
        if n_pcs is None:
            n_pcs = self.V.shape[1]
        print("\n=== PC Regressions Analysis ===")
        for k in range(n_pcs):
            if k not in self.pc_regressions:
                continue
            data = self.pc_regressions[k]
            print(f"\nPC_{k + 1}:")
            print(f"Selected factors: {', '.join(data['factor_names'])}")
            print("Individual regressions:")
            individual_r2 = data.get('individual_r2', {})
            for name in data['factor_names']:
                ind = individual_r2.get(name, {'r2': 'N/A', 'pvalue': 'N/A'})
                print(f"  {name}: R² = {ind['r2']:.3f}, p-value = {ind['pvalue']:.4f}")
            print("VIF for selected factors:")
            for name, vif in data.get('vif', {}).items():
                print(f"  {name}: {vif:.2f}")
            equation = f"PC_{k + 1}_t = {data['alpha']:.4f}"
            for beta_val, name in zip(data['beta'], data['factor_names']):
                equation += f" + ({beta_val:.4f} * Δ{name}_t)"
            equation += " + ε_t"
            print("Full linear regression equation:")
            print(equation)
            print(f"Multiple R²: {data['r_squared']:.3f}")
            print(f"Correlation between regression line and PC: {data['correlation']:.3f}")
            print("\nRegression Summary:")
            print(data['model'].summary())

        # Collect unique factors for matrices
        unique_factors = set()
        for data in self.pc_regressions.values():
            unique_factors.update(data['factor_names'])
        unique_factors = sorted(unique_factors)

        # Beta coefficients matrix
        beta_matrix = np.zeros((n_pcs, len(unique_factors)))
        for k in range(n_pcs):
            if k in self.pc_regressions:
                data = self.pc_regressions[k]
                for b, name in zip(data['beta'], data['factor_names']):
                    col_idx = unique_factors.index(name)
                    beta_matrix[k, col_idx] = b
        df_beta = pd.DataFrame(beta_matrix, index=[f"PC_{i + 1}" for i in range(n_pcs)], columns=unique_factors)
        print("\n=== Beta Coefficients Matrix ===")
        print(df_beta.to_string(float_format="%.4f"))

        # Individual R² matrix
        ind_r2_matrix = np.full((n_pcs, len(unique_factors)), np.nan)
        for k in range(n_pcs):
            if k in self.pc_regressions:
                data = self.pc_regressions[k]
                for name in unique_factors:
                    if name in data['individual_r2']:
                        col_idx = unique_factors.index(name)
                        ind_r2_matrix[k, col_idx] = data['individual_r2'][name]['r2']
        df_r2 = pd.DataFrame(ind_r2_matrix, index=[f"PC_{i + 1}" for i in range(n_pcs)], columns=unique_factors)
        print("\n=== Individual R² Matrix ===")
        print(df_r2.to_string(float_format="%.3f"))

    def plot_portfolio_metrics(self):
        """
        11. Portfolio Metrics & Graphs
        """
        if not self.portfolio_history:
            print("No portfolio history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data for plotting
        timestamps = [trade['timestamp'] for trade in self.portfolio_history]
        long_exposure = [self.performance_metrics[ts]['long_exposure']
                         for ts in timestamps if ts in self.performance_metrics]
        short_exposure = [self.performance_metrics[ts]['short_exposure']
                          for ts in timestamps if ts in self.performance_metrics]

        # Portfolio composition by long/short exposure
        if long_exposure and short_exposure:
            axes[0, 0].plot(timestamps[:len(long_exposure)], long_exposure, label='Long (65%)', color='green')
            axes[0, 0].plot(timestamps[:len(short_exposure)], short_exposure, label='Short (35%)', color='red')
            axes[0, 0].set_title('Long/Short Exposure Over Time')
            axes[0, 0].set_ylabel('Dollar Exposure')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # Portfolio value over time
        portfolio_values = [np.sum(np.abs(trade['weights'])) for trade in self.portfolio_history]
        axes[0, 1].plot(timestamps, portfolio_values, marker='o')
        axes[0, 1].set_title('Portfolio Value Over Time')
        axes[0, 1].set_ylabel('Total Portfolio Value ($)')
        axes[0, 1].grid(True)

        # Transaction costs
        transaction_costs = [trade['transaction_cost'] for trade in self.portfolio_history]
        axes[1, 0].bar(range(len(transaction_costs)), transaction_costs)
        axes[1, 0].set_title('Transaction Costs per Rebalancing')
        axes[1, 0].set_ylabel('Transaction Cost ($)')
        axes[1, 0].set_xlabel('Rebalancing Period')
        axes[1, 0].grid(True)

        # Buy/Sell counts
        buy_counts = [np.sum(trade['weights'] > 0) for trade in self.portfolio_history]
        sell_counts = [np.sum(trade['weights'] < 0) for trade in self.portfolio_history]

        x = range(len(buy_counts))
        width = 0.35
        axes[1, 1].bar([i - width / 2 for i in x], buy_counts, width, label='Long Positions', color='green', alpha=0.7)
        axes[1, 1].bar([i + width / 2 for i in x], sell_counts, width, label='Short Positions', color='red', alpha=0.7)
        axes[1, 1].set_title('Number of Long/Short Positions')
        axes[1, 1].set_ylabel('Number of Positions')
        axes[1, 1].set_xlabel('Rebalancing Period')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

        return fig

    def get_optimization_stats(self):
        print("\n=== Optimization Statistics ===")
        print(f"Successful SLSQP Optimizations: {self.slsqp_success_count}")
        print(f"Least Squares Fallbacks: {self.lsq_fallback_count}")
        total = self.slsqp_success_count + self.lsq_fallback_count
        if total > 0:
            print(f"SLSQP Success Rate: {self.slsqp_success_count / total:.2%}")
        else:
            print("No optimizations performed yet")
        return {
            'slsqp_success_count': self.slsqp_success_count,
            'lsq_fallback_count': self.lsq_fallback_count
        }
    def get_variable_summary(self):
        """
        Display all variables and matrices as defined in the strategy
        """
        summary = {
            'R': f"Returns matrix shape: {self.R.shape if self.R is not None else 'Not computed'}",
            'Sigma': f"Covariance matrix shape: {self.Sigma.shape if self.Sigma is not None else 'Not computed'}",
            'V': f"Stock-to-PC loadings shape: {self.V.shape if self.V is not None else 'Not computed'}",
            'PC_day': f"PC time series shape: {self.PC_day.shape if self.PC_day is not None else 'Not computed'}",
            'sigma_PC': f"Historic PC std shape: {self.sigma_PC.shape if hasattr(self, 'sigma_PC') else 'Not computed'}",
            'delta_PC_pred': f"Predicted PC movement shape: {self.delta_PC_pred.shape if hasattr(self, 'delta_PC_pred') else 'Not computed'}",
            'r_hat': f"Predicted stock returns shape: {self.r_hat.shape if hasattr(self, 'r_hat') else 'Not computed'}",
            'u_centrality': f"Centrality weights shape: {self.u_centrality.shape if self.u_centrality is not None else 'Not computed'}",
            'r_hat_weighted': f"Weighted predicted returns shape: {self.r_hat_weighted.shape if hasattr(self, 'r_hat_weighted') else 'Not computed'}",
            'optimal_weights': f"Portfolio weights shape: {self.optimal_weights.shape if hasattr(self, 'optimal_weights') else 'Not computed'}"
        }

        return summary


    #the following functions are for validation of the back testing

    def validate_data_integrity_and_lookahead(self):
        """
        Comprehensive validation to detect look-ahead bias and data leakage
        """
        print("\n" + "=" * 70)
        print("DATA INTEGRITY AND LOOK-AHEAD BIAS VALIDATION")
        print("=" * 70)

        validation_results = {}
        issues_found = []

        # 1. Check if future data is being used in any calculations
        print("\n1. FUTURE DATA LEAKAGE CHECK:")
        print("-" * 40)

        for i, trade in enumerate(self.portfolio_history[:5]):  # Check first 5 trades
            trade_date = trade['timestamp']
            print(f"\nTrade {i + 1} on {trade_date}:")

            # Check if the window data extends beyond trade date
            self.set_current_window(trade_date)
            window_end = self.P.index[-1]
            window_start = self.P.index[0]

            if window_end > trade_date:
                issues_found.append(f"CRITICAL: Trade {i + 1} uses data after trade date!")
                print(f"  ❌ ISSUE: Window ends {window_end} but trade date is {trade_date}")
            else:
                print(f"  ✅ OK: Window ends {window_end}, trade date {trade_date}")

            print(f"     Window: {window_start} to {window_end} ({len(self.P)} days)")

            # Check factor data alignment
            if self.factors_data is not None:
                factor_end = self.factors_data.index[-1]
                if factor_end > trade_date:
                    issues_found.append(f"CRITICAL: Factor data leakage in trade {i + 1}")
                    print(f"  ❌ ISSUE: Factor data ends {factor_end} but trade date is {trade_date}")
                else:
                    print(f"  ✅ OK: Factor data ends {factor_end}")

        # 2. Check rolling window independence
        print(f"\n2. ROLLING WINDOW INDEPENDENCE CHECK:")
        print("-" * 40)

        if len(self.portfolio_history) >= 3:
            # Compare consecutive windows
            dates = [trade['timestamp'] for trade in self.portfolio_history[:3]]

            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]

                # Set windows
                self.set_current_window(current_date)
                window1_start = self.P.index[0]
                window1_end = self.P.index[-1]

                self.set_current_window(next_date)
                window2_start = self.P.index[0]
                window2_end = self.P.index[-1]

                # Check expected progression
                expected_days_diff = (next_date - current_date).days
                actual_start_diff = (window2_start - window1_start).days
                actual_end_diff = (window2_end - window1_end).days

                print(f"\nWindow progression {i + 1} to {i + 2}:")
                print(f"  Trade dates: {current_date} → {next_date} ({expected_days_diff} days)")
                print(f"  Window starts: {window1_start} → {window2_start} ({actual_start_diff} days)")
                print(f"  Window ends: {window1_end} → {window2_end} ({actual_end_diff} days)")

                if abs(actual_end_diff - expected_days_diff) > 2:  # Allow 2-day tolerance for weekends
                    issues_found.append(f"Window progression issue between trades {i + 1} and {i + 2}")
                    print(f"  ⚠️  Warning: Unexpected window progression")
                else:
                    print(f"  ✅ OK: Windows progress as expected")

        # 3. Check for data snooping in factor selection
        print(f"\n3. FACTOR SELECTION CONSISTENCY CHECK:")
        print("-" * 40)

        factor_usage = {}
        if len(self.portfolio_history) >= 5:
            # Check factor selection across multiple periods
            sample_dates = [trade['timestamp'] for trade in self.portfolio_history[::10]]  # Every 10th trade

            for date in sample_dates[:3]:
                self.set_current_window(date)
                self.prepare_returns_matrix()
                self.compute_covariance_and_pca(n_components=5)
                temp_regressions = self.factor_pc_regression()

                if temp_regressions:
                    for pc_idx, data in temp_regressions.items():
                        factors = data.get('factor_names', [])
                        date_str = date.strftime('%Y-%m-%d')
                        factor_usage[date_str] = factors
                        print(f"  {date_str}: {len(factors)} factors for PC regressions")

            # Check if same factors are always selected (potential overfitting)
            all_factors = set()
            for factors in factor_usage.values():
                all_factors.update(factors)

            common_factors = set.intersection(
                *[set(factors) for factors in factor_usage.values()]) if factor_usage else set()

            print(f"\n  Total unique factors used: {len(all_factors)}")
            print(f"  Factors used in ALL periods: {len(common_factors)} - {list(common_factors)}")

            if len(common_factors) > 5:
                issues_found.append("Potential overfitting: Same factors always selected")
                print(f"  ⚠️  Warning: High factor consistency may indicate overfitting")

        # Summary
        print(f"\n4. VALIDATION SUMMARY:")
        print("-" * 40)

        if issues_found:
            print("❌ CRITICAL ISSUES FOUND:")
            for issue in issues_found:
                print(f"  • {issue}")
        else:
            print("✅ No critical data integrity issues detected")

        validation_results['issues_found'] = issues_found
        validation_results['factor_usage'] = factor_usage if 'factor_usage' in locals() else {}

        return validation_results

    def analyze_prediction_accuracy(self):
        """
        Detailed analysis of prediction accuracy vs actual returns, using regression-based R²
        """
        print("\n" + "=" * 70)
        print("PREDICTION ACCURACY ANALYSIS")
        print("=" * 70)

        if len(self.portfolio_history) < 2:
            print("Insufficient data for prediction accuracy analysis")
            return {
                'avg_correlation': None,
                'correlation_by_period': [],
                'prediction_dates': [],
                'r_squared': None
            }

        predicted_returns = []
        actual_returns = []
        prediction_dates = []
        correlation_by_period = []

        print("\n1. PREDICTION vs ACTUAL CORRELATION BY PERIOD:")
        print("-" * 50)

        for i, trade in enumerate(self.portfolio_history):
            if 'actual_returns' in trade and 'expected_returns' in trade:
                pred = trade['expected_returns']
                actual = trade['actual_returns']
                date = trade['timestamp']

                # Calculate correlation for this period
                if len(pred) > 1 and len(actual) > 1:
                    correlation = np.corrcoef(pred, actual)[0, 1]
                    if not np.isnan(correlation):
                        correlation_by_period.append(correlation)
                        prediction_dates.append(date)

                        # Store individual predictions and actuals
                        predicted_returns.extend(pred)
                        actual_returns.extend(actual)

                        if i < 10:  # Show first 10 periods
                            print(f"  {date.strftime('%Y-%m-%d')}: Correlation = {correlation:.3f}")

        if not correlation_by_period:
            print("No valid correlation data available")
            return {
                'avg_correlation': None,
                'correlation_by_period': [],
                'prediction_dates': [],
                'r_squared': None
            }

        avg_correlation = np.mean(correlation_by_period)
        std_correlation = np.std(correlation_by_period)

        print(f"\n2. OVERALL PREDICTION STATISTICS:")
        print("-" * 50)
        print(f"  Average correlation: {avg_correlation:.3f}")
        print(f"  Std deviation: {std_correlation:.3f}")
        print(f"  Min correlation: {min(correlation_by_period):.3f}")
        print(f"  Max correlation: {max(correlation_by_period):.3f}")
        print(
            f"  Periods with positive correlation: {sum(1 for c in correlation_by_period if c > 0)}/{len(correlation_by_period)}")

        # Overall correlation across all predictions
        overall_corr = np.corrcoef(predicted_returns, actual_returns)[
            0, 1] if predicted_returns and actual_returns else np.nan
        print(f"  Overall correlation (all data): {overall_corr:.3f}")

        # Scatter plot with best-fit line and R²
        if len(predicted_returns) > 100:  # Only if we have enough data
            sample_size = min(10000, len(predicted_returns))  # Sample for visibility
            sample_indices = np.random.choice(len(predicted_returns), sample_size, replace=False)

            pred_sample = np.array(predicted_returns)[sample_indices]
            actual_sample = np.array(actual_returns)[sample_indices]

            # Check for scaling issues and rescale if necessary
            if np.mean(np.abs(pred_sample)) > 1:  # Assume scaling if predictions are unusually large
                pred_sample = pred_sample / 100
                print("Rescaled predicted returns by dividing by 100")

            plt.figure(figsize=(8, 6))
            plt.scatter(pred_sample, actual_sample, alpha=0.5, s=20)

            # Perfect prediction line (y = x) for reference
            min_val = min(np.min(pred_sample), np.min(actual_sample))
            max_val = max(np.max(pred_sample), np.max(actual_sample))
            margin = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
            line_range = [min_val - margin, max_val + margin]
            plt.plot(line_range, line_range, 'r--', alpha=0.7, label='Perfect Prediction (y=x)')

            # Best-fit line using linear regression
            coeffs = np.polyfit(pred_sample, actual_sample, 1)  # Linear regression
            best_fit_line = np.poly1d(coeffs)
            plt.plot(line_range, best_fit_line(line_range), 'b-', alpha=0.7, label='Best Fit')

            # Calculate R² for the best-fit line
            fitted_predictions = best_fit_line(pred_sample)
            ss_tot = np.sum((actual_sample - np.mean(actual_sample)) ** 2)
            ss_res = np.sum((actual_sample - fitted_predictions) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            # Display R² in top-right corner
            plt.text(line_range[1], line_range[0], f'R²: {r_squared:.3f}',
                     fontsize=10, ha='right', va='bottom', color='black')

            plt.xlabel('Predicted Returns')
            plt.ylabel('Actual Returns')
            plt.title(f'Predicted vs Actual Returns (Sample of {sample_size})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()
            print(f"Regression-based R²: {r_squared:.3f}")

        # Plot correlation over time
        plt.figure(figsize=(12, 4))
        plt.plot(prediction_dates, correlation_by_period, marker='o', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_correlation, color='g', linestyle='-', alpha=0.7,
                    label=f'Avg: {avg_correlation:.3f}')
        plt.title('Prediction Correlation Over Time')
        plt.ylabel('Correlation (Predicted vs Actual)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return {
            'avg_correlation': avg_correlation,
            'correlation_by_period': correlation_by_period,
            'prediction_dates': prediction_dates,
            'r_squared': r_squared
        }

    def analyze_portfolio_composition_over_time(self):
        """
        Detailed visualization of portfolio composition changes
        """
        print("\n" + "=" * 70)
        print("PORTFOLIO COMPOSITION ANALYSIS")
        print("=" * 70)

        if not self.portfolio_history:
            print("No portfolio history available")
            return

        # Extract data
        dates = [trade['timestamp'] for trade in self.portfolio_history]
        weights_history = np.array([trade['weights'] for trade in self.portfolio_history])
        stock_names = self.stocks

        print(f"\n1. PORTFOLIO STATISTICS:")
        print("-" * 30)
        print(f"  Number of rebalancing periods: {len(dates)}")
        print(f"  Number of stocks: {len(stock_names)}")
        print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

        # Calculate position statistics
        long_counts = np.sum(weights_history > 0, axis=1)
        short_counts = np.sum(weights_history < 0, axis=1)

        print(f"  Avg long positions: {np.mean(long_counts):.1f} ± {np.std(long_counts):.1f}")
        print(f"  Avg short positions: {np.mean(short_counts):.1f} ± {np.std(short_counts):.1f}")

        # 2. Long/Short exposure over time
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        long_exposure = []
        short_exposure = []
        net_exposure = []

        for weights in weights_history:
            long_exp = np.sum(weights[weights > 0])
            short_exp = np.abs(np.sum(weights[weights < 0]))
            net_exp = np.sum(weights)

            long_exposure.append(long_exp)
            short_exposure.append(short_exp)
            net_exposure.append(net_exp)

        # Long/Short exposure
        axes[0, 0].plot(dates, long_exposure, label='Long', color='green', alpha=0.8)
        axes[0, 0].plot(dates, short_exposure, label='Short', color='red', alpha=0.8)
        axes[0, 0].axhline(y=0.65 * self.C_0, color='green', linestyle='--', alpha=0.5, label='Target Long (65%)')
        axes[0, 0].axhline(y=0.35 * self.C_0, color='red', linestyle='--', alpha=0.5, label='Target Short (35%)')
        axes[0, 0].set_title('Long/Short Exposure Over Time')
        axes[0, 0].set_ylabel('Dollar Exposure')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Net exposure (should be near zero for market neutral)
        axes[0, 1].plot(dates, net_exposure, color='blue', alpha=0.8)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Net Exposure (Market Neutrality)')
        axes[0, 1].set_ylabel('Net Dollar Exposure')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Position counts over time
        axes[1, 0].plot(dates, long_counts, label='Long Positions', color='green', marker='o', markersize=3)
        axes[1, 0].plot(dates, short_counts, label='Short Positions', color='red', marker='o', markersize=3)
        axes[1, 0].set_title('Number of Positions Over Time')
        axes[1, 0].set_ylabel('Number of Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Portfolio concentration (top positions)
        concentration_ratios = []
        for weights in weights_history:
            abs_weights = np.abs(weights)
            sorted_weights = np.sort(abs_weights)[::-1]
            top_5_concentration = np.sum(sorted_weights[:5]) / np.sum(abs_weights) if np.sum(abs_weights) > 0 else 0
            concentration_ratios.append(top_5_concentration)

        axes[1, 1].plot(dates, concentration_ratios, color='purple', alpha=0.8)
        axes[1, 1].set_title('Portfolio Concentration (Top 5 Positions)')
        axes[1, 1].set_ylabel('Top 5 Weight Ratio')
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Individual stock weight heatmap (sample of stocks)
        sample_stocks = stock_names[:10]  # Show first 10 stocks
        sample_indices = list(range(10))
        sample_weights = weights_history[:, sample_indices].T

        im = axes[2, 0].imshow(sample_weights, cmap='RdBu_r', aspect='auto',
                               vmin=-np.max(np.abs(sample_weights)), vmax=np.max(np.abs(sample_weights)))
        axes[2, 0].set_title('Stock Weights Over Time (Sample)')
        axes[2, 0].set_xlabel('Rebalancing Period')
        axes[2, 0].set_ylabel('Stock')
        axes[2, 0].set_yticks(range(len(sample_stocks)))
        axes[2, 0].set_yticklabels(sample_stocks)
        plt.colorbar(im, ax=axes[2, 0], label='Weight ($)')

        # 6. Turnover analysis
        turnover_rates = []
        for i in range(1, len(weights_history)):
            prev_weights = weights_history[i - 1]
            curr_weights = weights_history[i]
            turnover = np.sum(np.abs(curr_weights - prev_weights)) / (2 * self.C_0)  # Two-way turnover
            turnover_rates.append(turnover)

        axes[2, 1].plot(dates[1:], turnover_rates, color='orange', alpha=0.8)
        axes[2, 1].set_title('Portfolio Turnover Rate')
        axes[2, 1].set_ylabel('Turnover Rate')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\n2. TURNOVER ANALYSIS:")
        print("-" * 30)
        if turnover_rates:
            print(f"  Average turnover rate: {np.mean(turnover_rates):.2%}")
            print(f"  Median turnover rate: {np.median(turnover_rates):.2%}")
            print(f"  Max turnover rate: {np.max(turnover_rates):.2%}")
            print(f"  High turnover periods (>50%): {sum(1 for t in turnover_rates if t > 0.5)}")

        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'turnover_rates': turnover_rates,
            'concentration_ratios': concentration_ratios
        }


    def detect_overfitting_signals(self):
        """
        Analyze potential overfitting by examining strategy behavior patterns
        """
        print("\n" + "=" * 70)
        print("OVERFITTING DETECTION ANALYSIS")
        print("=" * 70)

        overfitting_signals = []

        # 1. Check if win rate is too consistent
        if hasattr(self, 'performance_metrics'):
            win_rates_by_period = []

            for trade in self.portfolio_history:
                if 'actual_returns' in trade:
                    weights = trade['weights']
                    actual_returns = trade['actual_returns']
                    pos_pnl = weights * actual_returns

                    if len(pos_pnl) > 0:
                        wins = np.sum(pos_pnl > 0)
                        total = len(pos_pnl)
                        win_rate = wins / total
                        win_rates_by_period.append(win_rate)

            if win_rates_by_period:
                avg_win_rate = np.mean(win_rates_by_period)
                win_rate_std = np.std(win_rates_by_period)

                print(f"\n1. WIN RATE CONSISTENCY CHECK:")
                print(f"   Average win rate: {avg_win_rate:.1%}")
                print(f"   Win rate std dev: {win_rate_std:.1%}")
                print(f"   Coefficient of variation: {win_rate_std / avg_win_rate:.2f}")

                if avg_win_rate > 0.75:
                    overfitting_signals.append("Exceptionally high win rate (>75%)")

                if win_rate_std < 0.05:  # Very low variability
                    overfitting_signals.append("Win rate too consistent (low variability)")

        # 2. Check period-specific performance
        print(f"\n2. PERIOD-SPECIFIC PERFORMANCE:")

        # Split into year-by-year performance
        yearly_performance = {}
        for trade in self.portfolio_history:
            year = trade['timestamp'].year
            ts = trade['timestamp']
            metrics = self.performance_metrics.get(ts, {})
            pnl = metrics.get('realized_pnl', 0)

            if year not in yearly_performance:
                yearly_performance[year] = []
            yearly_performance[year].append(pnl)

        for year, pnls in yearly_performance.items():
            if pnls:
                year_return = sum(pnls) / self.C_0
                print(f"   {year}: {year_return:.1%} return ({len(pnls)} trades)")

        # Check if performance is too concentrated in specific periods
        year_returns = [sum(pnls) / self.C_0 for pnls in yearly_performance.values() if pnls]
        if year_returns and len(year_returns) > 1:
            max_year_return = max(year_returns)
            if max_year_return > 2 * np.mean(year_returns):  # One year dominates
                overfitting_signals.append("Performance highly concentrated in specific period")

        # 3. Check strategy stability across different market conditions
        print(f"\n3. MARKET CONDITION SENSITIVITY:")

        # Use VIX-like proxy or market volatility
        if self.factors_data_full is not None and '^VIX' in self.factors_data_full.columns:
            # Analyze performance in high vs low volatility periods
            vix_data = self.factors_data_full['^VIX'].dropna()

            if len(vix_data) > 100:
                vix_median = vix_data.median()
                high_vol_performance = []
                low_vol_performance = []

                for trade in self.portfolio_history:
                    trade_date = trade['timestamp']
                    # Get VIX around trade date
                    vix_window = vix_data[vix_data.index <= trade_date].tail(5)
                    if len(vix_window) > 0:
                        avg_vix = vix_window.mean()
                        ts = trade['timestamp']
                        metrics = self.performance_metrics.get(ts, {})
                        pnl = metrics.get('realized_pnl', 0)

                        if avg_vix > vix_median:
                            high_vol_performance.append(pnl)
                        else:
                            low_vol_performance.append(pnl)

                if high_vol_performance and low_vol_performance:
                    high_vol_return = sum(high_vol_performance) / self.C_0 / len(high_vol_performance) * 52
                    low_vol_return = sum(low_vol_performance) / self.C_0 / len(low_vol_performance) * 52

                    print(
                        f"   High volatility periods: {high_vol_return:.1%} annualized ({len(high_vol_performance)} trades)")
                    print(
                        f"   Low volatility periods: {low_vol_return:.1%} annualized ({len(low_vol_performance)} trades)")

                    if abs(high_vol_return - low_vol_return) > 0.5:  # 50% difference
                        overfitting_signals.append("Large performance difference across volatility regimes")

        # 4. Summary
        print(f"\n4. OVERFITTING RISK ASSESSMENT:")
        print("-" * 40)

        if overfitting_signals:
            print("⚠️  POTENTIAL OVERFITTING SIGNALS DETECTED:")
            for signal in overfitting_signals:
                print(f"   • {signal}")

            print(
                f"\n   Risk Level: {'HIGH' if len(overfitting_signals) >= 3 else 'MEDIUM' if len(overfitting_signals) >= 2 else 'LOW'}")
        else:
            print("✅ No obvious overfitting signals detected")

        return overfitting_signals

    def walk_forward_test(self, train_start, train_end, test_start, test_end, steps=4):
        """
        Walk-forward validation across multiple periods
        """
        print(f"\nWALK-FORWARD VALIDATION: {steps} steps")

        results = []

        # Calculate date ranges for each step
        total_days = (pd.to_datetime(test_end) - pd.to_datetime(train_start)).days
        step_days = total_days // steps

        for step in range(steps):
            # Calculate dates for this step
            step_train_start = pd.to_datetime(train_start) + pd.Timedelta(days=step * step_days)
            step_train_end = step_train_start + pd.Timedelta(days=step_days * 0.7)  # 70% train
            step_test_start = step_train_end
            step_test_end = step_train_start + pd.Timedelta(days=step_days)

            print(f"\nStep {step + 1}: Train {step_train_start.date()} to {step_train_end.date()}")
            print(f"          Test  {step_test_start.date()} to {step_test_end.date()}")

            # Create new strategy for this step
            step_strategy = UltimateDetailedStrategy()
            step_strategy.load_full_data(self.stocks,
                                         step_train_start.strftime('%Y-%m-%d'),
                                         step_test_end.strftime('%Y-%m-%d'),
                                         {s: s for s in self.factors_data_full.columns if
                                          s in self.factors_data_full.columns})

            # Test on out-of-sample period
            step_results = step_strategy.backtest_strategy(
                step_test_start.strftime('%Y-%m-%d'),
                step_test_end.strftime('%Y-%m-%d'),
                rebalance_freq=5
            )

            if step_results:
                results.append(step_results['total_return'])
                print(f"  Return: {step_results['total_return']:.2%}")

        return results


# Add this to the main strategy class - update the run_strategy_example function
def run_enhanced_validation_example(strategy):
    """
    Run comprehensive validation for the given strategy object
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION ANALYSIS")
    print("=" * 80)

    # 1. Data integrity and look-ahead bias check
    validation_results = strategy.validate_data_integrity_and_lookahead()

    # 2. Prediction accuracy analysis
    prediction_analysis = strategy.analyze_prediction_accuracy()

    # 3. Portfolio composition analysis
    composition_analysis = strategy.analyze_portfolio_composition_over_time()

    # 4. Transaction cost stress test
    cost_analysis = strategy.stress_test_transaction_costs()

    # 5. Overfitting detection
    overfitting_signals = strategy.detect_overfitting_signals()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    critical_issues = validation_results.get('issues_found', [])
    if critical_issues:
        print("❌ CRITICAL ISSUES THAT NEED INVESTIGATION:")
        for issue in critical_issues:
            print(f"   • {issue}")

    if overfitting_signals:
        print("\n⚠️  OVERFITTING CONCERNS:")
        for signal in overfitting_signals:
            print(f"   • {signal}")

    if prediction_analysis and prediction_analysis.get('avg_correlation'):
        corr = prediction_analysis['avg_correlation']
        print(f"\n📊 PREDICTION QUALITY: {corr:.3f} average correlation")
        if corr < 0.1:
            print("   ⚠️  Low prediction accuracy may indicate noise fitting")

    print(f"\n💰 TRANSACTION COST SENSITIVITY:")
    if cost_analysis:
        base_return = cost_analysis[0.001]['annualized_return']
        realistic_return = cost_analysis[0.005]['annualized_return']  # 0.5% costs
        print(f"   Current costs (0.1%): {base_return:.1%} annual return")
        print(f"   Realistic costs (0.5%): {realistic_return:.1%} annual return")
        print(f"   Impact: {(base_return - realistic_return) * 100:.1f} percentage points")

    return strategy



# After running your backtest, add these validation calls:
def run_strategy_example():
    """
    Complete example with validation
    """
    print("=== Ultimate Detailed Strategy Example ===")

    # Initialize strategy
    strategy = UltimateDetailedStrategy()

    # Define your universes
    stock_symbols = ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'EOG', 'DVN', 'APA']
    factor_symbols = ['XLE', 'XOP', 'OIH', 'CL=F', '^VIX', '^TNX']

    # 1. IN-SAMPLE TEST (your original period)
    print("\n1. IN-SAMPLE TEST (2021-2023)")
    strategy.load_full_data(stock_symbols, '2021-01-01', '2024-01-01',
                            {symbol: symbol for symbol in factor_symbols})

    backtest_results = strategy.backtest_strategy('2021-01-01', '2023-12-31', rebalance_freq=5)

    # Run validations
    strategy.validate_no_lookahead()
    mc_results = strategy.monte_carlo_test(num_simulations=1000)

    # 2. OUT-OF-SAMPLE TEST (completely different period)
    print("\n" + "=" * 60)
    print("2. OUT-OF-SAMPLE TEST (2018-2020)")
    print("=" * 60)

    # Create new strategy instance for clean test
    strategy_oos = UltimateDetailedStrategy()
    strategy_oos.load_full_data(stock_symbols, '2018-01-01', '2021-01-01',
                                {symbol: symbol for symbol in factor_symbols})

    oos_results = strategy_oos.backtest_strategy('2018-01-01', '2020-12-31', rebalance_freq=5)

    # Run validations on out-of-sample
    strategy_oos.validate_no_lookahead()
    oos_mc_results = strategy_oos.monte_carlo_test(num_simulations=1000)

    walk_forward_returns = strategy.walk_forward_test(
        train_start='2018-01-01',
        train_end='2023-12-31',
        test_start='2019-01-01',
        test_end='2023-12-31',
        steps=5
    )

    print(f"\nWalk-Forward Average Return: {np.mean(walk_forward_returns):.2%}")
    print(f"Walk-Forward Std Dev: {np.std(walk_forward_returns):.2%}")

    # 3. Compare results
    print("\n" + "=" * 60)
    print("COMPARISON: In-Sample vs Out-of-Sample")
    print("=" * 60)

    is_return = backtest_results['total_return'] if backtest_results else 0
    oos_return = oos_results['total_return'] if oos_results else 0

    print(f"In-Sample Return (2021-2023): {is_return:.2%}")
    print(f"Out-of-Sample Return (2018-2020): {oos_return:.2%}")
    print(f"Performance Drop: {(is_return - oos_return):.2%} points")

    return strategy, strategy_oos


# Additional utility functions for analysis
def analyze_pc_loadings(strategy, pc_index=0):
    """
    Analyze and visualize PC loadings for a specific principal component
    """
    if strategy.V is None:
        print("PCA not computed yet")
        return

    loadings = strategy.V[:, pc_index]
    stock_names = strategy.stocks

    # Create DataFrame for easier analysis
    loadings_df = pd.DataFrame({
        'Stock': stock_names,
        'Loading': loadings,
        'Abs_Loading': np.abs(loadings)
    }).sort_values('Abs_Loading', ascending=False)

    # Plot loadings
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(loadings)), loadings)
    plt.title(f'PC_{pc_index + 1} Loadings by Stock')
    plt.xlabel('Stock Index')
    plt.ylabel('Loading')
    plt.xticks(range(len(stock_names)), stock_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return loadings_df


def analyze_factor_importance(strategy):
    """
    Analyze which factors are most important across all PCs
    """
    if not hasattr(strategy, 'pc_regressions') or not strategy.pc_regressions:
        print("Factor regressions not available")
        return

    factor_importance = {}

    print("\n=== Factor Importance Analysis ===")
    for pc_idx, regression_data in strategy.pc_regressions.items():
        print(f"\nPC_{pc_idx + 1} (R² = {regression_data['r_squared']:.3f}):")

        for i, (factor_name, coef) in enumerate(zip(regression_data['factor_names'],
                                                    regression_data['beta'])):
            print(f"  {factor_name}: β = {coef:.4f}")

            if factor_name not in factor_importance:
                factor_importance[factor_name] = []
            factor_importance[factor_name].append(abs(coef))

    # Overall factor importance (mean absolute coefficient)
    overall_importance = {factor: np.mean(coeffs)
                          for factor, coeffs in factor_importance.items()}

    print(f"\n=== Overall Factor Importance (Mean |β|) ===")
    for factor, importance in sorted(overall_importance.items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"{factor}: {importance:.4f}")

    return factor_importance, overall_importance
#
# def setup_logging():
#     """Set up logging for debugging."""
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Execute the full strategy example with rolling window
    strategy = run_strategy_example()

    # Additional analysis (only if we have recent data)
    if hasattr(strategy, 'V') and strategy.V is not None:
        print("\n" + "=" * 50)
        print("ADDITIONAL ANALYSIS")
        print("=" * 50)

        # Analyze PC loadings for the most recent window
        analyze_pc_loadings(strategy, pc_index=0)

        # Analyze factor importance for the most recent window
        analyze_factor_importance(strategy)

    print("\n=== Rolling Window Strategy Implementation Complete ===")
    print("All components now use a rolling 252-day window:")
    print("✅ Full dataset loaded with extended historical data")
    print("✅ Each rebalancing recalculates PCA matrix V")
    print("✅ Each rebalancing recalculates covariance matrix Σ")
    print("✅ Each rebalancing recalculates centrality matrix C")
    print("✅ Each rebalancing recalculates factor-PC regressions")
    print("✅ Each rebalancing uses fresh 252-day window")
    print("✅ Weekly progression: +5 trading days forward, -5 trading days back")
    print("✅ All matrix definitions and calculations preserved exactly")