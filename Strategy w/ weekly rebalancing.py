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

warnings.filterwarnings('ignore')


class UltimateDetailedStrategy:


    def __init__(self):
        # 1. Universe & Capital
        self.stocks = None
        self.C_0 = 10000
        self.rebalancing_period = 4
        self.L = 252
        self.transaction_costs = 0.001
        self.previous_prices = None

        # Data storage
        self.P_full = None
        self.factors_data_full = None

        # Current window data
        self.P = None
        self.R = None
        self.Sigma = None
        self.V = None
        self.PC_day = None
        self.factors_data = None
        self.centrality_matrix = None
        self.u_centrality = None
        self.previous_weights_backtest = None
        self.current_positions = None  # Initialize as None

        # Results storage
        self.portfolio_history = []
        self.performance_metrics = {}

        # Optimization counters
        self.slsqp_success_count = 0
        self.lsq_fallback_count = 0

        self.previous_weights = None

    ###########################################################################
    # DATA LOADING AND PREPARATION
    ###########################################################################

    def load_full_data(self, stock_symbols, start_date, end_date, factors_dict=None):
        """Load full dataset from 252 days before start_date until end_date"""
        print("Loading full stock and factor data...")
        self.stocks = stock_symbols
        self.current_positions = np.zeros(len(self.stocks))

        # Calculate actual start date
        start_date_dt = pd.to_datetime(start_date)
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

        self.backtest_start_date = start_date

    def set_current_window(self, current_date):
        """Set the current 252-day lookback window ending at current_date"""
        current_date = pd.to_datetime(current_date)
        available_data = self.P_full[self.P_full.index <= current_date]

        if len(available_data) >= self.L:
            self.P = available_data.tail(self.L)
        else:
            self.P = available_data

        if self.factors_data_full is not None:
            available_factor_data = self.factors_data_full[self.factors_data_full.index <= current_date]
            if len(available_factor_data) >= self.L:
                self.factors_data = available_factor_data.tail(self.L)
            else:
                self.factors_data = available_factor_data
        else:
            self.factors_data = None

    def prepare_returns_matrix(self, standardize=False):
        """Construct returns matrix R of shape (T, N) for current window"""
        self.R = np.log(self.P / self.P.shift(1)).dropna()

        if standardize:
            scaler = StandardScaler()
            self.R = pd.DataFrame(scaler.fit_transform(self.R),
                                  index=self.R.index, columns=self.R.columns)

        return self.R

    ###########################################################################
    # CORE STRATEGY COMPONENTS
    ###########################################################################

    def compute_covariance_and_pca(self, n_components=None):
        """Covariance and PCA for current window"""
        T, N = self.R.shape
        self.Sigma = np.cov(self.R.T, ddof=1)

        eigenvalues, eigenvectors = linalg.eigh(self.Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if n_components is None:
            n_components = min(N, 10)

        self.V = eigenvectors[:, :n_components]
        self.eigenvalues = eigenvalues[:n_components]
        self.PC_day = self.R.values @ self.V

        return self.V, self.PC_day

    def factor_pc_regression(self, min_r_squared=0.15, max_correlation=0.9, max_vif=5):
        """Factor-PC Regression for current window"""
        if self.factors_data is None:
            return None

        common_dates = self.R.index.intersection(self.factors_data.index)
        factors_aligned = self.factors_data.loc[common_dates]
        pc_aligned = pd.DataFrame(self.PC_day, index=self.R.index).loc[common_dates]

        delta_factors = np.log(factors_aligned / factors_aligned.shift(1)).dropna()
        pc_aligned = pc_aligned.loc[delta_factors.index]

        self.pc_regressions = {}

        for k in range(self.V.shape[1]):
            y = pc_aligned.iloc[:, k].values
            X = delta_factors.values

            selected_factors = []
            factor_r2 = []

            for i, factor_name in enumerate(delta_factors.columns):
                X_single = X[:, i].reshape(-1, 1)
                X_single_const = sm.add_constant(X_single)
                reg_single = sm.OLS(y, X_single_const).fit()
                r2_single = reg_single.rsquared
                p_value = reg_single.pvalues[1]

                if r2_single >= min_r_squared:
                    selected_factors.append(i)
                    factor_r2.append((i, factor_name, r2_single, p_value))

            if not selected_factors:
                continue

            self.pc_regressions[k] = {
                'individual_r2': {name: {'r2': r2, 'pvalue': pval} for i, name, r2, pval in factor_r2}
            }

            X_selected = X[:, selected_factors]

            if X_selected.shape[1] > 1:
                corr_matrix = np.corrcoef(X_selected.T)
                high_corr_pairs = np.where((np.abs(corr_matrix) > max_correlation) &
                                           (np.abs(corr_matrix) < 1.0))

                to_remove = set()
                for loc_i, loc_j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                    if loc_i < loc_j:
                        r2_i = factor_r2[loc_i][2]
                        r2_j = factor_r2[loc_j][2]
                        if r2_i < r2_j:
                            to_remove.add(loc_i)
                        else:
                            to_remove.add(loc_j)

                selected_factors = [f for idx, f in enumerate(selected_factors) if idx not in to_remove]
                X_selected = X[:, selected_factors]

            if X_selected.shape[1] > 1:
                try:
                    vif_scores = [variance_inflation_factor(X_selected, i)
                                  for i in range(X_selected.shape[1])]
                    high_vif = [i for i, vif in enumerate(vif_scores) if vif > max_vif]

                    selected_factors = [selected_factors[i] for i in range(len(selected_factors))
                                        if i not in high_vif]
                    X_selected = X[:, selected_factors]
                except:
                    pass

            if X_selected.shape[1] == 0:
                continue

            X_selected_const = sm.add_constant(X_selected)
            reg = sm.OLS(y, X_selected_const).fit()
            alpha = reg.params[0]
            beta = reg.params[1:]
            r_squared = reg.rsquared
            y_hat = reg.fittedvalues
            correlation = np.corrcoef(y, y_hat)[0, 1]

            if X_selected.shape[1] > 1:
                vif_scores = [variance_inflation_factor(X_selected, i) for i in range(X_selected.shape[1])]
            else:
                vif_scores = [1.0]

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
        """Historic PC Standard Deviation for 5-day horizon"""
        if lookback_window is None:
            lookback_window = min(self.L, len(self.PC_day))
        PC_df = pd.DataFrame(self.PC_day, index=self.R.index)
        self.sigma_PC = PC_df.std(ddof=1).values * np.sqrt(5)  # Scale to 5-day horizon
        return self.sigma_PC

    def calculate_current_factor_changes(self, lookback_days=4):
        """Calculate actual factor changes from current window factor data"""
        if self.factors_data is None:
            print("No factor data available")
            return {}

        recent_data = self.factors_data.tail(lookback_days + 1)

        if len(recent_data) < 2:
            print("Insufficient factor data for change calculation")
            return {}

        current_prices = recent_data.iloc[-1]
        previous_prices = recent_data.iloc[-(lookback_days + 1)]

        factor_changes = {}
        for factor in self.factors_data.columns:
            if not np.isnan(current_prices[factor]) and not np.isnan(previous_prices[factor]):
                if previous_prices[factor] > 0:
                    change = np.log(current_prices[factor] / previous_prices[factor])
                    factor_changes[factor] = change

        return factor_changes

    def calculate_stock_betas(self, benchmark_symbol='XLE', lookback_days=252):
        """Calculate stock betas relative to sector benchmark"""
        if self.factors_data is None or benchmark_symbol not in self.factors_data.columns:
            return np.ones(len(self.stocks))

        common_dates = self.R.index.intersection(self.factors_data.index)
        if len(common_dates) < lookback_days // 2:
            return np.ones(len(self.stocks))

        benchmark_prices = self.factors_data.loc[common_dates, benchmark_symbol]
        benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        aligned_stock_returns = self.R.loc[benchmark_returns.index]

        betas = np.zeros(len(self.stocks))
        benchmark_var = np.var(benchmark_returns, ddof=1)

        if benchmark_var <= 0:
            return np.ones(len(self.stocks))

        for i, stock in enumerate(self.stocks):
            if stock in aligned_stock_returns.columns:
                stock_returns = aligned_stock_returns[stock]
                covariance = np.cov(stock_returns, benchmark_returns, ddof=1)[0, 1]
                betas[i] = covariance / benchmark_var
            else:
                betas[i] = 1.0

        return betas

    def predict_pc_movement(self, current_factor_changes):
        if not hasattr(self, 'pc_regressions') or not self.pc_regressions:
            return np.zeros(self.V.shape[1])

        self.delta_PC_pred = np.zeros(self.V.shape[1])

        for k, regression_data in self.pc_regressions.items():
            alpha = regression_data['alpha']
            beta = regression_data['beta']
            selected_factors = regression_data['selected_factors']

            current_changes = np.array([current_factor_changes.get(self.factors_data.columns[i], 0)
                                        for i in selected_factors])

            self.delta_PC_pred[k] = alpha + np.dot(beta, current_changes)  # No * 5

        self.predicted_pct_change_PC = self.delta_PC_pred * self.sigma_PC
        return self.delta_PC_pred

    def predict_stock_returns(self):
        """Predicted Stock Returns scaled to five-day period"""
        # Check if delta_PC_pred exists and has valid data
        if not hasattr(self, 'delta_PC_pred') or self.delta_PC_pred is None:
            print("Warning: delta_PC_pred is None, using zero predictions")
            self.r_hat = np.zeros(self.V.shape[0])
            return self.r_hat

        # Check if sigma_PC exists
        if not hasattr(self, 'sigma_PC') or self.sigma_PC is None:
            print("Warning: sigma_PC is None, using delta_PC_pred directly")
            pc_weighted = self.delta_PC_pred
        else:
            pc_weighted = self.delta_PC_pred * self.sigma_PC

        self.r_hat = self.V @ pc_weighted
        return self.r_hat

    def compute_centrality_weighting(self, method='correlation'):
        """Fixed centrality weighting - handles None values properly"""
        import numpy as np
        from numpy import linalg

        # Check if r_hat exists and is not None
        if not hasattr(self, 'r_hat') or self.r_hat is None:
            print("Warning: r_hat is None, cannot compute centrality weighting")
            # Create default centrality weights (all ones)
            N = len(self.stocks)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = np.zeros(N)  # Return zeros if no predictions
            return self.r_hat_weighted

        # Check if r_hat has valid data
        if np.all(np.isnan(self.r_hat)) or np.all(self.r_hat == 0):
            print("Warning: r_hat contains only NaN or zeros")
            N = len(self.r_hat)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = self.r_hat.copy()  # Just pass through
            return self.r_hat_weighted

        if method == 'correlation':
            self.centrality_matrix = np.corrcoef(self.R.T)
        else:
            self.centrality_matrix = self.Sigma

        # Check for valid centrality matrix
        if np.any(np.isnan(self.centrality_matrix)) or np.any(np.isinf(self.centrality_matrix)):
            print("Warning: Centrality matrix contains NaN/Inf values")
            N = len(self.r_hat)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = self.r_hat.copy()
            return self.r_hat_weighted

        try:
            eigenvalues, eigenvectors = linalg.eigh(self.centrality_matrix)
            max_eigenvalue_idx = np.argmax(eigenvalues)
            self.u_centrality = np.abs(eigenvectors[:, max_eigenvalue_idx])
        except Exception as e:
            print(f"Warning: Eigenvalue decomposition failed: {e}")
            N = len(self.r_hat)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = self.r_hat.copy()
            return self.r_hat_weighted

        # Ensure non-negative
        self.u_centrality = np.maximum(self.u_centrality, 0)

        # Normalize to mean=1 and std=0.13 (very small deviation)
        current_mean = np.mean(self.u_centrality)
        current_std = np.std(self.u_centrality, ddof=1)

        if current_mean > 0 and current_std > 0:  # Avoid division by zero
            # Center to mean=1
            self.u_centrality = self.u_centrality - current_mean + 1
            # Scale to std=0.13 (very conservative)
            self.u_centrality = self.u_centrality / current_std * 0.13
            # Adjust mean back to 1
            final_mean = np.mean(self.u_centrality)
            self.u_centrality = self.u_centrality + (1 - final_mean)

            # Compress range to approximately [0.7, 1.3]
            # Desired range [0.7, 1.3] has width 0.6, centered at 1
            # Scale deviations from mean to fit within ±0.3
            deviations = self.u_centrality - 1
            current_range = np.max(np.abs(deviations))
            if current_range > 0:
                # Scale deviations to have max deviation of 0.3 (for range [0.7, 1.3])
                scale_factor = 0.3 / current_range
                self.u_centrality = 1 + deviations * scale_factor
                # Re-normalize to std=0.13
                current_std = np.std(self.u_centrality, ddof=1)
                if current_std > 0:
                    self.u_centrality = 1 + (self.u_centrality - 1) / current_std * 0.13
                # Ensure mean=1
                final_mean = np.mean(self.u_centrality)
                self.u_centrality = self.u_centrality + (1 - final_mean)
        else:
            # Fallback: if we can't normalize properly, just set to all 1s
            self.u_centrality = np.ones_like(self.u_centrality)

        print(f"u_centrality mean: {np.mean(self.u_centrality):.4f}")
        print(f"u_centrality std: {np.std(self.u_centrality, ddof=1):.4f}")
        print(f"u_centrality range: [{np.min(self.u_centrality):.4f}, {np.max(self.u_centrality):.4f}]")

        # Safe multiplication with proper error handling
        try:
            # Ensure both arrays are valid and same length
            if len(self.r_hat) != len(self.u_centrality):
                print(f"Warning: Length mismatch - r_hat: {len(self.r_hat)}, u_centrality: {len(self.u_centrality)}")
                # Resize u_centrality to match r_hat
                if len(self.r_hat) > len(self.u_centrality):
                    # Extend u_centrality with ones
                    extension = np.ones(len(self.r_hat) - len(self.u_centrality))
                    self.u_centrality = np.concatenate([self.u_centrality, extension])
                else:
                    # Truncate u_centrality
                    self.u_centrality = self.u_centrality[:len(self.r_hat)]

            # Perform the multiplication
            self.r_hat_weighted = self.r_hat * self.u_centrality

            # Check result
            if np.any(np.isnan(self.r_hat_weighted)) or np.any(np.isinf(self.r_hat_weighted)):
                print("Warning: r_hat_weighted contains NaN/Inf values after multiplication")
                # Fallback to unweighted
                self.r_hat_weighted = self.r_hat.copy()

            return self.r_hat_weighted

        except Exception as e:
            print(f"Error in centrality weighting multiplication: {e}")
            print(f"r_hat type: {type(self.r_hat)}, shape: {getattr(self.r_hat, 'shape', 'N/A')}")
            print(f"u_centrality type: {type(self.u_centrality)}, shape: {getattr(self.u_centrality, 'shape', 'N/A')}")

            # Fallback: return unweighted predictions
            self.r_hat_weighted = self.r_hat.copy()
            return self.r_hat_weighted
    ###########################################################################
    # PORTFOLIO OPTIMIZATION
    ###########################################################################
    def optimize_portfolio(self, current_prices, stock_betas=None):
        """Optimized portfolio allocation using updated portfolio value (self.C_0)"""
        N = len(self.r_hat_weighted)
        C_total = self.C_0  # Use current portfolio value

        if len(current_prices) != N or np.any(np.isnan(current_prices)) or np.any(np.isinf(current_prices)) or np.any(
                current_prices <= 0):
            return np.zeros(N)
        if np.any(np.isnan(self.r_hat_weighted)) or np.any(np.isinf(self.r_hat_weighted)):
            return np.zeros(N)

        r_hat_scaled = self.r_hat_weighted * 100

        def objective(v):
            return -np.dot(v, r_hat_scaled)

        constraints = []

        # Market neutral constraint (net exposure = 0)
        def market_neutral_constraint(v):
            return np.sum(v)  # Should equal 0

        constraints.append({'type': 'eq', 'fun': market_neutral_constraint})

        # Total capital utilization constraint (exactly C_total)
        def total_allocation_constraint(v):
            return np.sum(np.abs(v)) - C_total  # Should equal 0

        constraints.append({'type': 'eq', 'fun': total_allocation_constraint})

        # Long allocation constraints (65% target)
        def long_allocation_lower(v):
            long_positions = np.sum(v[v > 0])
            return long_positions - 0.60 * C_total  # At least 60%

        def long_allocation_upper(v):
            return 0.70 * C_total - np.sum(v[v > 0])  # At most 70%

        constraints.append({'type': 'ineq', 'fun': long_allocation_lower})
        constraints.append({'type': 'ineq', 'fun': long_allocation_upper})

        # Short allocation constraints (35% target)
        def short_allocation_lower(v):
            short_positions = np.abs(np.sum(v[v < 0]))
            return short_positions - 0.30 * C_total  # At least 30%

        def short_allocation_upper(v):
            return 0.40 * C_total - np.abs(np.sum(v[v < 0]))  # At most 40%

        constraints.append({'type': 'ineq', 'fun': short_allocation_lower})
        constraints.append({'type': 'ineq', 'fun': short_allocation_upper})

        # Portfolio beta constraint (if betas provided)
        if stock_betas is not None:
            def portfolio_beta_constraint(v):
                portfolio_beta = np.abs(np.dot(v, stock_betas) / C_total)
                return 0.15 - portfolio_beta  # More relaxed beta constraint

            constraints.append({'type': 'ineq', 'fun': portfolio_beta_constraint})

        # Individual position bounds (more relaxed)
        bounds = []
        for i in range(N):
            bounds.append((-0.20 * C_total, 0.20 * C_total))  # Max 20% in any position

        # Improved initial guess that ensures market neutrality with 65/35 split
        v0 = np.zeros(N)

        # Sort stocks by expected returns
        sorted_indices = np.argsort(r_hat_scaled)[::-1]  # Descending order

        # Allocate top half to long positions, bottom half to short
        n_long = len(sorted_indices) // 2
        n_short = len(sorted_indices) - n_long

        long_indices = sorted_indices[:n_long]
        short_indices = sorted_indices[n_long:]

        # Initial long allocation (target 65% of capital)
        if len(long_indices) > 0:
            v0[long_indices] = 0.65 * C_total / len(long_indices)

        # Initial short allocation (target -35% of capital)
        if len(short_indices) > 0:
            v0[short_indices] = -0.35 * C_total / len(short_indices)

        # Adjust for beta neutrality if needed
        if stock_betas is not None:
            portfolio_beta = np.dot(v0, stock_betas) / C_total
            if abs(portfolio_beta) > 0.1:
                beta_adjustment = -portfolio_beta / np.mean(stock_betas ** 2) if np.mean(stock_betas ** 2) > 0 else 0
                v0 += beta_adjustment * stock_betas * C_total / N

        try:
            result = opt.minimize(objective, v0, method='SLSQP', bounds=bounds,
                                  constraints=constraints,
                                  options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False})

            if result.success and np.sum(np.abs(result.x)) > 0.8 * C_total:
                self.optimal_weights = result.x
                self.slsqp_success_count += 1
            else:
                self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, C_total)
                self.lsq_fallback_count += 1
        except Exception:
            self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, C_total)
            self.lsq_fallback_count += 1

        self.share_quantities = self.optimal_weights / current_prices
        return self.optimal_weights

    def analyze_strategy_attribution(self):
        """Analyze where P&L is coming from"""
        long_pnl = 0
        short_pnl = 0
        cost_drag = 0

        for trade in self.portfolio_history:
            if 'actual_returns' in trade:
                weights = trade['weights']
                returns = trade['actual_returns']

                long_mask = weights > 0
                short_mask = weights < 0

                long_pnl += np.sum(weights[long_mask] * returns[long_mask])
                short_pnl += np.sum(weights[short_mask] * returns[short_mask])
                cost_drag += trade['transaction_cost']

        print(f"Long contribution: ${long_pnl:.2f}")
        print(f"Short contribution: ${short_pnl:.2f}")
        print(f"Cost drag: ${cost_drag:.2f}")
        print(f"Net P&L: ${long_pnl + short_pnl - cost_drag:.2f}")

    def calculate_forward_returns(self, current_date, forward_days=4):
        """Calculate actual returns for the forward period (next 5 trading days)"""
        future_dates = self.P_full[self.P_full.index > current_date].index

        if len(future_dates) < forward_days:
            print(
                f"Warning: Insufficient future data for {current_date}. Need {forward_days} days, have {len(future_dates)}")
            return None, None

        try:
            # Select the date that corresponds to 5 trading days (4 weekdays + 1 rebalancing day)
            next_rebalance_date = future_dates[forward_days - 1]
            current_prices = self.P_full.loc[current_date]
            future_prices = self.P_full.loc[next_rebalance_date]
            # Calculate five-day returns
            forward_returns = (future_prices / current_prices - 1).values
            return forward_returns, next_rebalance_date

        except KeyError as e:
            print(f"KeyError in calculate_forward_returns for {current_date}: {e}")
            print(f"Available dates in P_full: {self.P_full.index[-5:]}")
            return None, None
        except Exception as e:
            print(f"Error in calculate_forward_returns for {current_date}: {e}")
            return None, None

    def _least_squares_fallback(self, r_hat_scaled, bounds, stock_betas, C_total):
        """Enhanced least squares fallback that ensures market neutrality with 65/35 split"""
        N = len(r_hat_scaled)

        # Sort by expected returns
        sorted_indices = np.argsort(r_hat_scaled)[::-1]

        # Create market neutral target
        target = np.zeros(N)

        # Split into long and short based on expected returns
        n_long = N // 2
        long_indices = sorted_indices[:n_long]
        short_indices = sorted_indices[n_long:]

        # Weight by expected returns within each group
        if len(long_indices) > 0:
            long_returns = r_hat_scaled[long_indices]
            long_returns_positive = np.maximum(long_returns, 0.001)  # Ensure positive
            long_weights = long_returns_positive / np.sum(long_returns_positive)
            target[long_indices] = long_weights * 0.65 * C_total

        if len(short_indices) > 0:
            short_returns = r_hat_scaled[short_indices]
            short_returns_negative = np.minimum(short_returns, -0.001)  # Ensure negative
            short_weights = np.abs(short_returns_negative) / np.sum(np.abs(short_returns_negative))
            target[short_indices] = -short_weights * 0.35 * C_total

        # Ensure exact market neutrality
        net_exposure = np.sum(target)
        if abs(net_exposure) > 1e-6:
            # Adjust all positions proportionally to achieve neutrality
            adjustment = -net_exposure / N
            target += adjustment

        # Ensure total capital utilization
        current_total = np.sum(np.abs(target))
        if current_total > 0:
            target *= C_total / current_total

        # Apply beta adjustment if needed
        if stock_betas is not None:
            target_beta = np.dot(target, stock_betas) / C_total
            if abs(target_beta) > 0.15:
                # Beta adjustment while maintaining market neutrality
                beta_adj = -target_beta / np.sum(stock_betas ** 2) * stock_betas * C_total
                beta_adj_neutral = beta_adj - np.mean(beta_adj)  # Remove mean to maintain neutrality
                target += beta_adj_neutral

        # Apply bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        target = np.clip(target, lb, ub)

        # Final adjustment to maintain capital and neutrality constraints
        net_after_bounds = np.sum(target)
        total_after_bounds = np.sum(np.abs(target))

        if abs(net_after_bounds) > 1e-6:
            target -= net_after_bounds / N

        if total_after_bounds > 0 and abs(total_after_bounds - C_total) > 1e-6:
            target *= C_total / total_after_bounds

        return target

    def validate_optimization_results(self):
        """Enhanced validation that checks all constraints for 65/35 allocation"""
        print(f"\n=== OPTIMIZATION VALIDATION ===")

        print(f"Capital: ${self.C_0:,.0f}")

        long_exposure = np.sum(self.optimal_weights[self.optimal_weights > 0])
        short_exposure = np.abs(np.sum(self.optimal_weights[self.optimal_weights < 0]))
        net_exposure = np.sum(self.optimal_weights)
        total_exposure = np.sum(np.abs(self.optimal_weights))

        print(f"Total long exposure: ${long_exposure:,.0f}")
        print(f"Total short exposure: ${short_exposure:,.0f}")
        print(f"Net exposure: ${net_exposure:,.0f}")
        print(f"Capital utilization: {total_exposure / self.C_0:.1%}")
        print(f"Long exposure: {long_exposure / self.C_0:.1%}")
        print(f"Short exposure: {short_exposure / self.C_0:.1%}")
        print(f"Market neutrality: {net_exposure / self.C_0:.1%}")

        # Check if constraints are satisfied
        issues = []
        if abs(net_exposure) > 100:  # Allow $100 deviation
            issues.append(f"Market neutrality violated: ${net_exposure:.0f}")
        if abs(total_exposure - self.C_0) > 100:  # Allow $100 deviation
            issues.append(f"Capital utilization off: ${total_exposure:.0f} vs ${self.C_0}")
        if long_exposure < 0.60 * self.C_0:
            issues.append(f"Long exposure too low: {long_exposure / self.C_0:.1%}")
        if long_exposure > 0.70 * self.C_0:
            issues.append(f"Long exposure too high: {long_exposure / self.C_0:.1%}")
        if short_exposure < 0.30 * self.C_0:
            issues.append(f"Short exposure too low: {short_exposure / self.C_0:.1%}")
        if short_exposure > 0.40 * self.C_0:
            issues.append(f"Short exposure too high: {short_exposure / self.C_0:.1%}")

        if issues:
            print("⚠️ CONSTRAINT VIOLATIONS:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ All constraints satisfied")


    def plot_pc_regression_r2_over_time(self, n_pcs=5):
        """Plot R² of factor-PC regressions over time for the first n_pcs principal components"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        if not hasattr(self, 'portfolio_history') or not self.portfolio_history:
            print("No portfolio history available")
            return None

        # Initialize storage for R² values
        dates = [trade['timestamp'] for trade in self.portfolio_history]
        r2_by_pc = {f'PC_{i + 1}': [] for i in range(min(n_pcs, self.V.shape[1]))}

        # Iterate through each rebalancing period
        for date in dates:
            self.set_current_window(date)
            self.prepare_returns_matrix(standardize=False)
            self.compute_covariance_and_pca(n_components=n_pcs)
            regressions = self.factor_pc_regression(min_r_squared=0.1, max_correlation=0.9, max_vif=5)

            # Collect R² for each PC
            for i in range(min(n_pcs, self.V.shape[1])):
                if regressions and i in regressions:
                    r2_by_pc[f'PC_{i + 1}'].append(regressions[i]['r_squared'])
                else:
                    r2_by_pc[f'PC_{i + 1}'].append(np.nan)

        # Create DataFrame for plotting
        r2_df = pd.DataFrame(r2_by_pc, index=dates)

        # Plot
        plt.figure(figsize=(12, 6))
        for column in r2_df.columns:
            plt.plot(r2_df.index, r2_df[column], marker='o', label=column, alpha=0.7)

        plt.title(f'R² of Factor-PC Regressions Over Time (First {n_pcs} PCs)')
        plt.xlabel('Date')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return r2_df

    ###########################################################################
    # TRADE EXECUTION AND PERFORMANCE TRACKING
    ###########################################################################

    def execute_trades(self, current_prices):
        """Execute trades according to the computed v_i"""
        # Calculate turnover (only the changed portion)
        if self.previous_weights is not None:
            turnover = np.sum(np.abs(self.optimal_weights - self.previous_weights))
        else:
            # First trade: entire portfolio is turnover
            turnover = np.sum(np.abs(self.optimal_weights))

        transaction_cost = turnover * self.transaction_costs

        trade_record = {
            'timestamp': self.P.index[-1],
            'weights': self.optimal_weights.copy(),
            'shares': self.share_quantities.copy(),
            'prices': current_prices.copy(),
            'transaction_cost': transaction_cost,
            'expected_returns': self.r_hat_weighted.copy() if hasattr(self, 'r_hat_weighted') else None
        }

        self.portfolio_history.append(trade_record)

        # Store for next transaction cost calculation
        self.previous_weights = self.optimal_weights.copy()

        return trade_record

    def record_performance_metrics(self, actual_returns):
        """FIXED performance metrics recording with correct P&L calculation"""
        if actual_returns is None:
            return {'return': 0, 'gross_exposure': 0, 'net_exposure': 0}

        # Ensure we have current positions (dollar amounts, not shares)
        if not hasattr(self, 'optimal_weights') or self.optimal_weights is None:
            print("Warning: No optimal weights available")
            return {'return': 0, 'gross_exposure': 0, 'net_exposure': 0}

        try:
            # FIXED: P&L = weights * actual_returns (both are dollar amounts)
            period_pnl = np.sum(self.optimal_weights * actual_returns)

            # Calculate exposures using current weights
            gross_exposure = np.sum(np.abs(self.optimal_weights))
            net_exposure = np.sum(self.optimal_weights)

            # Store in the last portfolio history entry
            if len(self.portfolio_history) > 0:
                last_trade = self.portfolio_history[-1]
                last_trade['actual_returns'] = actual_returns.copy()
                last_trade['period_pnl'] = period_pnl

            return {
                'return': period_pnl,
                'realized_pnl': period_pnl,
                'gross_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'portfolio_value': self.C_0  # Don't update here, update in backtest
            }

        except Exception as e:
            print(f"Error in performance calculation: {e}")
            return {'return': 0, 'gross_exposure': 0, 'net_exposure': 0}

    ###########################################################################
    # BACKTESTING AND STRATEGY EXECUTION
    ###########################################################################
    def weekly_rebalancing_step(self, current_date, actual_returns=None):
        self.set_current_window(current_date)
        if len(self.P) < 50:
            self.current_positions = np.zeros(len(self.stocks))  # Set to zero if no trade
            return None, None

        self.prepare_returns_matrix(standardize=False)
        self.compute_covariance_and_pca(n_components=5)
        self.factor_pc_regression(min_r_squared=0.1, max_correlation=0.9, max_vif=5)
        self.compute_historic_pc_std()

        current_factor_changes = self.calculate_current_factor_changes(lookback_days=4)
        current_prices = self.P.iloc[-1].values
        stock_betas = self.calculate_stock_betas(benchmark_symbol='XLE', lookback_days=252)

        # FIXED ORDER: First predict PC movement and stock returns
        self.predict_pc_movement(current_factor_changes)
        self.predict_stock_returns()  # This sets self.r_hat

        # THEN apply centrality weighting (now r_hat exists)
        self.compute_centrality_weighting()

        self.optimize_portfolio(current_prices, stock_betas)
        self.validate_optimization_results()

        trade_record = self.execute_trades(current_prices)
        self.current_positions = self.share_quantities.copy()  # Update current_positions
        metrics = self.record_performance_metrics(actual_returns)

        return trade_record, metrics

    def validate_optimization_results(self):
        """Validate that optimization results make sense"""
        print(f"\n=== OPTIMIZATION VALIDATION ===")
        print(f"Date: {self.P.index[-1]}")
        print(f"Capital: ${self.C_0:,.0f}")
        print(f"Total long exposure: ${np.sum(self.optimal_weights[self.optimal_weights > 0]):,.0f}")
        print(f"Total short exposure: ${np.abs(np.sum(self.optimal_weights[self.optimal_weights < 0])):,.0f}")
        print(f"Net exposure: ${np.sum(self.optimal_weights):,.0f}")
        print(f"Capital utilization: {np.sum(np.abs(self.optimal_weights)) / self.C_0:.1%}")

        # Check constraints
        long_exposure = np.sum(self.optimal_weights[self.optimal_weights > 0])
        short_exposure = np.abs(np.sum(self.optimal_weights[self.optimal_weights < 0]))

        print(f"Long exposure (should be ~65%): {long_exposure / self.C_0:.1%}")
        print(f"Short exposure (should be ~35%): {short_exposure / self.C_0:.1%}")
        print(f"Market neutrality (should be ~0): {np.sum(self.optimal_weights) / self.C_0:.1%}")

    def backtest_strategy(self, start_date, end_date, rebalance_freq=4):
        """Backtest with proper portfolio value compounding"""
        print(f"\n=== Backtesting Strategy from {start_date} to {end_date} ===")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        backtest_dates = self.P_full[(self.P_full.index >= start_dt) &
                                     (self.P_full.index <= end_dt)].index

        if len(backtest_dates) == 0:
            print("No data available for backtesting period")
            return None

        # Initialize tracking variables
        initial_capital = 10000
        self.C_0 = initial_capital
        portfolio_values = [self.C_0]
        rebalance_dates = []
        self.previous_weights_backtest = None

        print(f"Starting backtest with ${initial_capital:,.0f}")

        for i in range(0, len(backtest_dates), rebalance_freq):
            current_date = backtest_dates[i]
            rebalance_dates.append(current_date)

            # Calculate forward returns for the NEXT period
            forward_returns, next_date = self.calculate_forward_returns(current_date, forward_days=rebalance_freq)

            try:
                # Execute rebalancing for current date
                trade_record, metrics = self.weekly_rebalancing_step(current_date, None)

                if trade_record is not None:
                    # Calculate transaction costs
                    if self.previous_weights_backtest is not None:
                        turnover = np.sum(np.abs(self.optimal_weights - self.previous_weights_backtest))
                    else:
                        turnover = np.sum(np.abs(self.optimal_weights))

                    transaction_cost = turnover * self.transaction_costs

                    # Update portfolio value with P&L and transaction costs
                    if forward_returns is not None:
                        period_pnl = np.sum(self.optimal_weights * forward_returns)
                        self.C_0 += period_pnl - transaction_cost

                        # Store the actual performance
                        trade_record['actual_returns'] = forward_returns.copy()
                        trade_record['period_pnl'] = period_pnl
                        trade_record['transaction_cost'] = transaction_cost

                        if i < 5:  # Debug first few periods
                            print(f"Period {len(portfolio_values)}: Prev value: ${portfolio_values[-1]:.2f}")
                            print(f"  Transaction cost: ${transaction_cost:.2f}")
                            print(f"  Period P&L: ${period_pnl:.2f}")
                            print(f"  New value: ${self.C_0:.2f}")
                            print("---")

                    portfolio_values.append(self.C_0)
                    self.previous_weights_backtest = self.optimal_weights.copy()

                else:
                    portfolio_values.append(portfolio_values[-1])

            except Exception as e:
                print(f"Error at {current_date}: {e}")
                portfolio_values.append(portfolio_values[-1])

        # Calculate performance metrics
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / initial_capital) - 1
            trading_days = (backtest_dates[-1] - backtest_dates[0]).days
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        else:
            total_return = 0
            annualized_return = 0

        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        print(f"Number of rebalancing periods: {len(rebalance_dates)}")

        # Store portfolio values for other methods
        self.portfolio_values_from_backtest = portfolio_values

        # Plot results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        # Ensure plot_dates and portfolio_values have the same length
        plot_dates = [self.P_full.index[0]] + rebalance_dates  # Include initial date
        if len(plot_dates) > len(portfolio_values):
            plot_dates = plot_dates[:len(portfolio_values)]
        elif len(portfolio_values) > len(plot_dates):
            portfolio_values = portfolio_values[:len(plot_dates)]
        plt.plot(plot_dates, portfolio_values, marker='o', label='Strategy Portfolio Value')
        plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital ($10,000)')
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

    ###########################################################################
    # PERFORMANCE ANALYSIS AND METRICS
    ###########################################################################
    def compute_profitability_metrics(self, risk_free_rate=0.02, periods_per_year=52):
        """Compute profitability metrics using compounded portfolio values"""
        if not self.portfolio_history:
            return {}

        # Use portfolio values from backtest
        if hasattr(self, 'portfolio_values_from_backtest'):
            portfolio_values = self.portfolio_values_from_backtest
        else:
            # Fallback to reconstructing portfolio values
            portfolio_values = [10000]
            current_value = 10000
            for trade in self.portfolio_history:
                if 'period_pnl' in trade and 'transaction_cost' in trade:
                    period_pnl = trade['period_pnl']
                    transaction_cost = trade['transaction_cost']
                    current_value += period_pnl - transaction_cost
                    portfolio_values.append(current_value)

        initial_capital = portfolio_values[0]
        final_portfolio_value = portfolio_values[-1]

        # Calculate period returns
        period_returns = []
        for i in range(1, len(portfolio_values)):
            period_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            period_returns.append(period_return)

        if not period_returns:
            return {}

        period_returns = np.array(period_returns)
        total_return = (final_portfolio_value - initial_capital) / initial_capital

        mean_return = np.mean(period_returns)
        std_return = np.std(period_returns, ddof=1)

        excess_return = mean_return - risk_free_rate / periods_per_year
        sharpe = excess_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0

        downside_returns = period_returns[period_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else std_return
        sortino = excess_return / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0

        portfolio_values_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values_array)
        drawdowns = (running_max - portfolio_values_array) / running_max
        max_drawdown = np.max(drawdowns)

        num_periods = len(period_returns)
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1 if num_periods > 0 else 0

        calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf

        winning_periods = [ret for ret in period_returns if ret > 0]
        losing_periods = [ret for ret in period_returns if ret < 0]

        avg_win = np.mean(winning_periods) if winning_periods else 0
        avg_loss = np.abs(np.mean(losing_periods)) if losing_periods else 0
        win_rate = len(winning_periods) / len(period_returns) if period_returns.size > 0 else 0
        profit_factor = sum(winning_periods) / sum(np.abs(losing_periods)) if losing_periods else np.inf

        total_transaction_costs = sum(trade.get('transaction_cost', 0) for trade in self.portfolio_history)

        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_periods': num_periods,
            'total_transaction_costs': total_transaction_costs,
            'final_portfolio_value': final_portfolio_value,
        }

        print(f"\n=== CORRECTED Profitability Metrics ===")
        print(f"Initial Capital: ${initial_capital:,.0f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.0f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Period Win Rate: {win_rate:.1%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.1%}")
        print(f"Total Transaction Costs: ${total_transaction_costs:,.0f}")

        return results

    def _calculate_position_win_rates(self):
        """Calculate win rates at the position level"""
        long_wins = []
        short_wins = []
        overall_wins = []

        for trade in self.portfolio_history:
            if 'actual_returns' in trade:
                weights = trade['weights']
                actual_returns = trade['actual_returns']
                position_pnls = weights * actual_returns

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

                if len(position_pnls) > 0:
                    overall_win_rate = np.sum(position_pnls > 0) / len(position_pnls)
                    overall_wins.append(overall_win_rate)

        return {
            'avg_long_position_win_rate': np.mean(long_wins) if long_wins else 0,
            'avg_short_position_win_rate': np.mean(short_wins) if short_wins else 0,
            'avg_overall_position_win_rate': np.mean(overall_wins) if overall_wins else 0,
        }

    ###########################################################################
    # VALIDATION AND DIAGNOSTIC FUNCTIONS
    ###########################################################################

    def stress_test_transaction_costs(self, cost_levels=[0.001, 0.003, 0.005, 0.01]):
        """Test strategy performance under different transaction cost assumptions"""
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
            total_pnl = 0
            total_costs = 0

            for trade in self.portfolio_history:
                ts = trade['timestamp']
                metrics = self.performance_metrics.get(ts, {})
                gross_pnl = metrics.get('realized_pnl', 0)

                total_trade_value = np.sum(np.abs(trade['weights']))
                new_transaction_cost = total_trade_value * cost_level

                net_pnl = gross_pnl - new_transaction_cost
                total_pnl += net_pnl
                total_costs += new_transaction_cost

            total_return = total_pnl / self.C_0
            num_periods = len(self.portfolio_history)
            annualized_return = (1 + total_return) ** (52 / num_periods) - 1 if num_periods > 0 else 0

            results[cost_level] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'total_costs': total_costs,
                'avg_cost_per_trade': total_costs / num_periods if num_periods > 0 else 0
            }

        print(f"\n{'Cost Level':<12} {'Total Ret':<12} {'Annual Ret':<12} {'Total Costs':<12} {'Avg/Trade':<12}")
        print("-" * 60)
        for cost, metrics in results.items():
            print(f"{cost:<8.1%} {metrics['total_return']:<8.1%} "
                  f"{metrics['annualized_return']:<8.1%} ${metrics['total_costs']:<11,.0f} "
                  f"${metrics['avg_cost_per_trade']:<11,.0f}")

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

    def validate_no_lookahead(self):
        """Check for data leakage in all trades"""
        print("\n" + "=" * 60)
        print("DATA LEAKAGE VALIDATION")
        print("=" * 60)

        issues_found = 0
        for i, trade in enumerate(self.portfolio_history):
            trade_date = trade['timestamp']
            self.set_current_window(trade_date)

            window_end = self.P.index[-1]
            if window_end > trade_date:
                # print(f"❌ Trade {i + 1} on {trade_date}: Window ends {window_end} (LOOKAHEAD BIAS)")
                issues_found += 1
            # else:
                # print(f"✅ Trade {i + 1} on {trade_date}: Window ends {window_end} (OK)")

        print(f"\nFound {issues_found} lookahead issues out of {len(self.portfolio_history)} trades")
        return issues_found == 0

    def monte_carlo_test(self, num_simulations=1000):
        """Fixed Monte Carlo test for statistical significance"""
        print("\n" + "=" * 60)
        print("MONTE CARLO SIGNIFICANCE TEST (FIXED)")
        print("=" * 60)

        if not self.portfolio_history or len(self.portfolio_history) < 2:
            print("No portfolio history available")
            return None

        # Extract portfolio values from backtest - FIXED APPROACH
        portfolio_values = [self.C_0]  # Start with initial capital

        # Build portfolio values by accumulating returns
        current_value = self.C_0
        actual_returns = []

        print(f"Building portfolio value series from {len(self.portfolio_history)} trades...")

        for i, trade in enumerate(self.portfolio_history):
            ts = trade['timestamp']

            # Get the realized P&L for this period
            if ts in self.performance_metrics:
                period_pnl = self.performance_metrics[ts].get('realized_pnl', 0)
                current_value += period_pnl
                portfolio_values.append(current_value)

                # Calculate period return
                if portfolio_values[-2] > 0:
                    period_return = period_pnl / portfolio_values[-2]
                    actual_returns.append(period_return)

                if i < 5:  # Debug first few periods
                    print(
                        f"  Period {i + 1}: PnL = ${period_pnl:.2f}, Value = ${current_value:.2f}, Return = {period_return:.4f}")

        print(f"Generated {len(actual_returns)} period returns from {len(portfolio_values)} portfolio values")

        if len(actual_returns) < 2:
            print("Insufficient return data for Monte Carlo analysis")
            return None

        # Remove any NaN or infinite returns
        clean_returns = []
        for ret in actual_returns:
            if np.isfinite(ret):
                clean_returns.append(ret)

        actual_returns = np.array(clean_returns)

        if len(actual_returns) == 0:
            print("No valid returns after cleaning")
            return None

        # Calculate actual strategy statistics
        actual_total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        actual_mean_return = np.mean(actual_returns)
        actual_std_return = np.std(actual_returns, ddof=1) if len(actual_returns) > 1 else 0.01

        print(f"\nActual Strategy Performance:")
        print(f"  Initial Capital: ${portfolio_values[0]:,.0f}")
        print(f"  Final Value: ${portfolio_values[-1]:,.0f}")
        print(f"  Total Return: {actual_total_return:.2%}")
        print(f"  Number of periods: {len(actual_returns)}")
        print(f"  Avg Period Return: {actual_mean_return:.4%}")
        print(f"  Period Return Std Dev: {actual_std_return:.4%}")

        # Ensure we have valid statistics
        if actual_std_return <= 0:
            print("Warning: Invalid standard deviation, using default")
            actual_std_return = 0.01  # 1% default std dev

        if not np.isfinite(actual_mean_return):
            print("Warning: Invalid mean return, using zero")
            actual_mean_return = 0.0

        # Monte Carlo simulation - FIXED
        random_final_values = []
        random_total_returns = []

        print(f"\nRunning {num_simulations} Monte Carlo simulations...")

        for sim in range(num_simulations):
            # Generate random returns with same statistics as actual strategy
            random_returns = np.random.normal(actual_mean_return, actual_std_return, len(actual_returns))

            # Simulate portfolio growth
            sim_portfolio_value = portfolio_values[0]

            for ret in random_returns:
                # Ensure return is reasonable (clip extreme values)
                ret = np.clip(ret, -0.5, 2.0)  # Cap losses at 50%, gains at 200%
                sim_portfolio_value *= (1 + ret)

            # Store results only if valid
            if np.isfinite(sim_portfolio_value) and sim_portfolio_value > 0:
                random_final_values.append(sim_portfolio_value)
                total_return = (sim_portfolio_value / portfolio_values[0]) - 1
                random_total_returns.append(total_return)

        # Convert to arrays and remove any remaining NaNs
        random_final_values = np.array(random_final_values)
        random_total_returns = np.array(random_total_returns)

        # Final cleaning
        valid_mask = np.isfinite(random_final_values) & np.isfinite(random_total_returns)
        random_final_values = random_final_values[valid_mask]
        random_total_returns = random_total_returns[valid_mask]

        if len(random_final_values) == 0:
            print("No valid simulation results generated")
            return None

        print(f"Generated {len(random_final_values)} valid simulations out of {num_simulations}")

        # Calculate p-value and statistics
        p_value = np.mean(random_final_values >= portfolio_values[-1])

        # Calculate confidence intervals
        if len(random_total_returns) > 10:
            ci_lower = np.percentile(random_total_returns, 2.5)
            ci_upper = np.percentile(random_total_returns, 97.5)
        else:
            ci_lower = ci_upper = 0

        print(f"\nMonte Carlo Results ({len(random_final_values)} valid simulations):")
        print(f"  Random strategy mean final value: ${np.mean(random_final_values):,.0f}")
        print(f"  Random strategy std final value: ${np.std(random_final_values):,.0f}")
        print(f"  Actual strategy final value: ${portfolio_values[-1]:,.0f}")
        print(f"  Probability of random strategy beating ours: {p_value:.3%}")
        print(f"  95% CI for random strategy total return: [{ci_lower:.2%}, {ci_upper:.2%}]")

        if p_value < 0.05:
            print(f"  ✅ Statistically significant (p < 0.05)")
        elif p_value < 0.10:
            print(f"  ⚠️  Marginally significant (p < 0.10)")
        else:
            print(f"  ❌ Not statistically significant")

        # Plot results - with error handling
        try:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(random_final_values, bins=min(50, len(random_final_values) // 10),
                     alpha=0.7, label='Random Strategies')
            plt.axvline(x=portfolio_values[-1], color='red', linewidth=3,
                        label=f'Actual Strategy (${portfolio_values[-1]:,.0f})')
            plt.xlabel('Final Portfolio Value ($)')
            plt.ylabel('Frequency')
            plt.title(f'Monte Carlo: Final Portfolio Values\np-value = {p_value:.3%}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.hist(random_total_returns, bins=min(50, len(random_total_returns) // 10),
                     alpha=0.7, label='Random Strategies')
            plt.axvline(x=actual_total_return, color='red', linewidth=3,
                        label=f'Actual Strategy ({actual_total_return:.1%})')
            if ci_lower != ci_upper:
                plt.axvline(x=ci_lower, color='gray', linestyle='--', alpha=0.7, label='95% CI')
                plt.axvline(x=ci_upper, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Total Return')
            plt.ylabel('Frequency')
            plt.title('Monte Carlo: Total Returns Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

        return {
            'actual_final_value': portfolio_values[-1],
            'actual_total_return': actual_total_return,
            'p_value': p_value,
            'random_final_values': random_final_values,
            'random_total_returns': random_total_returns,
            'confidence_interval': (ci_lower, ci_upper),
            'valid_simulations': len(random_final_values)
        }

    def walk_forward_test(self, train_start, train_end, test_start, test_end, steps=4):
        """Walk-forward validation across multiple periods"""
        print(f"\nWALK-FORWARD VALIDATION: {steps} steps")

        results = []

        total_days = (pd.to_datetime(test_end) - pd.to_datetime(train_start)).days
        step_days = total_days // steps

        for step in range(steps):
            step_train_start = pd.to_datetime(train_start) + pd.Timedelta(days=step * step_days)
            step_train_end = step_train_start + pd.Timedelta(days=step_days * 0.7)
            step_test_start = step_train_end
            step_test_end = step_train_start + pd.Timedelta(days=step_days)

            print(f"\nStep {step + 1}: Train {step_train_start.date()} to {step_train_end.date()}")
            print(f"          Test  {step_test_start.date()} to {step_test_end.date()}")

            step_strategy = UltimateDetailedStrategy()
            step_strategy.load_full_data(self.stocks,
                                         step_train_start.strftime('%Y-%m-%d'),
                                         step_test_end.strftime('%Y-%m-%d'),
                                         {s: s for s in self.factors_data_full.columns if
                                          s in self.factors_data_full.columns})

            step_results = step_strategy.backtest_strategy(
                step_test_start.strftime('%Y-%m-%d'),
                step_test_end.strftime('%Y-%m-%d'),
                rebalance_freq=4
            )

            if step_results:
                results.append(step_results['total_return'])
                print(f"  Return: {step_results['total_return']:.2%}")

        return results

    ###########################################################################
    # ANALYSIS AND VISUALIZATION FUNCTIONS
    ###########################################################################

    def analyze_pc_regressions(self, n_pcs=None):
        """Analyze PC regressions in detail"""
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

        unique_factors = set()
        for data in self.pc_regressions.values():
            unique_factors.update(data['factor_names'])
        unique_factors = sorted(unique_factors)

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
        """Portfolio Metrics & Graphs"""
        if not self.portfolio_history:
            print("No portfolio history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        timestamps = [trade['timestamp'] for trade in self.portfolio_history]
        long_exposure = [self.performance_metrics[ts]['long_exposure']
                         for ts in timestamps if ts in self.performance_metrics]
        short_exposure = [self.performance_metrics[ts]['short_exposure']
                          for ts in timestamps if ts in self.performance_metrics]

        if long_exposure and short_exposure:
            axes[0, 0].plot(timestamps[:len(long_exposure)], long_exposure, label='Long (65%)', color='green')
            axes[0, 0].plot(timestamps[:len(short_exposure)], short_exposure, label='Short (35%)', color='red')
            axes[0, 0].set_title('Long/Short Exposure Over Time')
            axes[0, 0].set_ylabel('Dollar Exposure')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        portfolio_values = [np.sum(np.abs(trade['weights'])) for trade in self.portfolio_history]
        axes[0, 1].plot(timestamps, portfolio_values, marker='o')
        axes[0, 1].set_title('Portfolio Value Over Time')
        axes[0, 1].set_ylabel('Total Portfolio Value ($)')
        axes[0, 1].grid(True)

        transaction_costs = [trade['transaction_cost'] for trade in self.portfolio_history]
        axes[1, 0].bar(range(len(transaction_costs)), transaction_costs)
        axes[1, 0].set_title('Transaction Costs per Rebalancing')
        axes[1, 0].set_ylabel('Transaction Cost ($)')
        axes[1, 0].set_xlabel('Rebalancing Period')
        axes[1, 0].grid(True)

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
        """Get optimization statistics"""
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
        """Display all variables and matrices as defined in the strategy"""
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

    ###########################################################################
    # ADVANCED VALIDATION AND ANALYSIS FUNCTIONS
    ###########################################################################

    def validate_data_integrity_and_lookahead(self):
        """Comprehensive validation to detect look-ahead bias and data leakage"""
        print("\n" + "=" * 70)
        print("DATA INTEGRITY AND LOOK-AHEAD BIAS VALIDATION")
        print("=" * 70)

        validation_results = {}
        issues_found = []

        print("\n1. FUTURE DATA LEAKAGE CHECK:")
        print("-" * 40)

        for i, trade in enumerate(self.portfolio_history[:5]):
            trade_date = trade['timestamp']
            print(f"\nTrade {i + 1} on {trade_date}:")

            self.set_current_window(trade_date)
            window_end = self.P.index[-1]
            window_start = self.P.index[0]

            if window_end > trade_date:
                issues_found.append(f"CRITICAL: Trade {i + 1} uses data after trade date!")
                print(f"  ❌ ISSUE: Window ends {window_end} but trade date is {trade_date}")
            else:
                print(f"  ✅ OK: Window ends {window_end}, trade date {trade_date}")

            print(f"     Window: {window_start} to {window_end} ({len(self.P)} days)")

            if self.factors_data is not None:
                factor_end = self.factors_data.index[-1]
                if factor_end > trade_date:
                    issues_found.append(f"CRITICAL: Factor data leakage in trade {i + 1}")
                    print(f"  ❌ ISSUE: Factor data ends {factor_end} but trade date is {trade_date}")
                else:
                    print(f"  ✅ OK: Factor data ends {factor_end}")

        print(f"\n2. ROLLING WINDOW INDEPENDENCE CHECK:")
        print("-" * 40)

        if len(self.portfolio_history) >= 3:
            dates = [trade['timestamp'] for trade in self.portfolio_history[:3]]

            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]

                self.set_current_window(current_date)
                window1_start = self.P.index[0]
                window1_end = self.P.index[-1]

                self.set_current_window(next_date)
                window2_start = self.P.index[0]
                window2_end = self.P.index[-1]

                expected_days_diff = (next_date - current_date).days
                actual_start_diff = (window2_start - window1_start).days
                actual_end_diff = (window2_end - window1_end).days

                print(f"\nWindow progression {i + 1} to {i + 2}:")
                print(f"  Trade dates: {current_date} → {next_date} ({expected_days_diff} days)")
                print(f"  Window starts: {window1_start} → {window2_start} ({actual_start_diff} days)")
                print(f"  Window ends: {window1_end} → {window2_end} ({actual_end_diff} days)")

                if abs(actual_end_diff - expected_days_diff) > 2:
                    issues_found.append(f"Window progression issue between trades {i + 1} and {i + 2}")
                    print(f"  ⚠️  Warning: Unexpected window progression")
                else:
                    print(f"  ✅ OK: Windows progress as expected")

        print(f"\n3. FACTOR SELECTION CONSISTENCY CHECK:")
        print("-" * 40)

        factor_usage = {}
        if len(self.portfolio_history) >= 5:
            sample_dates = [trade['timestamp'] for trade in self.portfolio_history[::10]]

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
        """Detailed analysis of prediction accuracy vs actual returns"""
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

                if len(pred) > 1 and len(actual) > 1:
                    correlation = np.corrcoef(pred, actual)[0, 1]
                    if not np.isnan(correlation):
                        correlation_by_period.append(correlation)
                        prediction_dates.append(date)

                        predicted_returns.extend(pred)
                        actual_returns.extend(actual)

                        if i < 10:
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

        overall_corr = np.corrcoef(predicted_returns, actual_returns)[
            0, 1] if predicted_returns and actual_returns else np.nan
        print(f"  Overall correlation (all data): {overall_corr:.3f}")

        if len(predicted_returns) > 100:
            sample_size = min(10000, len(predicted_returns))
            sample_indices = np.random.choice(len(predicted_returns), sample_size, replace=False)

            pred_sample = np.array(predicted_returns)[sample_indices]
            actual_sample = np.array(actual_returns)[sample_indices]

            if np.mean(np.abs(pred_sample)) > 1:
                pred_sample = pred_sample
                print("Rescaled predicted returns by dividing by 100")

            plt.figure(figsize=(8, 6))
            plt.scatter(pred_sample, actual_sample, alpha=0.5, s=20)

            min_val = min(np.min(pred_sample), np.min(actual_sample))
            max_val = max(np.max(pred_sample), np.max(actual_sample))
            margin = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
            line_range = [min_val - margin, max_val + margin]
            plt.plot(line_range, line_range, 'r--', alpha=0.7, label='Perfect Prediction (y=x)')

            coeffs = np.polyfit(pred_sample, actual_sample, 1)
            best_fit_line = np.poly1d(coeffs)
            plt.plot(line_range, best_fit_line(line_range), 'b-', alpha=0.7, label='Best Fit')

            fitted_predictions = best_fit_line(pred_sample)
            ss_tot = np.sum((actual_sample - np.mean(actual_sample)) ** 2)
            ss_res = np.sum((actual_sample - fitted_predictions) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

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
        """Detailed visualization of portfolio composition changes"""
        print("\n" + "=" * 70)
        print("PORTFOLIO COMPOSITION ANALYSIS")
        print("=" * 70)

        if not self.portfolio_history:
            print("No portfolio history available")
            return

        dates = [trade['timestamp'] for trade in self.portfolio_history]
        weights_history = np.array([trade['weights'] for trade in self.portfolio_history])
        stock_names = self.stocks

        print(f"\n1. PORTFOLIO STATISTICS:")
        print("-" * 30)
        print(f"  Number of rebalancing periods: {len(dates)}")
        print(f"  Number of stocks: {len(stock_names)}")
        print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

        long_counts = np.sum(weights_history > 0, axis=1)
        short_counts = np.sum(weights_history < 0, axis=1)

        print(f"  Avg long positions: {np.mean(long_counts):.1f} ± {np.std(long_counts):.1f}")
        print(f"  Avg short positions: {np.mean(short_counts):.1f} ± {np.std(short_counts):.1f}")

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

        axes[0, 0].plot(dates, long_exposure, label='Long', color='green', alpha=0.8)
        axes[0, 0].plot(dates, short_exposure, label='Short', color='red', alpha=0.8)
        axes[0, 0].axhline(y=0.65 * self.C_0, color='green', linestyle='--', alpha=0.5, label='Target Long (65%)')
        axes[0, 0].axhline(y=0.35 * self.C_0, color='red', linestyle='--', alpha=0.5, label='Target Short (35%)')
        axes[0, 0].set_title('Long/Short Exposure Over Time')
        axes[0, 0].set_ylabel('Dollar Exposure')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(dates, net_exposure, color='blue', alpha=0.8)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Net Exposure (Market Neutrality)')
        axes[0, 1].set_ylabel('Net Dollar Exposure')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(dates, long_counts, label='Long Positions', color='green', marker='o', markersize=3)
        axes[1, 0].plot(dates, short_counts, label='Short Positions', color='red', marker='o', markersize=3)
        axes[1, 0].set_title('Number of Positions Over Time')
        axes[1, 0].set_ylabel('Number of Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

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

        sample_stocks = stock_names[:10]
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

        turnover_rates = []
        for i in range(1, len(weights_history)):
            prev_weights = weights_history[i - 1]
            curr_weights = weights_history[i]
            turnover = np.sum(np.abs(curr_weights - prev_weights)) / (2 * self.C_0)
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
        """Analyze potential overfitting by examining strategy behavior patterns"""
        print("\n" + "=" * 70)
        print("OVERFITTING DETECTION ANALYSIS")
        print("=" * 70)

        overfitting_signals = []

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

                if win_rate_std < 0.05:
                    overfitting_signals.append("Win rate too consistent (low variability)")

        print(f"\n2. PERIOD-SPECIFIC PERFORMANCE:")

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

        year_returns = [sum(pnls) / self.C_0 for pnls in yearly_performance.values() if pnls]
        if year_returns and len(year_returns) > 1:
            max_year_return = max(year_returns)
            if max_year_return > 2 * np.mean(year_returns):
                overfitting_signals.append("Performance highly concentrated in specific period")

        print(f"\n3. MARKET CONDITION SENSITIVITY:")

        if self.factors_data_full is not None and '^VIX' in self.factors_data_full.columns:
            vix_data = self.factors_data_full['^VIX'].dropna()

            if len(vix_data) > 100:
                vix_median = vix_data.median()
                high_vol_performance = []
                low_vol_performance = []

                for trade in self.portfolio_history:
                    trade_date = trade['timestamp']
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

                    if abs(high_vol_return - low_vol_return) > 0.5:
                        overfitting_signals.append("Large performance difference across volatility regimes")

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


###############################################################################
# UTILITY FUNCTIONS FOR ANALYSIS AND VALIDATION
###############################################################################

def run_enhanced_validation_example(strategy):
    """Run comprehensive validation for the given strategy object"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION ANALYSIS")
    print("=" * 80)

    validation_results = strategy.validate_data_integrity_and_lookahead()
    prediction_analysis = strategy.analyze_prediction_accuracy()
    composition_analysis = strategy.analyze_portfolio_composition_over_time()
    cost_analysis = strategy.stress_test_transaction_costs()
    overfitting_signals = strategy.detect_overfitting_signals()

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
        realistic_return = cost_analysis[0.005]['annualized_return']
        print(f"   Current costs (0.1%): {base_return:.1%} annual return")
        print(f"   Realistic costs (0.5%): {realistic_return:.1%} annual return")
        print(f"   Impact: {(base_return - realistic_return) * 100:.1f} percentage points")

    return strategy

def plot_portfolio_composition(strategy):
    """Plot portfolio composition over time with different colors for each stock"""
    if not strategy.portfolio_history:
        print("No portfolio history available for composition plot")
        return

    # Extract data
    dates = [trade['timestamp'] for trade in strategy.portfolio_history]
    weights_history = np.array([trade['weights'] for trade in strategy.portfolio_history])
    stock_names = strategy.stocks

    # Convert to percentages of total portfolio value
    portfolio_values = np.sum(np.abs(weights_history), axis=1)
    portfolio_values = np.where(portfolio_values == 0, 1, portfolio_values)  # Avoid division by zero
    weights_pct = weights_history / portfolio_values[:, np.newaxis] * 100

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Separate long and short positions
    long_data = np.where(weights_pct > 0, weights_pct, 0)
    short_data = np.where(weights_pct < 0, np.abs(weights_pct), 0)

    # Plot long positions (positive)
    ax.stackplot(dates, long_data.T, labels=stock_names, alpha=0.8)

    # Plot short positions (negative)
    ax.stackplot(dates, -short_data.T, labels=[f"{name} (Short)" for name in stock_names], alpha=0.8)

    # Add zero line to separate longs and shorts
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)

    # Formatting
    ax.set_title('Portfolio Composition Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Allocation (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_strategy_example():
    """Complete example with validation - Single period 2018-2024"""
    print("=== Ultimate Detailed Strategy Example ===")
    print("Period: 2018-01-01 to 2024-01-01")

    strategy = UltimateDetailedStrategy()

    stock_symbols = [
        'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
        'COP', 'EOG', 'DVN', 'APA',
        'MPC', 'PSX', 'VLO', 'PBF', 'DK',
        'KMI', 'WMB', 'OKE', 'ET', 'ENB',
        'SLB', 'HAL', 'BKR', 'FTI', 'NOV',
        'FANG',  # Added as a Pioneer proxy
        'HES', 'CTRA'
    ]
    # factor_symbols = [
    #     'XLE', 'XOP', 'OIH', 'VDE', 'IXC',
    #     'CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F',
    #     'ICLN', 'TAN', 'FAN', 'PBW', 'QCLN',
    #     'CRAK', 'PXE', 'FCG', 'MLPX', 'AMLP',
    #     'FENY', 'OILK', 'USO', 'BNO', 'UNG',
    #     '^SP500-15', '^DJUSEN', '^XOI', '^OSX',
    #     'ENOR', 'ENZL', 'KWT', 'GEX', 'URA',
    #     'RSPG', '^TNX', '^VIX', 'COAL', 'URA',
    #     'XES', 'IEO', 'PXI', 'TIP', 'GLD'
    # ]
    factor_symbols = [
        'XLE', 'XOP', 'OIH', 'VDE',  # Energy ETFs
        'CL=F', 'NG=F',  # Commodities (Crude, Nat Gas)
        'USO', 'UNG',  # Commodity ETFs
        '^VIX', '^TNX',  # Volatility, Interest Rates
        'GLD', 'SLV',  # Precious Metals
        'SPY', 'QQQ',  # Broad Market
        'DBC',  # Commodity Basket
        'UUP', 'FXE',  # Currency ETFs
    ]

    # Load data for the full period 2018-2024
    strategy.load_full_data(stock_symbols, '2022-05-01', '2024-01-01',
                            {symbol: symbol for symbol in factor_symbols})

    # Run backtest for the full period
    backtest_results = strategy.backtest_strategy('2022-05-01', '2023-12-31', rebalance_freq=4)

    r2_df = strategy.plot_pc_regression_r2_over_time(n_pcs=5)
    print(r2_df.head())
    # Calculate and print performance metrics only
    performance = strategy.compute_profitability_metrics()

    # Run validation tests but don't print extra output
    strategy.validate_no_lookahead()

    # Run analysis functions to generate plots (only once)
    strategy.monte_carlo_test(num_simulations=1000)
    run_returns_analysis(strategy)

    # Add the portfolio composition plot
    plot_portfolio_composition(strategy)

    return strategy


def analyze_pc_loadings(strategy, pc_index=0):
    """Analyze and visualize PC loadings for a specific principal component"""
    if strategy.V is None:
        print("PCA not computed yet")
        return

    loadings = strategy.V[:, pc_index]
    stock_names = strategy.stocks

    loadings_df = pd.DataFrame({
        'Stock': stock_names,
        'Loading': loadings,
        'Abs_Loading': np.abs(loadings)
    }).sort_values('Abs_Loading', ascending=False)

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
    """Analyze which factors are most important across all PCs"""
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

    overall_importance = {factor: np.mean(coeffs)
                          for factor, coeffs in factor_importance.items()}

    print(f"\n=== Overall Factor Importance (Mean |β|) ===")
    for factor, importance in sorted(overall_importance.items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"{factor}: {importance:.4f}")

    return factor_importance, overall_importance
def analyze_prediction_vs_actual_detailed(self):
    """Create detailed scatter plot of predicted vs actual returns with corrected 5-day scaling"""
    predicted_all = []
    actual_all = []
    stock_names = []
    dates = []

    print("Collecting prediction vs actual data (5-day returns)...")

    for i, trade in enumerate(self.portfolio_history):
        if 'actual_returns' in trade and 'expected_returns' in trade:
            pred_returns = trade['expected_returns']
            actual_returns = trade['actual_returns']

            # Verify scaling: actual_returns should already be 5-day returns from calculate_forward_returns
            # pred_returns should be r_hat_weighted, scaled to 5-day in predict_stock_returns
            for j, stock in enumerate(self.stocks):
                if j < len(pred_returns) and j < len(actual_returns):
                    predicted_all.append(pred_returns[j])
                    actual_all.append(actual_returns[j])
                    stock_names.append(stock)
                    dates.append(trade['timestamp'])

    if len(predicted_all) == 0:
        print("No prediction data available")
        return None

    predicted_all = np.array(predicted_all)
    actual_all = np.array(actual_all)

    # Debugging: Print statistics to verify 5-day scaling
    print("\n=== DEBUG: Scale Comparison (Before Outlier Removal) ===")
    pred_mean = np.mean(np.abs(predicted_all))
    actual_mean = np.mean(np.abs(actual_all))
    pred_std = np.std(predicted_all)
    actual_std = np.std(actual_all)
    pred_range = (np.min(predicted_all), np.max(predicted_all))
    actual_range = (np.min(actual_all), np.max(actual_all))
    scale_ratio = pred_mean / actual_mean if actual_mean != 0 else float('inf')

    print(f"Predicted Returns (5-day):")
    print(f"  Mean Absolute: {pred_mean:.6f}")
    print(f"  Std Dev: {pred_std:.6f}")
    print(f"  Range: [{pred_range[0]:.6f}, {pred_range[1]:.6f}]")
    print(f"Actual Returns (5-day):")
    print(f"  Mean Absolute: {actual_mean:.6f}")
    print(f"  Std Dev: {actual_std:.6f}")
    print(f"  Range: [{actual_range[0]:.6f}, {actual_range[1]:.6f}]")
    print(f"Scale Ratio (Pred Mean / Actual Mean): {scale_ratio:.3f}")
    if scale_ratio > 2 or scale_ratio < 0.5:
        print("⚠️ Warning: Predicted and actual returns may not be on the same scale")

    # Remove outliers
    percentile_5 = np.percentile(predicted_all, 5)
    percentile_95 = np.percentile(predicted_all, 95)
    actual_5 = np.percentile(actual_all, 5)
    actual_95 = np.percentile(actual_all, 95)

    mask = ((predicted_all >= percentile_5) & (predicted_all <= percentile_95) &
            (actual_all >= actual_5) & (actual_all <= actual_95))

    pred_clean = predicted_all[mask]
    actual_clean = actual_all[mask]

    # Debugging: Print statistics after outlier removal
    print("\n=== DEBUG: Scale Comparison (After Outlier Removal) ===")
    pred_clean_mean = np.mean(np.abs(pred_clean))
    actual_clean_mean = np.mean(np.abs(actual_clean))
    pred_clean_std = np.std(pred_clean)
    actual_clean_std = np.std(actual_clean)
    pred_clean_range = (np.min(pred_clean), np.max(pred_clean))
    actual_clean_range = (np.min(actual_clean), np.max(actual_clean))
    clean_scale_ratio = pred_clean_mean / actual_clean_mean if actual_clean_mean != 0 else float('inf')

    print(f"Predicted Returns (5-day, Clean):")
    print(f"  Mean Absolute: {pred_clean_mean:.6f}")
    print(f"  Std Dev: {pred_clean_std:.6f}")
    print(f"  Range: [{pred_clean_range[0]:.6f}, {pred_clean_range[1]:.6f}]")
    print(f"Actual Returns (5-day, Clean):")
    print(f"  Mean Absolute: {actual_clean_mean:.6f}")
    print(f"  Std Dev: {actual_clean_std:.6f}")
    print(f"  Range: [{actual_clean_range[0]:.6f}, {actual_clean_range[1]:.6f}]")
    print(f"Scale Ratio (Pred Mean / Actual Mean, Clean): {clean_scale_ratio:.3f}")
    if clean_scale_ratio > 2 or clean_scale_ratio < 0.5:
        print("⚠️ Warning: Cleaned predicted and actual returns may not be on the same scale")

    print(f"Using {len(pred_clean)} data points after outlier removal")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(pred_clean, actual_clean, alpha=0.3, s=20, c='blue')

    correlation = np.corrcoef(pred_clean, actual_clean)[0, 1]

    min_val = min(np.min(pred_clean), np.min(actual_clean))
    max_val = max(np.max(pred_clean), np.max(actual_clean))
    perfect_line_range = [min_val, max_val]
    ax1.plot(perfect_line_range, perfect_line_range, 'r--',
             linewidth=2, label='Perfect Prediction (y=x)', alpha=0.8)

    lr = LinearRegression()
    pred_clean_reshaped = pred_clean.reshape(-1, 1)
    lr.fit(pred_clean_reshaped, actual_clean)

    r_squared = lr.score(pred_clean_reshaped, actual_clean)
    slope = lr.coef_[0]
    intercept = lr.intercept_

    best_fit_y = lr.predict(np.array(perfect_line_range).reshape(-1, 1))
    ax1.plot(perfect_line_range, best_fit_y, 'g-',
             linewidth=2, label=f'Best Fit (slope={slope:.2f})', alpha=0.8)

    ax1.set_xlabel('Predicted Returns (5-day)')
    ax1.set_ylabel('Actual Returns (5-day)')
    ax1.set_title(f'Predicted vs Actual Returns (5-day)\nCorr: {correlation:.3f}, R²: {r_squared:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    stats_text = f'N = {len(pred_clean)}\nCorrelation = {correlation:.3f}\nR² = {r_squared:.3f}\nSlope = {slope:.3f}\nIntercept = {intercept:.6f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    errors = actual_clean - pred_clean
    ax2.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean Error: {np.mean(errors):.6f}')

    ax2.set_xlabel('Prediction Error (Actual - Predicted, 5-day)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors (5-day)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n=== PREDICTION ACCURACY STATISTICS ===")
    print(f"Total data points: {len(predicted_all)}")
    print(f"After outlier removal: {len(pred_clean)}")
    print(f"Correlation: {correlation:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Best fit slope: {slope:.4f} (should be ~1.0 for perfect prediction)")
    print(f"Best fit intercept: {intercept:.6f} (should be ~0.0 for unbiased prediction)")
    print(f"Mean prediction error: {np.mean(errors):.6f}")
    print(f"Std of prediction errors: {np.std(errors):.6f}")
    print(f"Mean absolute error: {np.mean(np.abs(errors)):.6f}")

    print(f"\n=== INTERPRETATION ===")
    if abs(correlation) < 0.05:
        print("⚠️ Very weak correlation - predictions are essentially random")
    elif abs(correlation) < 0.15:
        print("⚠️ Weak correlation - limited predictive power")
    elif abs(correlation) < 0.3:
        print("✓ Moderate correlation - some predictive signal")
    else:
        print("✓ Strong correlation - good predictive signal")

    if abs(slope - 1.0) > 0.5:
        print("⚠️ Slope far from 1.0 - predictions may be on wrong scale")
    if abs(intercept) > np.std(actual_clean) * 0.1:
        print("⚠️ Large intercept - predictions may be biased")

    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'mean_error': np.mean(errors),
        'n_points': len(pred_clean)
    }
def fix_return_calculations(strategy):
    """Fix the return calculation methodology in the strategy"""
    print("\n=== ANALYZING RETURN CALCULATION ISSUES ===")

    if not strategy.portfolio_history:
        print("No portfolio history available")
        return

    for i, trade in enumerate(strategy.portfolio_history[:3]):
        print(f"\nPeriod {i + 1} - {trade['timestamp']}:")

        if 'actual_returns' in trade:
            actual_rets = trade['actual_returns']
            weights = trade['weights']

            print(f"  Weights range: [{np.min(weights):.2f}, {np.max(weights):.2f}]")
            print(f"  Weights sum: {np.sum(weights):.2f} (should be ~0 for market neutral)")
            print(f"  |Weights| sum: {np.sum(np.abs(weights)):.2f} (total exposure)")
            print(f"  Actual returns range: [{np.min(actual_rets):.4f}, {np.max(actual_rets):.4f}]")
            print(f"  Actual returns mean: {np.mean(actual_rets):.6f}")

            calculated_pnl = np.dot(weights, actual_rets)

            ts = trade['timestamp']
            metrics = strategy.performance_metrics.get(ts, {})
            stored_pnl = metrics.get('gross_pnl', 0)

            print(f"  Calculated P&L: ${calculated_pnl:.2f}")
            print(f"  Stored P&L: ${stored_pnl:.2f}")
            print(f"  Difference: ${abs(calculated_pnl - stored_pnl):.2f}")

            if abs(calculated_pnl - stored_pnl) > 1:
                print("  ⚠️  Large discrepancy in P&L calculation!")


def run_returns_analysis(strategy):
    """Run the returns analysis on your strategy object"""
    print("Running prediction vs actual analysis...")

    fix_return_calculations(strategy)
    results = analyze_prediction_vs_actual_detailed(strategy)

    return results


if __name__ == "__main__":
    strategy = run_strategy_example()
    #
    # print("Analyzing IN-SAMPLE strategy:")
    # results_in_sample = run_returns_analysis(strategy)
    #
    # print("\nAnalyzing OUT-OF-SAMPLE strategy:")
    # results_out_sample = run_returns_analysis(strategy_oos)
    #
    # if hasattr(strategy, 'V') and strategy.V is not None:
    #     print("\n" + "=" * 50)
    #     print("ADDITIONAL ANALYSIS")
    #     print("=" * 50)
    #
    #     analyze_pc_loadings(strategy, pc_index=0)
    #     analyze_factor_importance(strategy)
    #
    # print("\n=== Rolling Window Strategy Implementation Complete ===")
    # print("All components now use a rolling 252-day window:")
    # print("✅ Full dataset loaded with extended historical data")
    # print("✅ Each rebalancing recalculates PCA matrix V")
    # print("✅ Each rebalancing recalculates covariance matrix Σ")
    # print("✅ Each rebalancing recalculates centrality matrix C")
    # print("✅ Each rebalancing recalculates factor-PC regressions")
    # print("✅ Each rebalancing uses fresh 252-day window")
    # print("✅ Weekly progression: +5 trading days forward, -5 trading days back")
    # print("✅ All matrix definitions and calculations preserved exactly")