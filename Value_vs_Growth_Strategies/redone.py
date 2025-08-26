import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.optimize as opt
from scipy import linalg
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore')

class UltimateDetailedStrategy:
    """
    A comprehensive long-short portfolio strategy using PCA, factor regression, and centrality weighting.
    Optimized for robust backtesting and accurate prediction vs. actual analysis.
    """
    def __init__(self, initial_capital=10000, rebalancing_period=4, lookback_days=252, transaction_cost_rate=0.001):
        """Initialize strategy parameters and data storage."""
        # Portfolio parameters
        self.C_0 = initial_capital
        self.rebalancing_period = rebalancing_period
        self.L = lookback_days
        self.transaction_costs = transaction_cost_rate

        # Data storage
        self.stocks = None
        self.P_full = None  # Full price data
        self.factors_data_full = None  # Full factor data
        self.P = None  # Current window prices
        self.R = None  # Current window returns
        self.Sigma = None  # Covariance matrix
        self.V = None  # PCA loadings
        self.PC_day = None  # PC time series
        self.sigma_PC = None  # PC standard deviations
        self.factors_data = None  # Current window factor data
        self.pc_regressions = None  # Factor-PC regressions
        self.delta_PC_pred = None  # Predicted PC movements
        self.r_hat = None  # Predicted stock returns
        self.centrality_matrix = None  # Correlation/covariance matrix
        self.u_centrality = None  # Centrality weights
        self.r_hat_weighted = None  # Weighted predicted returns
        self.optimal_weights = None  # Portfolio weights
        self.share_quantities = None  # Share quantities
        self.previous_weights = None  # For transaction cost calculation

        # Results storage
        self.portfolio_history = []
        self.performance_metrics = {}
        self.slsqp_success_count = 0
        self.lsq_fallback_count = 0

    # =========================================================================
    # DATA LOADING AND PREPARATION
    # =========================================================================
    def load_full_data(self, stock_symbols, start_date, end_date, factors_dict=None):
        """Load and validate stock and factor price data."""
        print("Loading full stock and factor data...")
        self.stocks = stock_symbols
        start_date_dt = pd.to_datetime(start_date)
        extended_start_date = start_date_dt - pd.Timedelta(days=360)
        stock_data = yf.download(stock_symbols, start=extended_start_date, end=end_date, auto_adjust=True)
        prices = stock_data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(stock_symbols[0])

        # Validate stock price data
        prices = prices.dropna(how='all')
        valid_stocks = []
        for stock in stock_symbols:
            if stock in prices.columns:
                stock_prices = prices[stock]
                if stock_prices.isna().sum() / len(stock_prices) > 0.5:
                    print(f"Warning: Dropping {stock} due to excessive missing data")
                    continue
                if (stock_prices <= 0).any():
                    print(f"Warning: Dropping {stock} due to zero or negative prices")
                    continue
                valid_stocks.append(stock)
            else:
                print(f"Warning: No data for {stock}")
        self.P_full = prices[valid_stocks]
        self.stocks = valid_stocks
        print(f"Loaded price data shape: {self.P_full.shape}, stocks: {valid_stocks}")

        # Load factor data
        if factors_dict:
            factor_symbols = list(factors_dict.keys())
            factor_data = yf.download(factor_symbols, start=extended_start_date, end=end_date, auto_adjust=True)
            factors_prices = factor_data['Close']
            if isinstance(factors_prices, pd.Series):
                factors_prices = factors_prices.to_frame(factor_symbols[0])
            factors_prices = factors_prices.dropna(how='all')
            valid_factors = []
            for factor in factor_symbols:
                if factor in factors_prices.columns:
                    factor_prices = factors_prices[factor]
                    if factor_prices.isna().sum() / len(factor_prices) > 0.5:
                        print(f"Warning: Dropping factor {factor} due to excessive missing data")
                        continue
                    if (factor_prices <= 0).any():
                        print(f"Warning: Dropping factor {factor} due to zero or negative prices")
                        continue
                    valid_factors.append(factor)
            self.factors_data_full = factors_prices[valid_factors]
            print(f"Loaded factor data shape: {self.factors_data_full.shape}")
        else:
            self.factors_data_full = None
            print("No factor data provided.")
        self.backtest_start_date = start_date

    def set_current_window(self, current_date):
        """Set the current lookback window ending at current_date."""
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
        """Construct returns matrix for current window."""
        if len(self.P) < 2:
            raise ValueError(f"Insufficient price data: only {len(self.P)} days available")
        returns = np.log(self.P / self.P.shift(1))
        valid_columns = [col for col in returns.columns if
                         not returns[col].isna().all() and not (returns[col] == np.inf).any() and
                         not (returns[col] == -np.inf).any()]
        self.R = returns[valid_columns].dropna()
        if self.R.empty:
            raise ValueError("No valid returns data after cleaning")
        if standardize:
            scaler = StandardScaler()
            self.R = pd.DataFrame(scaler.fit_transform(self.R), index=self.R.index, columns=self.R.columns)
        return self.R

    # =========================================================================
    # CORE STRATEGY COMPONENTS
    # =========================================================================
    def compute_covariance_and_pca(self, n_components=None):
        """Compute covariance matrix and PCA for current window."""
        T, N = self.R.shape
        if T < 2 or N < 1:
            raise ValueError(f"Invalid returns matrix shape: {self.R.shape}")
        self.Sigma = np.cov(self.R.T, ddof=1)
        if np.any(np.isnan(self.Sigma)) or np.any(np.isinf(self.Sigma)):
            valid_cols = ~np.any(np.isnan(self.R) | np.isinf(self.R), axis=0)
            if not np.any(valid_cols):
                raise ValueError("No valid stocks remain after removing NaN/Inf")
            self.R = self.R.iloc[:, valid_cols]
            self.stocks = [self.stocks[i] for i in range(len(self.stocks)) if valid_cols[i]]
            self.Sigma = np.cov(self.R.T, ddof=1)
        eigenvalues, eigenvectors = linalg.eigh(self.Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        n_components = min(N, 10) if n_components is None else n_components
        self.V = eigenvectors[:, :n_components]
        self.eigenvalues = eigenvalues[:n_components]
        self.PC_day = self.R.values @ self.V
        return self.V, self.PC_day

    def factor_pc_regression(self, min_r_squared=0.15, max_correlation=0.9, max_vif=5):
        """Perform factor-PC regression for current window."""
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
            self.pc_regressions[k] = {'individual_r2': {name: {'r2': r2, 'pvalue': pval} for i, name, r2, pval in factor_r2}}
            X_selected = X[:, selected_factors]
            if X_selected.shape[1] > 1:
                corr_matrix = np.corrcoef(X_selected.T)
                high_corr_pairs = np.where((np.abs(corr_matrix) > max_correlation) & (np.abs(corr_matrix) < 1.0))
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
                    vif_scores = [variance_inflation_factor(X_selected, i) for i in range(X_selected.shape[1])]
                    high_vif = [i for i, vif in enumerate(vif_scores) if vif > max_vif]
                    selected_factors = [selected_factors[i] for i in range(len(selected_factors)) if i not in high_vif]
                    X_selected = X[:, selected_factors]
                except:
                    pass
            if X_selected.shape[1] == 0:
                continue
            X_selected_const = sm.add_constant(X_selected)
            reg = sm.OLS(y, X_selected_const).fit()
            self.pc_regressions[k].update({
                'alpha': reg.params[0],
                'beta': reg.params[1:],
                'selected_factors': selected_factors,
                'factor_names': [delta_factors.columns[i] for i in selected_factors],
                'r_squared': reg.rsquared,
                'correlation': np.corrcoef(y, reg.fittedvalues)[0, 1],
                'model': reg,
                'vif': dict(zip([delta_factors.columns[i] for i in selected_factors],
                               [variance_inflation_factor(X_selected, i) for i in range(X_selected.shape[1])]))
            })
        return self.pc_regressions

    def compute_historic_pc_std(self, lookback_window=None):
        """Compute historic standard deviation of PCs."""
        lookback_window = min(self.L, len(self.PC_day)) if lookback_window is None else lookback_window
        PC_df = pd.DataFrame(self.PC_day, index=self.R.index)
        self.sigma_PC = PC_df.std(ddof=1).values
        return self.sigma_PC

    def calculate_current_factor_changes(self, lookback_days=4):
        """Calculate factor changes over the lookback period."""
        if self.factors_data is None:
            return {}
        recent_data = self.factors_data.tail(lookback_days + 1)
        if len(recent_data) < 2:
            return {}
        current_prices = recent_data.iloc[-1]
        previous_prices = recent_data.iloc[-(lookback_days + 1)]
        factor_changes = {}
        for factor in self.factors_data.columns:
            if not np.isnan(current_prices[factor]) and not np.isnan(previous_prices[factor]) and previous_prices[factor] > 0:
                factor_changes[factor] = np.log(current_prices[factor] / previous_prices[factor])
        return factor_changes

    def calculate_stock_betas(self, benchmark_symbol='XLE', lookback_days=252):
        """Calculate stock betas relative to benchmark."""
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
        """Predict PC movements based on factor changes."""
        if not hasattr(self, 'pc_regressions') or not self.pc_regressions:
            return np.zeros(self.V.shape[1])
        self.delta_PC_pred = np.zeros(self.V.shape[1])
        for k, regression_data in self.pc_regressions.items():
            alpha = regression_data['alpha']
            beta = regression_data['beta']
            selected_factors = regression_data['selected_factors']
            current_changes = np.array([current_factor_changes.get(self.factors_data.columns[i], 0)
                                       for i in selected_factors])
            self.delta_PC_pred[k] = alpha + np.dot(beta, current_changes)
        self.predicted_pct_change_PC = self.delta_PC_pred * self.sigma_PC
        return self.delta_PC_pred

    def predict_stock_returns(self):
        """Predict stock returns from PC movements."""
        self.r_hat = self.V @ (self.delta_PC_pred * self.sigma_PC)
        return self.r_hat

    def compute_centrality_weighting(self, method='correlation'):
        """Compute centrality-based weights for stock returns."""
        if not hasattr(self, 'r_hat') or self.r_hat is None or np.all(np.isnan(self.r_hat)) or np.all(self.r_hat == 0):
            N = len(self.stocks)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = np.zeros(N) if self.r_hat is None else self.r_hat.copy()
            return self.r_hat_weighted
        self.centrality_matrix = np.corrcoef(self.R.T) if method == 'correlation' else self.Sigma
        if np.any(np.isnan(self.centrality_matrix)) or np.any(np.isinf(self.centrality_matrix)):
            N = len(self.r_hat)
            self.u_centrality = np.ones(N)
            self.r_hat_weighted = self.r_hat.copy()
            return self.r_hat_weighted
        try:
            eigenvalues, eigenvectors = linalg.eigh(self.centrality_matrix)
            max_eigenvalue_idx = np.argmax(eigenvalues)
            self.u_centrality = np.abs(eigenvectors[:, max_eigenvalue_idx])
            self.u_centrality = np.maximum(self.u_centrality, 0)
            current_mean = np.mean(self.u_centrality)
            current_std = np.std(self.u_centrality, ddof=1)
            if current_mean > 0 and current_std > 0:
                self.u_centrality = self.u_centrality - current_mean + 1
                self.u_centrality = self.u_centrality / current_std * 0.13
                final_mean = np.mean(self.u_centrality)
                self.u_centrality = self.u_centrality + (1 - final_mean)
                deviations = self.u_centrality - 1
                current_range = np.max(np.abs(deviations))
                if current_range > 0:
                    scale_factor = 0.3 / current_range
                    self.u_centrality = 1 + deviations * scale_factor
                    current_std = np.std(self.u_centrality, ddof=1)
                    if current_std > 0:
                        self.u_centrality = 1 + (self.u_centrality - 1) / current_std * 0.13
                    final_mean = np.mean(self.u_centrality)
                    self.u_centrality = self.u_centrality + (1 - final_mean)
            else:
                self.u_centrality = np.ones_like(self.u_centrality)
            self.r_hat_weighted = self.r_hat * self.u_centrality
            if np.any(np.isnan(self.r_hat_weighted)) or np.any(np.isinf(self.r_hat_weighted)):
                self.r_hat_weighted = self.r_hat.copy()
        except Exception as e:
            print(f"Error in centrality weighting: {e}")
            self.u_centrality = np.ones(len(self.r_hat))
            self.r_hat_weighted = self.r_hat.copy()
        return self.r_hat_weighted

    # =========================================================================
    # PORTFOLIO OPTIMIZATION
    # =========================================================================
    def optimize_portfolio(self, current_prices, stock_betas=None):
        """Optimize portfolio weights for 65/35 long-short allocation."""
        N = len(self.r_hat_weighted)
        if len(current_prices) != N or np.any(np.isnan(current_prices)) or np.any(np.isinf(current_prices)) or np.any(current_prices <= 0):
            return np.zeros(N)
        if np.any(np.isnan(self.r_hat_weighted)) or np.any(np.isinf(self.r_hat_weighted)):
            return np.zeros(N)
        r_hat_scaled = self.r_hat_weighted * 100
        def objective(v):
            return -np.dot(v, r_hat_scaled)
        constraints = [
            {'type': 'eq', 'fun': lambda v: np.sum(np.abs(v)) - self.C_0},
            {'type': 'ineq', 'fun': lambda v: np.sum(v[v > 0]) - 0.60 * self.C_0},
            {'type': 'ineq', 'fun': lambda v: 0.70 * self.C_0 - np.sum(v[v > 0])},
            {'type': 'ineq', 'fun': lambda v: np.abs(np.sum(v[v < 0])) - 0.30 * self.C_0},
            {'type': 'ineq', 'fun': lambda v: 0.40 * self.C_0 - np.abs(np.sum(v[v < 0]))}
        ]
        if stock_betas is not None:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda v: np.dot(v, stock_betas) / self.C_0 + 0.15},
                {'type': 'ineq', 'fun': lambda v: 0.15 - np.dot(v, stock_betas) / self.C_0}
            ])
        bounds = [(-0.20 * self.C_0, 0.20 * self.C_0) for _ in range(N)]
        v0 = np.zeros(N)
        sorted_indices = np.argsort(r_hat_scaled)[::-1]
        n_long = N // 2
        long_indices = sorted_indices[:n_long]
        short_indices = sorted_indices[n_long:]
        if len(long_indices) > 0:
            v0[long_indices] = 0.65 * self.C_0 / len(long_indices)
        if len(short_indices) > 0:
            v0[short_indices] = -0.35 * self.C_0 / len(short_indices)
        if stock_betas is not None:
            portfolio_beta = np.dot(v0, stock_betas) / self.C_0
            if abs(portfolio_beta) > 0.1:
                beta_adjustment = -portfolio_beta / np.mean(stock_betas ** 2) if np.mean(stock_betas ** 2) > 0 else 0
                v0 += beta_adjustment * stock_betas * self.C_0 / N
        try:
            result = opt.minimize(objective, v0, method='SLSQP', bounds=bounds,
                                 constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-9})
            if result.success and np.sum(np.abs(result.x)) > 0.8 * self.C_0:
                self.optimal_weights = result.x
                self.slsqp_success_count += 1
            else:
                self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, self.C_0)
                self.lsq_fallback_count += 1
        except Exception:
            self.optimal_weights = self._least_squares_fallback(r_hat_scaled, bounds, stock_betas, self.C_0)
            self.lsq_fallback_count += 1
        self.share_quantities = self.optimal_weights / current_prices
        return self.optimal_weights

    def _least_squares_fallback(self, r_hat_scaled, bounds, stock_betas, C_total):
        """Fallback optimization using least squares for 65/35 split."""
        N = len(r_hat_scaled)
        target = np.zeros(N)
        sorted_indices = np.argsort(r_hat_scaled)[::-1]
        n_long = N // 2
        long_indices = sorted_indices[:n_long]
        short_indices = sorted_indices[n_long:]
        if len(long_indices) > 0:
            long_returns = r_hat_scaled[long_indices]
            long_returns_positive = np.maximum(long_returns, 0.001)
            long_weights = long_returns_positive / np.sum(long_returns_positive)
            target[long_indices] = long_weights * 0.65 * C_total
        if len(short_indices) > 0:
            short_returns = r_hat_scaled[short_indices]
            short_returns_negative = np.minimum(short_returns, -0.001)
            short_weights = np.abs(short_returns_negative) / np.sum(np.abs(short_returns_negative))
            target[short_indices] = -short_weights * 0.35 * C_total
        current_total = np.sum(np.abs(target))
        if current_total > 0:
            target *= C_total / current_total
        if stock_betas is not None:
            target_beta = np.dot(target, stock_betas) / C_total
            if abs(target_beta) > 0.15:
                beta_adj = -target_beta / np.sum(stock_betas ** 2) * stock_betas * C_total if np.sum(stock_betas ** 2) > 0 else 0
                target += beta_adj
                current_total = np.sum(np.abs(target))
                if current_total > 0:
                    target *= C_total / current_total
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        target = np.clip(target, lb, ub)
        total_after_bounds = np.sum(np.abs(target))
        if total_after_bounds > 0 and abs(total_after_bounds - C_total) > 1e-6:
            target *= C_total / total_after_bounds
        return target

    def calculate_forward_returns(self, current_date, forward_days=4):
        """Calculate actual arithmetic returns for the forward period."""
        current_date = pd.to_datetime(current_date)
        future_dates = self.P_full[self.P_full.index > current_date].index
        if len(future_dates) < forward_days:
            print(f"Warning: Insufficient future data for {current_date}. Need {forward_days} days, have {len(future_dates)}")
            return None, None
        try:
            next_rebalance_date = future_dates[forward_days - 1]
            current_prices = self.P_full.loc[current_date]
            future_prices = self.P_full.loc[next_rebalance_date]
            forward_returns = (future_prices / current_prices - 1)
            forward_returns = forward_returns.replace([np.inf, -np.inf], np.nan)
            if forward_returns.isna().all():
                print(f"Warning: No valid returns for {current_date} to {next_rebalance_date}")
                return None, None
            return forward_returns.values, next_rebalance_date
        except KeyError as e:
            print(f"KeyError in calculate_forward_returns for {current_date}: {e}")
            return None, None
        except Exception as e:
            print(f"Error in calculate_forward_returns for {current_date}: {e}")
            return None, None

    # =========================================================================
    # TRADE EXECUTION AND PERFORMANCE TRACKING
    # =========================================================================
    def execute_trades(self, current_prices):
        """Execute trades and calculate transaction costs."""
        turnover = np.sum(np.abs(self.optimal_weights - self.previous_weights)) if self.previous_weights is not None else np.sum(np.abs(self.optimal_weights))
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
        self.previous_weights = self.optimal_weights.copy()
        return trade_record

    def record_performance_metrics(self, actual_returns=None):
        """Record performance metrics for the current trade."""
        if not self.portfolio_history:
            return
        latest_trade = self.portfolio_history[-1]
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
        if actual_returns is not None:
            gross_pnl = np.dot(latest_trade['weights'], actual_returns)
            net_pnl = gross_pnl - latest_trade['transaction_cost']
            metrics.update({
                'gross_pnl': gross_pnl,
                'realized_pnl': net_pnl,
                'expected_pnl': np.dot(latest_trade['weights'], latest_trade['expected_returns'])
            })
            latest_trade['actual_returns'] = actual_returns.copy()
        self.performance_metrics[latest_trade['timestamp']] = metrics
        return metrics

    # =========================================================================
    # BACKTESTING
    # =========================================================================
    def weekly_rebalancing_step(self, current_date):
        """Perform a single rebalancing step."""
        current_date = pd.to_datetime(current_date)
        self.set_current_window(current_date)
        if len(self.P) < 50:
            return None, None
        self.prepare_returns_matrix(standardize=False)
        self.compute_covariance_and_pca(n_components=5)
        self.factor_pc_regression(min_r_squared=0.1, max_correlation=0.9, max_vif=5)
        self.compute_historic_pc_std()
        current_factor_changes = self.calculate_current_factor_changes(lookback_days=4)
        current_prices = self.P.iloc[-1].values
        stock_betas = self.calculate_stock_betas(benchmark_symbol='XLE', lookback_days=252)
        self.predict_pc_movement(current_factor_changes)
        self.predict_stock_returns()
        self.compute_centrality_weighting()
        self.optimize_portfolio(current_prices, stock_betas)
        trade_record = self.execute_trades(current_prices)
        return trade_record

    def backtest_strategy(self, start_date, end_date, rebalance_freq=4):
        """Backtest the strategy with robust validation."""
        print(f"\n=== Backtesting Strategy from {start_date} to {end_date} ===")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        backtest_dates = self.P_full[(self.P_full.index >= start_dt) & (self.P_full.index <= end_dt)].index
        if len(backtest_dates) == 0:
            print("No data available for backtesting period")
            return None
        portfolio_values = [self.C_0]
        rebalance_dates = []
        self.current_positions = None
        previous_prices = None
        self.previous_weights_backtest = None
        for i in range(0, len(backtest_dates), rebalance_freq):
            current_date = backtest_dates[i]
            rebalance_dates.append(current_date)
            current_prices = self.P_full.loc[current_date].values
            position_pnl = 0
            if self.current_positions is not None and previous_prices is not None:
                try:
                    price_returns = (current_prices / previous_prices) - 1
                    position_pnl = np.sum(self.current_positions * previous_prices * price_returns)
                    self.C_0 += position_pnl
                except Exception as e:
                    print(f"Error calculating P&L for {current_date}: {e}")
            actual_forward_returns, next_rebalance_date = self.calculate_forward_returns(current_date, rebalance_freq)
            if actual_forward_returns is None:
                portfolio_values.append(portfolio_values[-1])
                continue
            try:
                trade_record = self.weekly_rebalancing_step(current_date)
                if trade_record is not None:
                    turnover = np.sum(np.abs(self.optimal_weights - self.previous_weights_backtest)) if self.previous_weights_backtest is not None else np.sum(np.abs(self.optimal_weights))
                    transaction_cost = turnover * self.transaction_costs
                    self.previous_weights_backtest = self.optimal_weights.copy()
                    trade_record['transaction_cost'] = transaction_cost
                    new_portfolio_value = portfolio_values[-1] + position_pnl - transaction_cost
                    portfolio_values.append(new_portfolio_value)
                    self.current_positions = self.share_quantities.copy()
                    previous_prices = current_prices.copy()
                    self.record_performance_metrics(actual_forward_returns)
                else:
                    new_portfolio_value = portfolio_values[-1] + position_pnl
                    portfolio_values.append(new_portfolio_value)
                    if self.current_positions is None:
                        self.current_positions = np.zeros(len(self.stocks))
            except Exception as e:
                print(f"Error at {current_date}: {e}")
                portfolio_values.append(portfolio_values[-1] + position_pnl)
                if self.current_positions is None:
                    self.current_positions = np.zeros(len(self.stocks))
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        trading_days = (backtest_dates[-1] - backtest_dates[0]).days
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        plt.figure(figsize=(12, 6))
        plt.plot(rebalance_dates[:len(portfolio_values) - 1], portfolio_values[1:], marker='o', label='Portfolio Value')
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

    # =========================================================================
    # ANALYSIS AND VISUALIZATION
    # =========================================================================
    def analyze_prediction_accuracy(self):
        """Analyze prediction vs actual returns."""
        if len(self.portfolio_history) < 2:
            return {'avg_correlation': None, 'correlation_by_period': [], 'prediction_dates': [], 'r_squared': None}
        predicted_returns = []
        actual_returns = []
        prediction_dates = []
        correlation_by_period = []
        for trade in self.portfolio_history:
            if 'actual_returns' in trade and 'expected_returns' in trade:
                pred = trade['expected_returns']
                actual = trade['actual_returns']
                date = trade['timestamp']
                if len(pred) > 1 and len(actual) > 1:
                    valid_mask = ~(np.isnan(pred) | np.isnan(actual))
                    if np.sum(valid_mask) > 0:
                        correlation = np.corrcoef(pred[valid_mask], actual[valid_mask])[0, 1]
                        if not np.isnan(correlation):
                            correlation_by_period.append(correlation)
                            prediction_dates.append(date)
                            predicted_returns.extend(pred[valid_mask])
                            actual_returns.extend(actual[valid_mask])
        if not correlation_by_period:
            return {'avg_correlation': None, 'correlation_by_period': [], 'prediction_dates': [], 'r_squared': None}
        avg_correlation = np.mean(correlation_by_period)
        overall_corr = np.corrcoef(predicted_returns, actual_returns)[0, 1] if predicted_returns else np.nan
        sample_size = min(10000, len(predicted_returns))
        sample_indices = np.random.choice(len(predicted_returns), sample_size, replace=False) if sample_size > 0 else []
        pred_sample = np.array(predicted_returns)[sample_indices]
        actual_sample = np.array(actual_returns)[sample_indices]
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_sample, actual_sample, alpha=0.5, s=20)
        min_val = min(np.min(pred_sample), np.min(actual_sample)) if len(pred_sample) > 0 else 0
        max_val = max(np.max(pred_sample), np.max(actual_sample)) if len(pred_sample) > 0 else 0
        margin = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
        line_range = [min_val - margin, max_val + margin]
        plt.plot(line_range, line_range, 'r--', alpha=0.7, label='Perfect Prediction (y=x)')
        coeffs = np.polyfit(pred_sample, actual_sample, 1) if len(pred_sample) > 0 else [0, 0]
        best_fit_line = np.poly1d(coeffs)
        plt.plot(line_range, best_fit_line(line_range), 'b-', alpha=0.7, label='Best Fit')
        ss_tot = np.sum((actual_sample - np.mean(actual_sample)) ** 2) if len(actual_sample) > 0 else 0
        ss_res = np.sum((actual_sample - best_fit_line(pred_sample)) ** 2) if len(pred_sample) > 0 else 0
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        plt.text(line_range[1], line_range[0], f'R²: {r_squared:.3f}', fontsize=10, ha='right', va='bottom')
        plt.xlabel('Predicted Returns')
        plt.ylabel('Actual Returns')
        plt.title(f'Predicted vs Actual Returns (Sample of {sample_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 4))
        plt.plot(prediction_dates, correlation_by_period, marker='o', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_correlation, color='g', linestyle='-', alpha=0.7, label=f'Avg: {avg_correlation:.3f}')
        plt.title('Prediction Correlation Over Time')
        plt.ylabel('Correlation')
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

    def plot_portfolio_metrics(self):
        """Plot portfolio metrics over time."""
        if not self.portfolio_history:
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

    def analyze_portfolio_composition_over_time(self):
        """Visualize portfolio composition changes."""
        if not self.portfolio_history:
            return
        dates = [trade['timestamp'] for trade in self.portfolio_history]
        weights_history = np.array([trade['weights'] for trade in self.portfolio_history])
        stock_names = self.stocks
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        long_exposure = [np.sum(weights[weights > 0]) for weights in weights_history]
        short_exposure = [np.abs(np.sum(weights[weights < 0])) for weights in weights_history]
        net_exposure = [np.sum(weights) for weights in weights_history]
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
        axes[0, 1].set_title('Net Exposure')
        axes[0, 1].set_ylabel('Net Dollar Exposure')
        axes[0, 1].grid(True, alpha=0.3)
        long_counts = np.sum(weights_history > 0, axis=1)
        short_counts = np.sum(weights_history < 0, axis=1)
        axes[1, 0].plot(dates, long_counts, label='Long Positions', color='green', marker='o', markersize=3)
        axes[1, 0].plot(dates, short_counts, label='Short Positions', color='red', marker='o', markersize=3)
        axes[1, 0].set_title('Number of Positions Over Time')
        axes[1, 0].set_ylabel('Number of Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        concentration_ratios = [np.sum(np.sort(np.abs(weights))[-5:]) / np.sum(np.abs(weights))
                               if np.sum(np.abs(weights)) > 0 else 0 for weights in weights_history]
        axes[1, 1].plot(dates, concentration_ratios, color='purple', alpha=0.8)
        axes[1, 1].set_title('Portfolio Concentration (Top 5 Positions)')
        axes[1, 1].set_ylabel('Top 5 Weight Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        sample_stocks = stock_names[:10]
        sample_indices = list(range(min(10, len(stock_names))))
        sample_weights = weights_history[:, sample_indices].T
        im = axes[2, 0].imshow(sample_weights, cmap='RdBu_r', aspect='auto',
                              vmin=-np.max(np.abs(sample_weights)), vmax=np.max(np.abs(sample_weights)))
        axes[2, 0].set_title('Stock Weights Over Time (Sample)')
        axes[2, 0].set_xlabel('Rebalancing Period')
        axes[2, 0].set_ylabel('Stock')
        axes[2, 0].set_yticks(range(len(sample_stocks)))
        axes[2, 0].set_yticklabels(sample_stocks)
        plt.colorbar(im, ax=axes[2, 0], label='Weight ($)')
        turnover_rates = [np.sum(np.abs(weights_history[i] - weights_history[i-1])) / (2 * self.C_0)
                         for i in range(1, len(weights_history))]
        axes[2, 1].plot(dates[1:], turnover_rates, color='orange', alpha=0.8)
        axes[2, 1].set_title('Portfolio Turnover Rate')
        axes[2, 1].set_ylabel('Turnover Rate')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'turnover_rates': turnover_rates,
            'concentration_ratios': concentration_ratios
        }

    def calculate_avg_rebalancing_period(self):
        """Calculate average days between rebalancing events."""
        if len(self.portfolio_history) < 2:
            return None
        rebalance_dates = [trade['timestamp'] for trade in self.portfolio_history]
        rebalance_dates.sort()
        day_diffs = [(rebalance_dates[i] - rebalance_dates[i-1]).days for i in range(1, len(rebalance_dates))]
        avg_days = np.mean(day_diffs) if day_diffs else 0
        plt.figure(figsize=(10, 6))
        plt.hist(day_diffs, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(avg_days, color='red', linestyle='--', label=f'Average: {avg_days:.2f} days')
        plt.xlabel('Days Between Rebalancing')
        plt.ylabel('Frequency')
        plt.title('Distribution of Rebalancing Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return avg_days, day_diffs

    def compute_profitability_metrics(self, risk_free_rate=0.02, periods_per_year=52):
        """Compute profitability metrics for the strategy."""
        if not self.portfolio_history:
            return {}
        period_pnls = []
        gross_pnls = []
        transaction_costs = []
        portfolio_values = [self.C_0]
        for trade in self.portfolio_history:
            ts = trade['timestamp']
            metrics = self.performance_metrics.get(ts, {})
            net_pnl = metrics.get('realized_pnl', 0)
            gross_pnl = metrics.get('gross_pnl', 0)
            cost = trade['transaction_cost']
            period_pnls.append(net_pnl)
            gross_pnls.append(gross_pnl)
            transaction_costs.append(cost)
            portfolio_values.append(portfolio_values[-1] + net_pnl)
        total_return = (portfolio_values[-1] / self.C_0) - 1
        period_returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                         for i in range(1, len(portfolio_values))]
        period_returns = np.array(period_returns)
        mean_return = np.mean(period_returns)
        std_return = np.std(period_returns, ddof=1)
        excess_return = mean_return - risk_free_rate / periods_per_year
        sharpe = excess_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0
        downside_returns = period_returns[period_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else std_return
        sortino = excess_return / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0
        drawdowns = [(np.max(portfolio_values[:i+1]) - portfolio_values[i]) / np.max(portfolio_values[:i+1])
                     for i in range(1, len(portfolio_values))]
        max_drawdown = np.max(drawdowns) if drawdowns else 0
        num_periods = len(period_pnls)
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1 if num_periods > 0 else 0
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
        winning_periods = [pnl for pnl in period_pnls if pnl > 0]
        losing_periods = [pnl for pnl in period_pnls if pnl < 0]
        avg_win = np.mean(winning_periods) if winning_periods else 0
        avg_loss = np.abs(np.mean(losing_periods)) if losing_periods else 0
        win_rate = len(winning_periods) / len(period_pnls) if period_pnls else 0
        profit_factor = sum(winning_periods) / sum(np.abs(losing_periods)) if losing_periods else np.inf
        total_gross_pnl = sum(gross_pnls)
        total_transaction_costs = sum(transaction_costs)
        cost_drag = total_transaction_costs / abs(total_gross_pnl) if total_gross_pnl != 0 else 0
        return {
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
            'transaction_cost_drag': cost_drag,
            'final_portfolio_value': portfolio_values[-1]
        }

    def validate_no_lookahead(self):
        """Validate absence of look-ahead bias in backtesting."""
        issues_found = 0
        for i, trade in enumerate(self.portfolio_history):
            trade_date = trade['timestamp']
            self.set_current_window(trade_date)
            window_end = self.P.index[-1]
            if window_end > trade_date:
                issues_found += 1
        return issues_found == 0


if __name__ == "__main__":
    # Define sample stocks and factors
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'WMT', 'PG', 'XOM']
    factors_dict = {
        'XLE': 'Energy Sector ETF',
        'SPY': 'S&P 500 ETF',
        'TLT': 'Treasury Bond ETF',
        'GLD': 'Gold ETF'
    }

    # Initialize strategy
    strategy = UltimateDetailedStrategy(
        initial_capital=10000,
        rebalancing_period=4,
        lookback_days=252,
        transaction_cost_rate=0.001
    )

    # Load data
    start_date = '2020-01-01'
    end_date = '2025-08-22'
    strategy.load_full_data(stock_symbols, start_date, end_date, factors_dict)

    # Run backtest
    backtest_results = strategy.backtest_strategy(start_date, end_date, rebalance_freq=4)

    # Generate plots
    print("\nGenerating portfolio metrics plot...")
    strategy.plot_portfolio_metrics()

    print("\nGenerating portfolio composition analysis...")
    strategy.analyze_portfolio_composition_over_time()

    print("\nGenerating rebalancing period distribution...")
    avg_days, day_diffs = strategy.calculate_avg_rebalancing_period()
    if avg_days is not None:
        print(f"Average rebalancing period: {avg_days:.2f} days")

    print("\nGenerating prediction accuracy analysis...")
    prediction_results = strategy.analyze_prediction_accuracy()
    if prediction_results['avg_correlation'] is not None:
        print(f"Average prediction correlation: {prediction_results['avg_correlation']:.3f}")
        print(f"R-squared: {prediction_results['r_squared']:.3f}")

    # Compute and display profitability metrics
    print("\nComputing profitability metrics...")
    profitability = strategy.compute_profitability_metrics(risk_free_rate=0.02, periods_per_year=52)
    if profitability:
        print("\nProfitability Metrics:")
        print(f"Total Return: {profitability['total_return'] * 100:.2f}%")
        print(f"Annualized Return: {profitability['annualized_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {profitability['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {profitability['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {profitability['calmar_ratio']:.2f}")
        print(f"Max Drawdown: {profitability['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {profitability['win_rate'] * 100:.2f}%")
        print(f"Average Win: ${profitability['avg_win']:.2f}")
        print(f"Average Loss: ${profitability['avg_loss']:.2f}")
        print(f"Profit Factor: {profitability['profit_factor']:.2f}")
        print(f"Total Transaction Costs: ${profitability['total_transaction_costs']:.2f}")
        print(f"Transaction Cost Drag: {profitability['transaction_cost_drag'] * 100:.2f}%")
        print(f"Final Portfolio Value: ${profitability['final_portfolio_value']:.2f}")

    # Validate backtest
    print("\nValidating backtest for look-ahead bias...")
    is_valid = strategy.validate_no_lookahead()
    print(f"Backtest validation: {'No look-ahead bias detected' if is_valid else 'Look-ahead bias detected'}")