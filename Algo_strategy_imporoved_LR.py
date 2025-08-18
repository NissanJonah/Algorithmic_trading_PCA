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
import statsmodels.api as sm

warnings.filterwarnings('ignore')


class UltimateDetailedStrategy:
    def __init__(self):
        # 1. Universe & Capital
        self.stocks = None  # S = {s_1, s_2, ..., s_N} from sector or watchlist
        self.C_0 = 10000  # Starting capital: C_0 = 10,000 USD
        self.rebalancing_period = 5  # Weekly (5 trading days)
        self.L = 252  # Lookback period for statistics: L = 252 trading days
        self.transaction_costs = 0.001  # fee per share or percentage per trade

        # Data storage
        self.P = None  # Price matrix
        self.R = None  # Returns matrix R of shape (T, N)
        self.Sigma = None  # Covariance matrix Σ = (1 / (T - 1)) * R^T · R
        self.V = None  # Loadings matrix V = [v_1, v_2, ..., v_K] shape (N, K)
        self.PC_day = None  # PC time series matrix PC_day = R · V shape (T, K)
        self.factors_data = None  # Factor data
        self.centrality_matrix = None  # Centrality matrix C
        self.u_centrality = None  # First eigenvector of centrality matrix

        # Results storage
        self.portfolio_history = []
        self.performance_metrics = {}

    def load_data(self, stock_symbols, start_date, end_date, factors_dict=None):
        """
        2. Data Preparation
        Load stock price data and factor data
        """
        print("Loading stock data...")
        self.stocks = stock_symbols

        # Load stock price data
        stock_data = yf.download(stock_symbols, start=start_date, end=end_date, auto_adjust=True)
        prices = stock_data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(stock_symbols[0])
        self.P = prices.dropna()
        # print(f"Loaded price data shape: {self.P.shape}")

        # Load factor data
        if factors_dict:
            factor_symbols = list(factors_dict.keys())
            print(f"Attempting to load factor data for {len(factor_symbols)} symbols...")
            factor_data = yf.download(factor_symbols, start=start_date, end=end_date, auto_adjust=True)
            factors_prices = factor_data['Close']
            if isinstance(factors_prices, pd.Series):
                factors_prices = factors_prices.to_frame(factor_symbols[0])

            # Drop columns with all NaN values and check if data is empty
            factors_prices = factors_prices.dropna(axis=1, how='all')
            if factors_prices.empty:
                print("Warning: No valid factor data loaded. Skipping factor data.")
                self.factors_data = None
            else:
                self.factors_data = factors_prices.dropna()
                print(f"Loaded factor data shape: {self.factors_data.shape}")
        else:
            self.factors_data = None
            print("No factor data provided.")

    def prepare_returns_matrix(self, standardize=False):
        """
        2b. Log Returns Matrix
        Construct returns matrix R of shape (T, N)
        """
        # print("Computing returns matrix...")

        # R[t, s] = log(P_t_s / P_(t-1)_s)
        self.R = np.log(self.P / self.P.shift(1)).dropna()
        # print(f"Returns matrix R shape: {self.R.shape}")  # T = number of trading days

        # 2c. Optional: Standardization
        if standardize:
            # print("Standardizing returns...")
            # R_std[:, s] = (R[:, s] - mean(R[:, s])) / std(R[:, s])
            scaler = StandardScaler()
            self.R = pd.DataFrame(scaler.fit_transform(self.R),
                                  index=self.R.index, columns=self.R.columns)

        return self.R

    def compute_covariance_and_pca(self, n_components=None):
        """
        3. Covariance and PCA
        """
        # print("Computing covariance matrix and PCA...")

        # 3a. Covariance Matrix - Σ = (1 / (T - 1)) * R^T · R, Shape: (N, N)
        T, N = self.R.shape
        self.Sigma = np.cov(self.R.T, ddof=1)  # Shape: (N, N)
        # print(f"Covariance matrix Σ shape: {self.Sigma.shape}")

        # 3b. Eigen Decomposition - Σ · v_k = λ_k · v_k
        eigenvalues, eigenvectors = linalg.eigh(self.Sigma)
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # λ_k = variance explained by PC_k
        # v_k = eigenvector = stock loadings for PC_k
        if n_components is None:
            n_components = min(N, 10)  # Default to 10 components or N if smaller

        # Stack eigenvectors to form loadings matrix V
        # V = [v_1, v_2, ..., v_K] shape (N, K), Each column = PC loadings
        self.V = eigenvectors[:, :n_components]
        self.eigenvalues = eigenvalues[:n_components]

        # print(f"Loadings matrix V shape: {self.V.shape}")
        # print(f"Explained variance ratio: {self.eigenvalues / np.sum(eigenvalues)}")

        # 3c. PC Time Series Matrix
        # Project returns onto PCs: PC_day = R · V shape (T, K)
        self.PC_day = self.R.values @ self.V
        # print(f"PC time series matrix shape: {self.PC_day.shape}")
        # Row t = [PC_1_t, PC_2_t, ..., PC_K_t]
        # Each PC series captures common movement of stock groupings

        return self.V, self.PC_day


    # Updated factor_pc_regression function (more intricate with statsmodels for detailed stats, p-values, and final VIF)
    def factor_pc_regression(self, min_r_squared=0.15, max_correlation=0.9, max_vif=5):
        """
        4. Factor-PC Regression
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
        5. Historic PC Standard Deviation
        Over lookback L: σ_PC_k = sqrt( Σ_{i=t-L}^{t-1} (PC_k_i - mean(PC_k))^2 / (L - 1) )
        """
        if lookback_window is None:
            lookback_window = self.L

        # print("Computing historic PC standard deviations...")

        # Convert PC_day to DataFrame for easier rolling operations
        PC_df = pd.DataFrame(self.PC_day, index=self.R.index)

        # PC_std = PC_ts.rolling(window=L).std(ddof=1)
        self.PC_std_rolling = PC_df.rolling(window=lookback_window, min_periods=lookback_window // 2).std(ddof=1)

        # Get the most recent standard deviations
        self.sigma_PC = self.PC_std_rolling.iloc[-1].values  # σ_PC vector

        # print(f"Historic PC standard deviations shape: {self.sigma_PC.shape}")
        return self.sigma_PC

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
    # Add new method to UltimateDetailedStrategy class

    def calculate_current_factor_changes(self, lookback_days=5):
        """
        Calculate actual factor changes from downloaded factor data
        Uses the most recent lookback_days to compute ΔF_f_t = log(F_f_t / F_f_(t-lookback_days))
        """
        if self.factors_data is None:
            print("No factor data available")
            return {}

        # Get the most recent data
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
        Calculate stock betas relative to sector benchmark using:
        β_i = Cov(R_i, R_benchmark) / Var(R_benchmark)

        Where:
        - R_i = returns of stock i
        - R_benchmark = returns of benchmark (e.g., XLE for energy sector)
        - β_i = beta of stock i relative to benchmark
        """
        if self.factors_data is None or benchmark_symbol not in self.factors_data.columns:
            print(f"Benchmark {benchmark_symbol} not found in factor data, using default beta of 1.0")
            return np.ones(len(self.stocks))

        # Get benchmark data aligned with stock data
        common_dates = self.R.index.intersection(self.factors_data.index)
        if len(common_dates) < lookback_days // 2:
            print("Insufficient overlapping data for beta calculation, using default beta of 1.0")
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
            print("Zero variance in benchmark, using default beta of 1.0")
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

    def compute_profitability_metrics(self, risk_free_rate=0.02, periods_per_year=52):
        if not self.portfolio_history:
            return {}

        pnls = []
        transaction_costs = []
        long_win_rates = []
        short_win_rates = []
        overall_win_rates = []
        num_positions = []

        for trade in self.portfolio_history:
            ts = trade['timestamp']
            metrics = self.performance_metrics.get(ts, {})
            pnl = metrics.get('realized_pnl', 0)
            cost = trade['transaction_cost']
            net_pnl = pnl - cost
            pnls.append(net_pnl)
            transaction_costs.append(cost)

            # Calculate win rates properly
            if 'actual_returns' in trade:
                actual_returns = trade['actual_returns']
                weights = trade['weights']

                # Calculate P&L for each position
                pos_pnl = weights * actual_returns

                # Separate long and short positions
                long_mask = weights > 0
                short_mask = weights < 0

                # Long positions win rate
                if np.any(long_mask):
                    long_wins = np.sum(pos_pnl[long_mask] > 0)
                    long_total = np.sum(long_mask)
                    long_win_rate = long_wins / long_total
                    long_win_rates.append(long_win_rate)

                # Short positions win rate
                if np.any(short_mask):
                    short_wins = np.sum(pos_pnl[short_mask] > 0)
                    short_total = np.sum(short_mask)
                    short_win_rate = short_wins / short_total
                    short_win_rates.append(short_win_rate)

                # Overall win rate
                total_positions = np.sum(long_mask) + np.sum(short_mask)
                if total_positions > 0:
                    total_wins = np.sum(pos_pnl > 0)
                    overall_win_rate = total_wins / total_positions
                    overall_win_rates.append(overall_win_rate)

            num_pos = metrics.get('num_long_positions', 0) + metrics.get('num_short_positions', 0)
            num_positions.append(num_pos)

        if not pnls:
            return {}

        # Calculate standard metrics
        cumulative_pnl = np.cumsum(pnls)
        total_return = cumulative_pnl[-1] / self.C_0
        periodic_returns = np.array(pnls) / self.C_0
        mean_ret = np.mean(periodic_returns)
        std_ret = np.std(periodic_returns)
        sharpe = (mean_ret - risk_free_rate / periods_per_year) / std_ret * np.sqrt(
            periods_per_year) if std_ret > 0 else 0

        avg_hold_positions = np.mean(num_positions) if num_positions else 0
        avg_transaction_cost = np.mean(transaction_costs) if transaction_costs else 0

        # Drawdown calculation
        cummax = np.maximum.accumulate(cumulative_pnl)
        drawdowns = (cummax - cumulative_pnl) / (self.C_0 + cummax)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Additional metrics
        positive_pnls = [p for p in pnls if p > 0]
        negative_pnls = [p for p in pnls if p < 0]
        avg_profit = np.mean(positive_pnls) if positive_pnls else 0
        avg_loss = np.abs(np.mean(negative_pnls)) if negative_pnls else 0
        profit_factor = sum(positive_pnls) / sum(np.abs(negative_pnls)) if negative_pnls else np.inf
        avg_pnl = np.mean(pnls)
        num_trades = len(pnls)
        total_trans_cost = sum(transaction_costs)

        # Sortino ratio
        downside_ret = [r for r in periodic_returns if r < 0]
        downside_std = np.std(downside_ret) if downside_ret else 0
        sortino = (mean_ret - risk_free_rate / periods_per_year) / downside_std * np.sqrt(
            periods_per_year) if downside_std > 0 else 0

        annualized_return = (1 + total_return) ** (periods_per_year / len(pnls)) - 1 if pnls else 0
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf

        results = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'average_profit': avg_profit,
            'average_loss': avg_loss,
            'average_pnl': avg_pnl,
            'num_trades': num_trades,
            'average_positions': avg_hold_positions,
            'average_transaction_cost': avg_transaction_cost,
            'total_transaction_costs': total_trans_cost,
        }

        # Add win rates
        if long_win_rates:
            results['long_win_rate'] = np.mean(long_win_rates)
        if short_win_rates:
            results['short_win_rate'] = np.mean(short_win_rates)
        if overall_win_rates:
            results['overall_win_rate'] = np.mean(overall_win_rates)

        # Print formatted results
        print("\n=== Profitability Metrics ===")
        metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
        metrics_df['Value'] = metrics_df['Value'].apply(
            lambda x: f"{x:.4f}" if np.isfinite(x) and isinstance(x, (float, np.float64)) else (
                np.nan if np.isnan(x) else x))
        print(metrics_df.to_string())

        return results

    def predict_pc_movement(self, current_factor_changes):
        """
        6. Predicted PC Movement
        Predict next week's PC movement: ΔPC_pred_k = α_f + Σ (β_f * ΔF_f_current)
        """
        if not hasattr(self, 'pc_regressions') or not self.pc_regressions:
            # print("No PC regressions available for prediction")
            return np.zeros(self.V.shape[1])

        # print("Predicting PC movements...")

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

        # print(f"Predicted PC movements: {self.delta_PC_pred}")
        return self.delta_PC_pred

    def predict_stock_returns(self):
        """
        7. Predicted Stock Returns
        """
        # print("Predicting stock returns...")

        # 7a. Loadings & Stock Prediction
        # Loadings matrix: V (N_stocks × K_PCs)
        # Predicted PC movement vector: ΔPC_pred = [ΔPC_pred_1, ..., ΔPC_pred_K]
        # Historic PC standard deviation vector: σ_PC = [σ_PC_1, ..., σ_PC_K]

        # Step-by-step order:
        # Element-wise multiply predicted PC movement by σ_PC: ΔPC_pred ⊙ σ_PC
        pc_weighted = self.delta_PC_pred * self.sigma_PC  # ΔPC_pred ⊙ σ_PC

        # Multiply loadings matrix V by resulting vector → gives r_hat (predicted % return per stock)
        self.r_hat = self.V @ pc_weighted  # Predicted stock returns vector: r_hat = V · (ΔPC_pred ⊙ σ_PC)

        # print(f"Predicted stock returns shape: {self.r_hat.shape}")
        # print(f"Predicted returns range: [{self.r_hat.min():.4f}, {self.r_hat.max():.4f}]")

        return self.r_hat

    def compute_centrality_weighting(self, method='correlation'):
        """
        8. Centrality Weighting
        Compute centrality matrix C from stock correlation or network adjacency
        """
        # print("Computing centrality weighting...")

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

        # Normalize centrality weights
        self.u_centrality = self.u_centrality / np.sum(self.u_centrality)

        # Weight predicted stock returns: r_hat_weighted = r_hat ⊙ u_centrality
        self.r_hat_weighted = self.r_hat * self.u_centrality  # Gives importance-adjusted expected returns

        # print(f"Centrality weights range: [{self.u_centrality.min():.4f}, {self.u_centrality.max():.4f}]")
        # print(f"Weighted returns range: [{self.r_hat_weighted.min():.4f}, {self.r_hat_weighted.max():.4f}]")

        return self.r_hat_weighted

    def optimize_portfolio(self, current_prices, stock_betas=None):
        """
        9. Optimization & Position Sizing
        """
        # print("Optimizing portfolio...")

        N = len(self.r_hat_weighted)  # Number of stocks
        C_total = self.C_0  # Total capital

        # Decision variable: weights v_i (dollar allocation per stock)
        def objective(v):
            # Objective: maximize expected return: Max Σ_i (v_i * r_hat_weighted_i)
            return -np.dot(v, self.r_hat_weighted)  # Negative for minimization

        # Constraints:
        constraints = []

        # Total portfolio allocation: Σ |v_i| ≤ C_total
        def total_allocation_constraint(v):
            return C_total - np.sum(np.abs(v))

        constraints.append({'type': 'ineq', 'fun': total_allocation_constraint})

        # Dollar-neutral allocation constraints
        def long_allocation_constraint(v):
            long_positions = v[v > 0]
            return np.sum(long_positions) - 0.65 * C_total  # Exactly 65% long

        constraints.append({'type': 'eq', 'fun': long_allocation_constraint})

        def short_allocation_constraint(v):
            short_positions = v[v < 0]
            return np.abs(np.sum(short_positions)) - 0.35 * C_total  # Exactly 35% short

        constraints.append({'type': 'eq', 'fun': short_allocation_constraint})

        # Individual stock limits and short only negative expected returns
        bounds = []
        for i in range(N):
            if self.r_hat_weighted[i] >= 0:
                # Long positions: 0 ≤ v_i_long ≤ 0.15 * C_total
                # Short only negative expected returns: v_i_short = 0 if r_hat_weighted_i ≥ 0
                bounds.append((0, 0.15 * C_total))
            else:
                # Short positions: -0.12 * C_total ≤ v_i_short ≤ 0
                bounds.append((-0.12 * C_total, 0))

        # Beta exposure constraint: |v · β| ≤ 0.1
        if stock_betas is not None:
            def beta_exposure_constraint(v):
                beta_exposure = np.dot(v, stock_betas)
                return 0.1 - np.abs(beta_exposure)

            constraints.append({'type': 'ineq', 'fun': beta_exposure_constraint})

        # Initial guess: equal weight allocation respecting long/short constraints
        v0 = np.zeros(N)
        long_stocks = self.r_hat_weighted >= 0
        short_stocks = self.r_hat_weighted < 0

        if np.any(long_stocks):
            v0[long_stocks] = (0.65 * C_total) / np.sum(long_stocks)
        if np.any(short_stocks):
            v0[short_stocks] = -(0.35 * C_total) / np.sum(short_stocks)

        # Solve optimization (linear programming)
        try:
            result = opt.minimize(objective, v0, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 1000})

            if result.success:
                self.optimal_weights = result.x  # v_i weights
                # print("Portfolio optimization successful")
                # print(f"Long positions: {np.sum(self.optimal_weights > 0)}")
                # print(f"Short positions: {np.sum(self.optimal_weights < 0)}")
                # print(f"Total long allocation: ${np.sum(self.optimal_weights[self.optimal_weights > 0]):.2f}")
                # print(f"Total short allocation: ${-np.sum(self.optimal_weights[self.optimal_weights < 0]):.2f}")
            else:
                # print("Portfolio optimization failed, using equal weights")
                self.optimal_weights = v0

        except Exception as e:
            # print(f"Optimization error: {e}, using equal weights")
            self.optimal_weights = v0

        # Convert to share quantities
        self.share_quantities = self.optimal_weights / current_prices

        return self.optimal_weights

    def execute_trades(self, current_prices):
        """
        Execute trades according to the computed v_i
        """
        # print("Executing trades...")

        # Calculate transaction costs
        total_trade_value = np.sum(np.abs(self.optimal_weights))
        transaction_cost = total_trade_value * self.transaction_costs

        # Record trade execution
        trade_record = {
            'timestamp': pd.Timestamp.now(),
            'weights': self.optimal_weights.copy(),
            'shares': self.share_quantities.copy(),
            'prices': current_prices.copy(),
            'transaction_cost': transaction_cost,
            'expected_returns': self.r_hat_weighted.copy()
        }

        self.portfolio_history.append(trade_record)

        # print(f"Trade executed with transaction cost: ${transaction_cost:.2f}")
        return trade_record

    def record_performance_metrics(self, actual_returns=None):
        """
        Record metrics: P&L, portfolio composition, long/short exposure, and trading stats
        """
        if not self.portfolio_history:
            return

        latest_trade = self.portfolio_history[-1]

        # Portfolio composition
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

        # Calculate P&L if actual returns provided
        if actual_returns is not None:
            pnl = np.dot(latest_trade['weights'], actual_returns)
            metrics['realized_pnl'] = pnl
            metrics['expected_pnl'] = np.dot(latest_trade['weights'], latest_trade['expected_returns'])
            # Store actual returns in latest_trade
            latest_trade['actual_returns'] = actual_returns.copy()

        self.performance_metrics[latest_trade['timestamp']] = metrics

        return metrics

    def weekly_rebalancing_step(self, current_factor_changes, current_prices,
                                stock_betas=None, actual_returns=None):
        """
        10. Weekly Rebalancing Steps
        """
        # print("\n=== Weekly Rebalancing Step ===")

        # Update factors, compute ΔF (current_factor_changes provided)
        # print("1. Updating factors...")

        # Predict PC movements ΔPC_pred
        # print("2. Predicting PC movements...")
        self.predict_pc_movement(current_factor_changes)

        # Compute PredictedPctChange_PC_k (done in predict_pc_movement)
        # print("3. Computing predicted PC percentage changes...")

        # Compute r_hat and r_hat_weighted
        # print("4. Computing stock return predictions...")
        self.predict_stock_returns()
        self.compute_centrality_weighting()

        # Solve optimization for weights w_i
        # print("5. Optimizing portfolio weights...")
        self.optimize_portfolio(current_prices, stock_betas)

        # Execute trades
        # print("6. Executing trades...")
        trade_record = self.execute_trades(current_prices)

        # Record performance metrics
        # print("7. Recording performance metrics...")
        metrics = self.record_performance_metrics(actual_returns)

        return trade_record, metrics

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

        # print("=== Variable Summary ===")
        # for var, description in summary.items():
            # print(f"{var}: {description}")

        return summary


def run_strategy_example():
    """
    Complete example of running the Ultimate Detailed Strategy
    """
    print("=== Ultimate Detailed Strategy Example ===\n")

    # Initialize strategy
    strategy = UltimateDetailedStrategy()

    # Define stock universe (example: energy sector ETFs and stocks)
    stock_symbols = [
        'XOM', 'CVX', 'SHEL', 'BP', 'TTE',
        'COP', 'EOG', 'DVN', 'APA',
        'MPC', 'PSX', 'VLO', 'PBF', 'DK',
        'KMI', 'WMB', 'OKE', 'ET', 'ENB',
        'SLB', 'HAL', 'BKR', 'FTI', 'NOV',
        'FANG',  # Added as a Pioneer proxy
        'HES', 'CTRA'
    ]

    # Define factor universe
    factor_symbols = [
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

    # Load data
    strategy.load_data(stock_symbols, '2020-01-01', '2024-01-01',
                       {symbol: symbol for symbol in factor_symbols})

    # Prepare returns matrix
    strategy.prepare_returns_matrix(standardize=False)

    # Compute covariance and PCA
    strategy.compute_covariance_and_pca(n_components=5)

    # Factor-PC regression
    strategy.factor_pc_regression(min_r_squared=.1, max_correlation=0.9, max_vif=5)

    # Analyze PC regressions
    strategy.analyze_pc_regressions(n_pcs=8)

    # Compute historic PC standard deviations
    strategy.compute_historic_pc_std()

    # Example of weekly rebalancing step with ACTUAL factor changes
    print("\nStep 6: Example weekly rebalancing...")

    # Calculate actual current factor changes from downloaded data
    current_factor_changes = strategy.calculate_current_factor_changes(lookback_days=5)
    print(f"Actual factor changes: {current_factor_changes}")

    # Get current prices (use last available prices)
    current_prices = strategy.P.iloc[-1].values

    # Calculate actual stock betas relative to sector benchmark (XLE)
    stock_betas = strategy.calculate_stock_betas(benchmark_symbol='XLE', lookback_days=252)
    print(f"Calculated betas - mean: {np.mean(stock_betas):.3f}, std: {np.std(stock_betas):.3f}")

    # Calculate actual returns for the period (use most recent week's actual returns)
    recent_returns = strategy.R.tail(5).mean().values  # Average returns over last 5 days

    # Execute weekly rebalancing
    trade_record, metrics = strategy.weekly_rebalancing_step(
        current_factor_changes, current_prices, stock_betas, recent_returns)

    print(f"Portfolio metrics: {metrics}")

    # Display variable summary
    strategy.get_variable_summary()

    # Add a few more rebalancing periods for better visualization
    for i in range(3):
        # Use actual factor changes from different periods
        offset_days = (i + 1) * 5
        if offset_days < len(strategy.R):
            # Calculate factor changes from historical data
            historical_factor_changes = strategy.calculate_current_factor_changes(lookback_days=5)
            historical_returns = strategy.R.iloc[-(offset_days + 5):-(offset_days)].mean().values
            strategy.weekly_rebalancing_step(historical_factor_changes, current_prices, stock_betas, historical_returns)

    strategy.plot_portfolio_metrics()
    strategy.compute_profitability_metrics()

    return strategy


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

    # print(f"\n=== PC_{pc_index + 1} Loadings Analysis ===")
    # print(loadings_df)

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
    print()
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


def backtest_strategy(strategy, start_date='2021-01-01', end_date='2023-12-31',
                      rebalance_freq=5):
    """
    Comprehensive backtesting of the strategy
    """
    print(f"\n=== Backtesting Strategy from {start_date} to {end_date} ===")

    # Get price data for backtesting period
    backtest_data = strategy.P.loc[start_date:end_date]

    if backtest_data.empty:
        print("No data available for backtesting period")
        return None

    # Calculate returns for backtesting
    backtest_returns = backtest_data.pct_change().dropna()

    portfolio_values = [strategy.C_0]  # Start with initial capital
    rebalance_dates = []

    # Simulate rebalancing every rebalance_freq days
    for i in range(rebalance_freq, len(backtest_returns), rebalance_freq):
        rebalance_date = backtest_returns.index[i]
        rebalance_dates.append(rebalance_date)

        # Get period returns
        period_returns = backtest_returns.iloc[i - rebalance_freq:i].mean().values

        # Simulate factor changes (in practice, these would be real factor data)
        factor_changes = {factor: np.random.normal(0, 0.02)
                          for factor in ['SPY', 'TLT', 'GLD', 'VXX', 'UUP']}

        current_prices = backtest_data.iloc[i].values
        stock_betas = np.random.normal(1.0, 0.3, len(strategy.stocks))

        try:
            # Execute rebalancing step
            trade_record, metrics = strategy.weekly_rebalancing_step(
                factor_changes, current_prices, stock_betas, period_returns)

            # Calculate portfolio return
            if hasattr(strategy, 'optimal_weights') and 'realized_pnl' in metrics:
                new_portfolio_value = portfolio_values[-1] + metrics['realized_pnl']
                portfolio_values.append(new_portfolio_value)
            else:
                portfolio_values.append(portfolio_values[-1])  # No change if no valid trades

        except Exception as e:
            print(f"Error at {rebalance_date}: {e}")
            portfolio_values.append(portfolio_values[-1])

    # Calculate performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(backtest_returns)) - 1

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")

    # Plot backtest results
    plt.figure(figsize=(12, 6))
    plt.plot(rebalance_dates[:len(portfolio_values) - 1], portfolio_values[1:],
             marker='o', label='Strategy Portfolio Value')
    plt.axhline(y=strategy.C_0, color='r', linestyle='--', label='Initial Capital')
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


# Run the complete example
if __name__ == "__main__":
    # Execute the full strategy example
    strategy = run_strategy_example()

    # Additional analysis
    print("\n" + "=" * 50)
    print("ADDITIONAL ANALYSIS")
    print("=" * 50)

    # Analyze PC loadings
    analyze_pc_loadings(strategy, pc_index=0)

    # Analyze factor importance
    analyze_factor_importance(strategy)

    # Run backtest
    backtest_results = backtest_strategy(strategy)
    #
    # print("\n=== Strategy Implementation Complete ===")
    # print("All components of the Ultimate Detailed Strategy have been implemented exactly as specified:")
    # print("✅ Universe & Capital management")
    # print("✅ Data Preparation with log returns matrix R")
    # print("✅ Covariance matrix Σ and PCA with loadings matrix V")
    # print("✅ PC time series matrix PC_day")
    # print("✅ Multi-factor regression with collinearity filtering")
    # print("✅ Historic PC standard deviation σ_PC")
    # print("✅ Predicted PC movements ΔPC_pred")
    # print("✅ Stock return predictions r_hat")
    # print("✅ Centrality weighting with u_centrality")
    # print("✅ Weighted predictions r_hat_weighted")
    # print("✅ Portfolio optimization with all constraints")
    # print("✅ Weekly rebalancing with complete workflow")
    # print("✅ Performance metrics and visualization")
    # print("✅ All matrix definitions and variable shapes as specified")