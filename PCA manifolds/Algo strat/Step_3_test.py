import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import scipy.linalg as la
from itertools import combinations
import pandas.tseries.offsets as offsets
from sklearn.linear_model import LassoCV

class PCAFactorStrategy:
    def __init__(self, stocks, start_date, end_date, lookback=252, initial_capital=10000, LR_lookback=30):
        self.stocks = stocks  # Industrial ETFs and stocks
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.LR_lookback = LR_lookback
        self.initial_capital = initial_capital
        self.data = None
        self.rebalance_dates = None
        self.factors = ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']

        # Original rotation pairs
        self.rotation_pairs = {
            "Growth vs Value": ("VUG", "VTV"),
            "Large vs Small Cap": ("SPY", "IWM"),
            "Tech vs Market": ("XLK", "SPY"),
            "Financials vs Market": ("XLF", "SPY"),
            "Banking vs Financials": ("KBE", "XLF"),
            "Regional vs Banks": ("KRE", "KBE")
        }

        # NEW: Additional factor categories
        self.momentum_factors = {
            "High vs Low Beta": ("SPHB", "SPLV"),  # High beta vs low volatility
            "Momentum vs Anti-momentum": ("MTUM", "VMOT"),  # Momentum vs min volatility
            "Quality vs Junk": ("QUAL", "SJNK")  # Quality vs high yield junk
        }

        self.macro_factors = {
            "Dollar Strength": ("UUP", "UDN"),  # Dollar up vs dollar down
            "Inflation Expectation": ("SCHP", "VTEB"),  # TIPS vs Tax-exempt bonds
            "Credit Spread": ("LQD", "HYG"),  # Investment grade vs High yield
            "Yield Curve": ("SHY", "TLT"),  # Short vs long treasury
            "Real vs Nominal": ("VTEB", "VGIT")  # Tax-exempt vs intermediate gov
        }

        self.sector_rotation_factors = {
            "Cyclical vs Defensive": ("XLI", "XLP"),  # Industrial vs Consumer staples
            "Risk-on vs Risk-off": ("XLY", "XLRE"),  # Consumer disc vs REITs
            "Energy vs Utilities": ("XLE", "XLU"),
            "Healthcare vs Tech": ("XLV", "XLK"),
            "Materials vs Staples": ("XLB", "XLP")
        }

        self.volatility_factors = {
            "Vol Surface": ("VXX", "XIV"),  # VIX futures vs inverse
            "Term Structure": ("VIX9D", "^VIX"),  # Short vs medium term vol
            "Equity vs Bond Vol": ("^VIX", "^MOVE")  # Equity vol vs bond vol (MOVE index)
        }

        # Store all factor categories
        self.all_factor_categories = {
            **self.rotation_pairs,
            **self.momentum_factors,
            **self.macro_factors,
            **self.sector_rotation_factors,
            **self.volatility_factors
        }

        self.factor_data = None
        self.pca_matrix_count = 0
        self.selected_factors = {}
        self.r2_history = {f'PC_{i + 1}': [] for i in range(5)}
        self.r2_training_history = {f'PC_{i + 1}': [] for i in range(5)}

    def download_data(self):
        nominal_start = pd.to_datetime(self.start_date)
        earliest_data_start = nominal_start - offsets.BDay(self.lookback + 15 + 20)

        # Download stock data
        raw_data = yf.download(self.stocks, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            self.data = raw_data['Close']
        else:
            self.data = raw_data
        self.data = self.data.dropna(axis=0, how='any')

        # Download individual factor data
        raw_factor_data = yf.download(self.factors, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        if isinstance(raw_factor_data.columns, pd.MultiIndex):
            self.factor_data = raw_factor_data['Close']
        else:
            self.factor_data = raw_factor_data
        self.factor_data = self.factor_data.dropna(axis=1, how='all').dropna(axis=0, how='any')

        # Download ALL factor category data (rotation + new factors)
        all_factor_tickers = []
        for pair in self.all_factor_categories.values():
            all_factor_tickers.extend(pair)
        all_factor_tickers = list(set(all_factor_tickers))  # Remove duplicates

        print(f"Attempting to download {len(all_factor_tickers)} unique factor tickers...")

        raw_rotation_data = yf.download(all_factor_tickers, start=earliest_data_start, end=self.end_date,
                                        auto_adjust=True)
        if isinstance(raw_rotation_data.columns, pd.MultiIndex):
            self.rotation_data = raw_rotation_data['Close']
        else:
            self.rotation_data = raw_rotation_data
        self.rotation_data = self.rotation_data.dropna(axis=1, how='all').dropna(axis=0, how='any')

        # Compute ALL factor categories and add to factor data
        computed_factors = self.compute_all_factor_categories()
        if not computed_factors.empty:
            # Align dates between factor_data and computed_factors
            common_dates = self.factor_data.index.intersection(computed_factors.index)
            self.factor_data = self.factor_data.loc[common_dates]
            computed_factors = computed_factors.loc[common_dates]

            # Combine factor data with computed factors
            self.factor_data = pd.concat([self.factor_data, computed_factors], axis=1)

        if self.factor_data.empty or len(self.factor_data.columns) == 0:
            raise ValueError("No valid factor data available")

        # Update factors list to include all computed factors
        self.factors = list(self.factor_data.columns)

        # Setup rebalance dates
        all_dates = self.data.index
        self.rebalance_dates = all_dates[all_dates.weekday == 4]
        first_possible_rebalance = all_dates[all_dates >= (nominal_start + offsets.BDay(self.lookback))][0]
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates >= first_possible_rebalance]
        rebalance_diffs = self.rebalance_dates[1:] - self.rebalance_dates[:-1]
        rebalance_days = [diff.days for diff in rebalance_diffs]

        print(
            f"Stock data shape: {self.data.shape}, Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Factor data shape: {self.factor_data.shape}, Total factors: {len(self.factors)}")
        print(f"Successfully added factor categories: {len(self.all_factor_categories)}")
        print(f"Rebalance dates: {len(self.rebalance_dates)}, Mean days between: {np.mean(rebalance_days):.2f}")

    def compute_all_factor_categories(self):
        """
        Compute ALL factor categories as the difference in log returns between paired assets.
        Returns a DataFrame with all factor categories as daily log return differences.
        """
        if not hasattr(self, 'rotation_data') or self.rotation_data.empty:
            print("Warning: No factor data available")
            return pd.DataFrame()

        all_factors = pd.DataFrame(index=self.rotation_data.index)

        # Compute log returns for all factor assets
        factor_returns = self.compute_log_returns(self.rotation_data)

        successful_factors = []
        failed_factors = []

        # Compute all factor categories
        for factor_name, (asset1, asset2) in self.all_factor_categories.items():
            if asset1 in factor_returns.columns and asset2 in factor_returns.columns:
                # Factor = log return of first asset - log return of second asset
                # Positive values indicate first asset outperforming second asset
                all_factors[factor_name] = factor_returns[asset1] - factor_returns[asset2]
                successful_factors.append(factor_name)
            else:
                missing_assets = [asset for asset in [asset1, asset2] if asset not in factor_returns.columns]
                failed_factors.append((factor_name, missing_assets))

        # Drop any NaN values
        all_factors = all_factors.dropna()

        print(f"Successfully computed {len(successful_factors)} factors:")
        for category, factors in [
            ("Rotation", list(self.rotation_pairs.keys())),
            ("Momentum", list(self.momentum_factors.keys())),
            ("Macro", list(self.macro_factors.keys())),
            ("Sector Rotation", list(self.sector_rotation_factors.keys())),
            ("Volatility", list(self.volatility_factors.keys()))
        ]:
            successful_in_category = [f for f in factors if f in successful_factors]
            if successful_in_category:
                print(f"  {category}: {successful_in_category}")

        if failed_factors:
            print(f"\nFailed to compute {len(failed_factors)} factors (missing data):")
            for factor_name, missing in failed_factors:
                print(f"  {factor_name}: Missing {missing}")

        print(f"Factor data date range: {all_factors.index[0].date()} to {all_factors.index[-1].date()}")
        print(f"Factor data shape: {all_factors.shape}")

        return all_factors

    def compute_rotation_factors(self):
        """
        Compute rotation factors as the difference in log returns between paired assets.
        Returns a DataFrame with rotation factors as daily log return differences.
        """
        if not hasattr(self, 'rotation_data') or self.rotation_data.empty:
            print("Warning: No rotation data available")
            return pd.DataFrame()

        rotation_factors = pd.DataFrame(index=self.rotation_data.index)

        # Compute log returns for all rotation assets
        rotation_returns = self.compute_log_returns(self.rotation_data)

        # Compute rotation factors as difference in log returns
        for factor_name, (asset1, asset2) in self.rotation_pairs.items():
            if asset1 in rotation_returns.columns and asset2 in rotation_returns.columns:
                # Rotation factor = log return of first asset - log return of second asset
                # Positive values indicate first asset outperforming second asset
                rotation_factors[factor_name] = rotation_returns[asset1] - rotation_returns[asset2]
            else:
                missing_assets = [asset for asset in [asset1, asset2] if asset not in rotation_returns.columns]
                print(f"Warning: Cannot compute {factor_name} factor. Missing assets: {missing_assets}")

        # Drop any NaN values
        rotation_factors = rotation_factors.dropna()

        print(f"Computed rotation factors: {list(rotation_factors.columns)}")
        print(f"Rotation factors date range: {rotation_factors.index[0].date()} to {rotation_factors.index[-1].date()}")
        print(f"Rotation factors shape: {rotation_factors.shape}")

        return rotation_factors

    def compute_log_returns(self, prices):
        """
        Compute log returns with proper handling of edge cases to avoid warnings.
        """
        # Replace any non-positive values with NaN before taking log
        prices_clean = prices.where(prices > 0)

        # Compute log returns
        returns = np.log(prices_clean / prices_clean.shift(1)).dropna()

        # Replace any infinite values with NaN
        returns = returns.replace([np.inf, -np.inf], np.nan)

        return returns

    def standardize_returns(self, returns):
        scaler = StandardScaler()
        std_returns = pd.DataFrame(scaler.fit_transform(returns), index=returns.index, columns=returns.columns)
        return std_returns, scaler

    def compute_pca(self, returns):
        cov_matrix = returns.cov()
        eigenvalues, eigenvectors = la.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained_var = eigenvalues / np.sum(eigenvalues)  # Individual explained variance
        k = min(5, len(eigenvalues))
        V = eigenvectors[:, :k]
        for i in range(k):
            if np.sum(V[:, i]) < 0:
                V[:, i] = -V[:, i]
        self.pca_matrix_count += 1
        return V, eigenvalues[:k], explained_var[:k]

    def compute_centrality_matrix(self, returns):
        corr_matrix = returns.corr()
        eigenvalues, eigenvectors = la.eigh(corr_matrix)
        idx = np.argmax(eigenvalues)
        u_centrality = eigenvectors[:, idx]
        lambda_max = eigenvalues[idx]
        # Eigenvalue verification
        ev_check = np.allclose(corr_matrix.to_numpy() @ u_centrality, lambda_max * u_centrality, atol=1e-6)
        u_scaled = u_centrality / np.std(u_centrality) * 0.13
        u_centrality = u_scaled + (1 - np.mean(u_scaled))
        return corr_matrix.to_numpy(), u_centrality

    def compute_pc_series(self, returns, V):
        returns_np = returns.to_numpy()
        pc_series_np = returns_np @ V
        pc_series = pd.DataFrame(pc_series_np, index=returns.index, columns=[f'PC_{i+1}' for i in range(V.shape[1])])
        if pc_series.isna().sum().sum() > 0:
            pc_series = pc_series.dropna()
        return pc_series

    def compute_factor_changes(self, factor_prices, start_date, end_date):
        """
        Updated to handle both regular factors and ALL computed factor categories.
        Note: All computed factors are already in daily change format from compute_all_factor_categories(),
        while regular factors need to be converted to log returns.
        """
        if factor_prices.empty:
            return pd.DataFrame()

        try:
            prices = factor_prices.loc[start_date:end_date]
            if len(prices) < 2:
                return pd.DataFrame()

            # Identify which columns are computed factors (already in return format)
            computed_factor_names = list(self.all_factor_categories.keys())
            regular_factors = [col for col in prices.columns if col not in computed_factor_names]
            computed_factors = [col for col in prices.columns if col in computed_factor_names]

            result = pd.DataFrame(index=prices.index[1:])  # Skip first row for regular factors

            # Process regular factors (convert to log returns)
            if regular_factors:
                regular_prices = prices[regular_factors]
                regular_returns = self.compute_log_returns(regular_prices)
                result = pd.concat([result, regular_returns], axis=1)

            # Process computed factors (already in return format, just slice the date range)
            if computed_factors:
                computed_data = prices[computed_factors].loc[result.index]  # Align with regular factors index
                result = pd.concat([result, computed_data], axis=1)

            return result
        except KeyError:
            return pd.DataFrame()

    def factor_regression(self, pc_series, factor_changes, rebalance_date, std_returns, V, scaler):
        if factor_changes.empty or rebalance_date not in pc_series.index or rebalance_date not in factor_changes.index:
            return {}

        lookback = self.LR_lookback
        try:
            rebalance_idx = factor_changes.index.get_loc(rebalance_date)
            start_idx = max(0, rebalance_idx - lookback + 1)
            period_dates = factor_changes.index[start_idx:rebalance_idx + 1]
            common_dates = pc_series.index.intersection(period_dates)
            if len(common_dates) < 20:
                return {}
            pc_series_window = pc_series.loc[common_dates]
            factor_changes_window = factor_changes.loc[common_dates]
        except (KeyError, IndexError):
            return {}

        results = {}
        self.selected_factors = {}

        # Use LASSO with CV for each PC
        for pc in pc_series_window.columns:
            y = pc_series_window[pc].to_numpy()
            X = factor_changes_window.to_numpy()
            if np.any(np.isnan(y)) or np.any(np.isnan(X)):
                self.r2_history[pc].append((rebalance_date, 0.0))
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                continue

            # LASSO with cross-validation (alphas grid) - increased max_iter and tol for convergence
            lasso = LassoCV(
                cv=5,
                alphas=np.logspace(-4, 0, 30),
                max_iter=10000,  # Increased from 5000
                tol=1e-4,  # Added explicit tolerance
                random_state=42  # Added for reproducibility
            )
            lasso.fit(X, y)
            best_r2_training = lasso.score(X, y)

            # Adjusted R²
            n_training = len(y)
            p = np.sum(lasso.coef_ != 0)
            adjusted_r2_training = 1 - (1 - best_r2_training) * (n_training - 1) / (
                    n_training - p - 1) if n_training > p + 1 and p > 0 else best_r2_training

            # Future prediction R² (next 5 days)
            r2_prediction = 0.0
            try:
                future_dates = self.data.index[self.data.index > rebalance_date][:5]
                if len(future_dates) >= 3:
                    future_end_date = future_dates[-1]
                    future_stock_prices = self.data.loc[rebalance_date:future_end_date]
                    future_returns = self.compute_log_returns(future_stock_prices)
                    if not future_returns.empty and len(future_returns) >= 3:
                        future_std_returns = pd.DataFrame(scaler.transform(future_returns),
                                                          index=future_returns.index,
                                                          columns=future_returns.columns)
                        pc_future = self.compute_pc_series(future_std_returns, V)
                        future_factor_prices = self.factor_data.loc[rebalance_date:future_end_date]
                        future_factor_changes = self.compute_factor_changes(future_factor_prices, rebalance_date,
                                                                            future_end_date)
                        common_future_dates = pc_future.index.intersection(future_factor_changes.index)
                        if len(common_future_dates) >= 3:
                            y_pred = pc_future.loc[common_future_dates, pc].to_numpy()
                            X_pred = future_factor_changes.loc[common_future_dates].to_numpy()
                            if not (np.any(np.isnan(y_pred)) or np.any(np.isnan(X_pred))):
                                r2_prediction = lasso.score(X_pred, y_pred)
            except Exception:
                r2_prediction = 0.0

            # Save only non-zero coefficients
            nonzero_idx = np.where(lasso.coef_ != 0)[0]
            best_factors = [self.factors[i] for i in nonzero_idx]
            betas = {self.factors[i]: lasso.coef_[i] for i in nonzero_idx}

            results[pc] = {
                'alpha': lasso.intercept_,
                'beta': betas,
                'r2_training': adjusted_r2_training,
                'r2_rebalancing': r2_prediction,
                'factors': best_factors
            }

            self.selected_factors[pc] = best_factors
            self.r2_training_history[pc].append((rebalance_date, adjusted_r2_training))
            self.r2_history[pc].append((rebalance_date, r2_prediction))

        return results


    def compute_predictions(self, pc_series, factor_changes, regression_results, rebalance_date, V, u_centrality):
        if not regression_results:
            return None, None, None, None, None

        k = V.shape[1]
        delta_pc_pred_vec = np.zeros(k)
        sigma_pc_vec = pc_series.std().values

        for pc_idx, pc in enumerate(pc_series.columns):
            if pc in regression_results:
                res = regression_results[pc]
                best_factors = res['factors']
                if not best_factors:
                    delta_pc_pred_vec[pc_idx] = 0.0
                    continue
                # Use the latest delta factors (on rebalance_date)
                try:
                    X_current = factor_changes.loc[rebalance_date, best_factors].to_numpy()
                    delta_pc_pred = res['alpha'] + np.dot(list(res['beta'].values()), X_current)
                    delta_pc_pred_vec[pc_idx] = delta_pc_pred
                except KeyError:
                    delta_pc_pred_vec[pc_idx] = 0.0

        # Scale for weekly (5 days)
        delta_pc_pred_vec *= 5

        # Predicted PCT change for PCs (weekly)
        predicted_pct_vec = delta_pc_pred_vec * sigma_pc_vec

        # Predicted stock returns (weekly)
        r_hat = V @ predicted_pct_vec

        # Weighted with centrality
        r_hat_weighted = r_hat * u_centrality

        return sigma_pc_vec, delta_pc_pred_vec, predicted_pct_vec, r_hat, r_hat_weighted

    def validate_step3(self):
        validation_results = []
        for rebalance_date in self.rebalance_dates:
            idx = self.data.index.get_loc(rebalance_date)
            lookback_start = self.data.index[max(0, idx - self.lookback + 1)]
            prices = self.data.loc[lookback_start:rebalance_date]
            returns = self.compute_log_returns(prices)
            std_returns, scaler = self.standardize_returns(returns)
            V, eigenvalues, explained_var = self.compute_pca(std_returns)
            C, u_centrality = self.compute_centrality_matrix(std_returns)
            pc_series = self.compute_pc_series(std_returns, V)
            factor_changes = self.compute_factor_changes(self.factor_data, lookback_start, rebalance_date)
            regression_results = self.factor_regression(pc_series, factor_changes, rebalance_date, std_returns, V,
                                                        scaler)
            sigma_pc_vec, delta_pc_pred_vec, predicted_pct_vec, r_hat, r_hat_weighted = self.compute_predictions(
                pc_series, factor_changes, regression_results, rebalance_date, V, u_centrality
            )
            validation_results.append({
                'date': rebalance_date,
                'returns_shape': returns.shape,
                'std_returns_mean': std_returns.mean().mean(),
                'std_returns_std': std_returns.std().mean(),
                'pca_eigenvalues': eigenvalues.tolist(),
                'pca_explained_var': explained_var.tolist(),
                'pca_orthogonal': np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-6),
                'centrality_mean': np.mean(u_centrality),
                'centrality_std': np.std(u_centrality),
                'pc_series_shape': pc_series.shape,
                'regression_results': regression_results,
                'sigma_pc_vec': sigma_pc_vec.tolist() if sigma_pc_vec is not None else None,
                'delta_pc_pred_vec': delta_pc_pred_vec.tolist() if delta_pc_pred_vec is not None else None,
                'predicted_pct_vec': predicted_pct_vec.tolist() if predicted_pct_vec is not None else None,
                'r_hat_mean': np.mean(r_hat) if r_hat is not None else None,
                'r_hat_std': np.std(r_hat) if r_hat is not None else None,
                'r_hat_min': np.min(r_hat) if r_hat is not None else None,
                'r_hat_max': np.max(r_hat) if r_hat is not None else None,
                'r_hat_weighted_mean': np.mean(r_hat_weighted) if r_hat_weighted is not None else None,
                'r_hat_weighted_std': np.std(r_hat_weighted) if r_hat_weighted is not None else None,
                'r_hat_weighted_min': np.min(r_hat_weighted) if r_hat_weighted is not None else None,
                'r_hat_weighted_max': np.max(r_hat_weighted) if r_hat_weighted is not None else None,
            })

        print("\nValidation Summary:")
        print(f"Total PCA matrices: {self.pca_matrix_count}")
        print(f"Total rebalance periods: {len(validation_results)}")
        print(f"Linear Regression lookback period: {self.LR_lookback} days")
        print("\nLast 5 Rebalance Periods:")
        for result in validation_results[-5:]:
            print(f"\nDate: {result['date'].date()}")
            print(f"Returns Shape: {result['returns_shape']}")
            print(f"Std Returns Mean: {result['std_returns_mean']:.6f}, Std: {result['std_returns_std']:.6f}")
            print(f"PC Sector Movement: " + ", ".join(
                f"PC_{i + 1}: {var * 100:.2f}%" for i, var in enumerate(result['pca_explained_var'])))
            print(f"PCA Orthogonal: {result['pca_orthogonal']}")
            print(f"Centrality Mean: {result['centrality_mean']:.6f}, Std: {result['centrality_std']:.6f}")
            print(f"PC Series Shape: {result['pc_series_shape']}")
            for pc, res in result['regression_results'].items():
                print(f"{pc}:")
                print(f"  Eq: PC = {res['alpha']:.6f} + " + " + ".join(
                    f"{coef:.6f}*{f}" for f, coef in res['beta'].items()))
                print(f"  Factors: {res['factors']}")
                print(
                    f"  R² (Training {self.LR_lookback}-day): {res['r2_training']:.6f}, R² (Prediction 5-day): {res['r2_rebalancing']:.6f}")
            if result['sigma_pc_vec'] is not None:
                print(f"Historic PC Std: " + ", ".join(
                    f"PC_{i + 1}: {val:.6f}" for i, val in enumerate(result['sigma_pc_vec'])))
                print(f"Delta PC Pred (weekly scaled): " + ", ".join(
                    f"PC_{i + 1}: {val:.6f}" for i, val in enumerate(result['delta_pc_pred_vec'])))
                print(f"Predicted PCT Vec (weekly): " + ", ".join(
                    f"PC_{i + 1}: {val:.6f}" for i, val in enumerate(result['predicted_pct_vec'])))
                print(
                    f"r_hat (weekly) - Mean: {result['r_hat_mean']:.6f}, Std: {result['r_hat_std']:.6f}, Min: {result['r_hat_min']:.6f}, Max: {result['r_hat_max']:.6f}")
                print(
                    f"r_hat_weighted (weekly) - Mean: {result['r_hat_weighted_mean']:.6f}, Std: {result['r_hat_weighted_std']:.6f}, Min: {result['r_hat_weighted_min']:.6f}, Max: {result['r_hat_weighted_max']:.6f}")

        if any(self.r2_history.values()):
            print("\nR² Summary (Top 3 PCs):")
            for pc in [f'PC_{i + 1}' for i in range(min(3, len(self.r2_history)))]:
                if self.r2_history[pc]:
                    training_r2s = [r2 for _, r2 in self.r2_training_history[pc]]
                    prediction_r2s = [r2 for _, r2 in self.r2_history[pc]]
                    print(f"{pc}:")
                    print(
                        f"  Training R² ({self.LR_lookback}-day) Mean: {np.mean(training_r2s):.4f}, Std: {np.std(training_r2s):.4f}")
                    print(
                        f"  Prediction R² (5-day) Mean: {np.mean(prediction_r2s):.4f}, Std: {np.std(prediction_r2s):.4f}")
                    if len(training_r2s) > 1 and len(prediction_r2s) > 1:
                        print(
                            f"  Training vs Prediction R² Correlation: {np.corrcoef(training_r2s, prediction_r2s)[0, 1]:.4f}")

        # New code to print percentage of sector movement for PC1-PC5 as the last output
        print("\nPercentage of Sector Movement Represented by PCs (Across All Rebalance Periods):")
        all_explained_vars = np.array([result['pca_explained_var'] for result in validation_results])
        mean_explained_vars = np.mean(all_explained_vars, axis=0)
        for i, var in enumerate(mean_explained_vars[:5]):  # Limit to PC1-PC5
            print(f"PC_{i + 1}: {var * 100:.2f}%")

        # Plotting remains unchanged
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Training R² (regression line fit on LR_lookback period)
        plotted_training = False
        for pc in [f'PC_{i + 1}' for i in range(min(3, len(self.r2_training_history)))]:
            if self.r2_training_history[pc]:
                dates, r2s = zip(*self.r2_training_history[pc])
                ax1.plot(dates, r2s, marker='o', label=f'Training R² {pc}', linewidth=2, markersize=4)
                plotted_training = True
        if plotted_training:
            ax1.set_title(f'Training R² ({self.LR_lookback}-day period): Regression Model Fit Quality', fontsize=14,
                          fontweight='bold')
            ax1.set_xlabel('Rebalance Date')
            ax1.set_ylabel(f'R² (regression fit on {self.LR_lookback}-day training period)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Prediction R² (regression line performance on next 5 days)
        plotted_prediction = False
        for pc in [f'PC_{i + 1}' for i in range(min(3, len(self.r2_history)))]:
            if self.r2_history[pc]:
                dates, r2s = zip(*self.r2_history[pc])
                ax2.plot(dates, r2s, marker='s', label=f'Prediction R² {pc}', linewidth=2, markersize=4)
                plotted_prediction = True
        if plotted_prediction:
            ax2.set_title('Prediction R² (5-day period): Regression Model Prediction Quality', fontsize=14,
                          fontweight='bold')
            ax2.set_xlabel('Rebalance Date')
            ax2.set_ylabel('R² (regression predictions on next 5 days)')
            ax2.set_ylim(-0.5, 1)  # Set y-axis limits from -0.5 to 1
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

        if plotted_training or plotted_prediction:
            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
    start_date = '2015-01-01'
    end_date = '2025-08-22'
    strategy = PCAFactorStrategy(stocks, start_date, end_date, LR_lookback=150)  # You can easily change this value
    strategy.download_data()
    strategy.validate_step3()

