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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

class PCAFactorStrategy:
    def __init__(self, stocks, start_date, end_date, lookback=252, initial_capital=10000, LR_lookback=30):
        self.stocks = stocks  # Industrial ETFs and stocks
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.LR_lookback = LR_lookback
        self.prediction_window = 10
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
        u_centrality = eigenvectors[:, idx]  # Don't take absolute value yet
        lambda_max = eigenvalues[idx]

        # Eigenvalue verification
        ev_check = np.allclose(corr_matrix.to_numpy() @ eigenvectors[:, idx], lambda_max * eigenvectors[:, idx],
                               atol=1e-6)

        # The centrality weighting should be the first eigenvector values
        # Take absolute value to ensure positive weights
        u_centrality = np.abs(u_centrality)

        # Normalize the centrality vector to have mean=1 and std=0.13 as specified
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
        """
        Updated to train regression models to predict weekly PC movements using weekly factor changes.
        Key fix: Properly store the scaler for each PC regression.
        """
        if factor_changes.empty or rebalance_date not in pc_series.index or rebalance_date not in factor_changes.index:
            print(f"Skipping regression for {rebalance_date}: Missing data")
            return {}

        lookback = self.LR_lookback  # Now in weeks
        try:
            rebalance_idx = factor_changes.index.get_loc(rebalance_date)
            start_idx = max(0, rebalance_idx - lookback + 1)
            period_dates = factor_changes.index[start_idx:rebalance_idx + 1]
            common_dates = pc_series.index.intersection(period_dates)
            if len(common_dates) < max(6, lookback // 2):  # Need minimum weekly samples
                print(
                    f"Skipping regression for {rebalance_date}: Insufficient data ({len(common_dates)} < {max(6, lookback // 2)})")
                return {}
            pc_series_window = pc_series.loc[common_dates]
            factor_changes_window = factor_changes.loc[common_dates]
        except (KeyError, IndexError):
            print(f"Skipping regression for {rebalance_date}: Data alignment error")
            return {}

        results = {}
        self.selected_factors = {}

        pc_settings = {
            'PC_1': {'alphas': np.logspace(-5, -1, 40), 'cv': min(5, len(common_dates) // 2), 'max_iter': 50000,
                     'tol': 1e-4},
            'PC_2': {'alphas': np.logspace(-4, 0, 30), 'cv': min(5, len(common_dates) // 2), 'max_iter': 30000,
                     'tol': 1e-3},
            'PC_3': {'alphas': np.logspace(-3, 1, 30), 'cv': min(5, len(common_dates) // 2), 'max_iter': 30000,
                     'tol': 1e-3},
            'PC_4': {'alphas': np.logspace(-3, 1, 30), 'cv': min(5, len(common_dates) // 2), 'max_iter': 30000,
                     'tol': 1e-3},
            'PC_5': {'alphas': np.logspace(-3, 1, 30), 'cv': min(5, len(common_dates) // 2), 'max_iter': 30000,
                     'tol': 1e-3}
        }

        for pc in pc_series_window.columns:
            settings = pc_settings.get(pc, {
                'alphas': np.logspace(-4, 0, 30),
                'cv': min(5, len(common_dates) // 2),
                'max_iter': 30000,
                'tol': 1e-3
            })

            # CREATE WEEKLY TARGETS: For each week, what was PC movement over next week?
            weekly_targets = []
            current_factors = []
            valid_dates = []

            for i, date in enumerate(pc_series_window.index[:-1]):  # Skip last date (no future data)
                try:
                    # Get PC value at current week
                    current_pc = pc_series_window.loc[date, pc]

                    # Get PC value next week
                    future_dates = pc_series_window.index[pc_series_window.index > date]
                    if len(future_dates) >= 1:
                        future_date = future_dates[0]  # Next week
                        future_pc = pc_series_window.loc[future_date, pc]

                        # Weekly PC movement = future_pc - current_pc
                        weekly_movement = future_pc - current_pc

                        # Get current weekly factor changes as features
                        current_factor_changes = factor_changes_window.loc[date].values

                        weekly_targets.append(weekly_movement)
                        current_factors.append(current_factor_changes)
                        valid_dates.append(date)

                except (KeyError, IndexError):
                    continue

            if len(weekly_targets) < max(6, lookback // 2):  # Need minimum samples
                print(f"Skipping regression for {pc}: Insufficient weekly samples ({len(weekly_targets)})")
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                self.r2_history[pc].append((rebalance_date, 0.0))
                continue

            # Convert to numpy arrays
            y = np.array(weekly_targets)  # Weekly PC movements
            X_raw = np.array(current_factors)  # Weekly factor changes

            if np.any(np.isnan(y)) or np.any(np.isnan(X_raw)):
                print(f"Skipping regression for {pc}: NaN values in weekly targets")
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                self.r2_history[pc].append((rebalance_date, 0.0))
                continue

            # Pre-filter features based on correlation with weekly targets
            correlations = np.abs([np.corrcoef(X_raw[:, i], y)[0, 1] for i in range(X_raw.shape[1])
                                   if not np.all(np.isnan(X_raw[:, i]))])
            n_select = min(15, X_raw.shape[1], len(weekly_targets) // 2)  # Conservative feature selection
            selected_features = np.argsort(correlations)[-n_select:]
            X_selected = X_raw[:, selected_features]
            selected_factors = [self.factors[i] for i in selected_features]

            # Standardize the selected features - CRITICAL: Store this scaler
            scaler_X = StandardScaler()
            X = scaler_X.fit_transform(X_selected)

            lasso = LassoCV(
                alphas=settings['alphas'],
                cv=settings['cv'],
                max_iter=settings['max_iter'],
                tol=settings['tol'],
                random_state=42,
                n_jobs=-1
            )

            try:
                lasso.fit(X, y)  # Training on weekly targets
            except:
                print(f"Fallback regression for {pc} on {rebalance_date}")
                lasso.alphas_ = np.logspace(-6, 2, 50)
                lasso.max_iter = 50000
                lasso.tol = 1e-3
                lasso.fit(X, y)

            training_r2 = lasso.score(X, y)

            # Adjusted R²
            n_samples = len(y)
            n_features = np.sum(lasso.coef_ != 0)
            adjusted_r2 = 1 - (1 - training_r2) * (n_samples - 1) / (
                    n_samples - n_features - 1) if n_samples > n_features + 1 else training_r2

            # Get final non-zero coefficient factors
            nonzero_idx = np.where(lasso.coef_ != 0)[0]
            if len(nonzero_idx) == 0:
                print(f"No non-zero coefficients for {pc} on {rebalance_date}")
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                self.r2_history[pc].append((rebalance_date, 0.0))
                continue

            final_features = selected_features[nonzero_idx]
            final_factors = [self.factors[i] for i in final_features]
            X_final = X_selected[:, nonzero_idx]

            # Re-fit scaler and model on final features - CRITICAL: This is what gets stored
            scaler_X_final = StandardScaler()
            X_final_scaled = scaler_X_final.fit_transform(X_final)

            lasso_final = LassoCV(
                alphas=settings['alphas'],
                cv=settings['cv'],
                max_iter=settings['max_iter'],
                tol=settings['tol'],
                random_state=42,
                n_jobs=-1
            )
            lasso_final.fit(X_final_scaled, y)

            betas = {final_factors[i]: lasso_final.coef_[i] for i in range(len(final_factors))}

            # Compute out-of-sample prediction R² for validation (next few weeks)
            prediction_r2 = 0.0
            try:
                future_weeks = self.weekly_data.index[self.weekly_data.index > rebalance_date][:4]  # Next 4 weeks
                if len(future_weeks) >= 2:
                    future_end_date = future_weeks[-1]
                    future_stock_prices = self.weekly_data.loc[rebalance_date:future_end_date]
                    future_returns = self.compute_weekly_log_returns(future_stock_prices)
                    if not future_returns.empty and len(future_returns) >= 2:
                        future_std_returns = pd.DataFrame(
                            scaler.transform(future_returns),
                            index=future_returns.index,
                            columns=future_returns.columns
                        )
                        pc_future = self.compute_pc_series(future_std_returns, V)

                        if len(pc_future) >= 2:
                            weekly_val_targets = []
                            val_factors = []

                            for i, date in enumerate(pc_future.index[:-1]):
                                try:
                                    current_pc = pc_future.loc[date, pc]
                                    next_dates = pc_future.index[pc_future.index > date]
                                    if len(next_dates) >= 1:
                                        future_pc = pc_future.loc[next_dates[0], pc]
                                        weekly_movement = future_pc - current_pc

                                        # Get weekly factor data for this date
                                        future_factor_prices = self.weekly_factor_data.loc[
                                            rebalance_date:future_end_date]
                                        future_factor_changes = self.compute_weekly_factor_changes(
                                            future_factor_prices, rebalance_date, future_end_date
                                        )

                                        if date in future_factor_changes.index:
                                            factor_vals = future_factor_changes.loc[date, final_factors].values
                                            weekly_val_targets.append(weekly_movement)
                                            val_factors.append(factor_vals)
                                except:
                                    continue

                            if len(weekly_val_targets) >= 2:
                                y_true = np.array(weekly_val_targets)
                                X_pred = scaler_X_final.transform(np.array(val_factors))

                                if not (np.any(np.isnan(y_true)) or np.any(np.isnan(X_pred))):
                                    y_pred = lasso_final.predict(X_pred)
                                    ss_res = np.sum((y_true - y_pred) ** 2)
                                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                                    prediction_r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                                    prediction_r2 = max(-0.5, min(1.0, prediction_r2))
            except:
                prediction_r2 = 0.0

            # CRITICAL FIX: Store the final scaler that matches the final factors
            results[pc] = {
                'alpha': lasso_final.intercept_,
                'beta': betas,
                'r2_training': adjusted_r2,
                'r2_rebalancing': prediction_r2,
                'factors': final_factors,
                'model_type': 'lasso_weekly',
                'scaler_X': scaler_X_final  # This is the key fix - store the right scaler
            }

            self.selected_factors[pc] = final_factors
            self.r2_training_history[pc].append((rebalance_date, adjusted_r2))
            self.r2_history[pc].append((rebalance_date, prediction_r2))

        return results

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
            print("\nR² Summary (Top 5 PCs):")
            for pc in [f'PC_{i + 1}' for i in range(min(5, len(self.r2_history)))]:
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

        # Fixed Plotting Section
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Training R² (regression line fit on LR_lookback period)
        plotted_training = False
        for pc in [f'PC_{i + 1}' for i in range(min(3, len(self.r2_training_history)))]:
            if self.r2_training_history[pc] and len(self.r2_training_history[pc]) > 0:
                # Extract dates and R² values safely
                dates = []
                r2s = []
                for date_r2_tuple in self.r2_training_history[pc]:
                    if isinstance(date_r2_tuple, tuple) and len(date_r2_tuple) == 2:
                        dates.append(date_r2_tuple[0])
                        r2s.append(date_r2_tuple[1])
                    elif isinstance(date_r2_tuple, list) and len(date_r2_tuple) == 2:
                        dates.append(date_r2_tuple[0])
                        r2s.append(date_r2_tuple[1])

                if len(dates) > 0 and len(r2s) > 0:
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
            if self.r2_history[pc] and len(self.r2_history[pc]) > 0:
                # Extract dates and R² values safely
                dates = []
                r2s = []
                for date_r2_tuple in self.r2_history[pc]:
                    if isinstance(date_r2_tuple, tuple) and len(date_r2_tuple) == 2:
                        dates.append(date_r2_tuple[0])
                        r2s.append(date_r2_tuple[1])
                    elif isinstance(date_r2_tuple, list) and len(date_r2_tuple) == 2:
                        dates.append(date_r2_tuple[0])
                        r2s.append(date_r2_tuple[1])

                if len(dates) > 0 and len(r2s) > 0:
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
        else:
            print("No data available for plotting R² history.")
            plt.close(fig)

    def compute_centrality_matrix(self, returns):
        corr_matrix = returns.corr()
        eigenvalues, eigenvectors = la.eigh(corr_matrix)
        idx = np.argmax(eigenvalues)
        u_centrality = eigenvectors[:, idx]  # Don't take absolute value yet
        lambda_max = eigenvalues[idx]

        # Eigenvalue verification
        ev_check = np.allclose(corr_matrix.to_numpy() @ eigenvectors[:, idx], lambda_max * eigenvectors[:, idx],
                               atol=1e-6)

        # The centrality weighting should be the first eigenvector values
        # Take absolute value to ensure positive weights
        u_centrality = np.abs(u_centrality)

        # Normalize the centrality vector to have mean=1 and std=0.13 as specified
        u_scaled = u_centrality / np.std(u_centrality) * 0.13
        u_centrality = u_scaled + (1 - np.mean(u_scaled))

        return corr_matrix.to_numpy(), u_centrality

    def compute_predictions(self, pc_series, factor_changes, regression_results, rebalance_date, V, u_centrality):
        """
        CORRECTED weekly predictions that properly scales PC movements while preserving factor regression logic.

        Key insight: The regression predicts weekly PC movements directly in the PC units.
        We don't need to multiply by historical std again since the regression is already trained on PC values.
        """
        if not regression_results:
            print("ERROR: No regression results available")
            return None, None, None, None, None

        k = V.shape[1]
        print(f"\nDEBUG - Starting weekly compute_predictions for {rebalance_date.date()}")
        print(f"DEBUG - Number of PCs (k): {k}")

        # Step 1: Get historical PC standard deviation (for reference only, not for scaling)
        print(f"\nStep 1: Computing Weekly Historical PC Standard Deviation (for reference)")
        try:
            rolling_std = pc_series.rolling(window=self.lookback, min_periods=self.lookback // 2).std(ddof=1)
            sigma_pc_vec = rolling_std.loc[rebalance_date].values

            if np.any(np.isnan(sigma_pc_vec)):
                print("WARNING: NaN values found in sigma_pc_vec, using fallback calculation")
                pc_window = pc_series.loc[:rebalance_date]
                sigma_pc_vec = pc_window.std(ddof=1).values

        except (KeyError, IndexError) as e:
            print(f"ERROR in Step 1: {e}")
            return None, None, None, None, None

        print(f"DEBUG - sigma_pc_vec (Weekly Historical PC Std): {sigma_pc_vec}")

        # Step 2: Predict weekly PC movement using your existing regression logic
        print(f"\nStep 2: Computing Weekly PC Movement Predictions")
        delta_pc_pred_week = np.zeros(k)

        for pc_idx, pc in enumerate(pc_series.columns):
            if pc in regression_results:
                res = regression_results[pc]
                best_factors = res['factors']
                print(f"\nDEBUG - {pc} (index {pc_idx}):")

                if not best_factors:
                    delta_pc_pred_week[pc_idx] = 0.0
                    continue

                try:
                    # Get current weekly factor changes - this is your existing logic
                    if hasattr(res, 'scaler_X') and res['scaler_X'] is not None:
                        # Use the scaler from the weekly regression
                        X_current_raw = factor_changes.loc[rebalance_date, best_factors].to_numpy()
                        X_current = res['scaler_X'].transform(X_current_raw.reshape(1, -1))[0]
                    else:
                        # Fallback to raw values
                        X_current = factor_changes.loc[rebalance_date, best_factors].to_numpy()

                    # Your existing regression prediction logic
                    # This predicts weekly PC movement directly in PC units
                    delta_pc_pred_week[pc_idx] = res['alpha'] + np.dot(list(res['beta'].values()), X_current)
                    print(f"DEBUG - Weekly PC prediction (in PC units): {delta_pc_pred_week[pc_idx]:.6f}")

                except KeyError as e:
                    print(f"ERROR - Could not find weekly factor data for {rebalance_date}: {e}")
                    delta_pc_pred_week[pc_idx] = 0.0
            else:
                delta_pc_pred_week[pc_idx] = 0.0

        print(f"DEBUG - delta_pc_pred_week (direct from regression): {delta_pc_pred_week}")

        # Step 3: CRITICAL FIX - Convert PC predictions to stock returns correctly
        # The regression gives us weekly PC movements in PC units
        # Transform these directly to stock space using PCA loadings
        print(f"\nStep 3: Converting Weekly PC Movements to Stock Returns")
        print(f"DEBUG - Using formula: r_hat = V @ delta_pc_pred_week")

        # Direct transformation - no additional scaling needed
        r_hat_base = V @ delta_pc_pred_week
        print(f"DEBUG - r_hat_base (weekly stock returns): {r_hat_base}")
        print(f"DEBUG - r_hat_base range: [{np.min(r_hat_base):.6f}, {np.max(r_hat_base):.6f}]")

        # Sanity check: weekly returns should typically be in range [-0.3, 0.3] (±30%)
        if np.max(np.abs(r_hat_base)) > 1.0:
            print(f"WARNING: Weekly returns seem too large (max abs: {np.max(np.abs(r_hat_base)):.6f})")
            print("This suggests a scaling issue in the PC predictions or PCA transformation")

        # Step 4: Apply centrality weighting (your existing logic)
        print(f"\nStep 4: Applying Centrality Weighting")
        print(f"DEBUG - u_centrality mean: {np.mean(u_centrality):.6f}, std: {np.std(u_centrality):.6f}")

        r_hat_weighted = r_hat_base * u_centrality
        print(f"DEBUG - r_hat_weighted (final weekly returns): {r_hat_weighted}")
        print(f"DEBUG - r_hat_weighted range: [{np.min(r_hat_weighted):.6f}, {np.max(r_hat_weighted):.6f}]")

        # Step 5: Additional validation and clipping
        print(f"\nStep 5: Final Validation and Clipping")

        # Clip extreme predictions to reasonable weekly return range
        r_hat_weighted = np.clip(r_hat_weighted, -0.5, 0.5)  # ±50% weekly return cap

        # Final summary
        print(f"\nSUMMARY for {rebalance_date.date()}:")
        print(f"Weekly PC movements (from regression):")
        for i in range(min(5, k)):
            pc_name = f"PC_{i + 1}"
            print(f"  {pc_name}: σ_hist={sigma_pc_vec[i]:.6f}, Δ_pred={delta_pc_pred_week[i]:.6f}")

        print(f"Final weekly stock predictions (should be reasonable ±30% range):")
        for stock, ret in zip(self.stocks, r_hat_weighted):
            print(f"  {stock}: {ret:.6f} ({ret * 100:.2f}%)")

        # For backward compatibility - but note these are now properly scaled
        predicted_pct_change_vec = delta_pc_pred_week  # These are already in the right units

        return sigma_pc_vec, delta_pc_pred_week, predicted_pct_change_vec, r_hat_base, r_hat_weighted

    def compute_actual_returns(self, rebalance_date):
        """
        Compute actual stock returns for the same period as predictions.
        Returns cumulative log returns for each stock over the rebalance period.
        """
        try:
            # Get period length (same as used in predictions)
            current_idx = list(self.rebalance_dates).index(rebalance_date)
            if current_idx < len(self.rebalance_dates) - 1:
                next_rebalance_date = self.rebalance_dates[current_idx + 1]
                period_length = (next_rebalance_date - rebalance_date).days
                end_date = next_rebalance_date
            else:
                period_length = 5
                # For the last rebalance, get approximately 5 trading days after
                future_dates = self.data.index[self.data.index > rebalance_date]
                if len(future_dates) == 0:
                    return None, 0
                # Try to get close to the period length in trading days
                end_date = future_dates[min(period_length - 1, len(future_dates) - 1)]
                period_length = (end_date - rebalance_date).days
        except (ValueError, IndexError):
            period_length = 5
            future_dates = self.data.index[self.data.index > rebalance_date]
            if len(future_dates) == 0:
                return None, 0
            end_date = future_dates[min(4, len(future_dates) - 1)]
            period_length = (end_date - rebalance_date).days

        print(f"\nDEBUG - Computing actual returns from {rebalance_date.date()} to {end_date.date()}")
        print(f"DEBUG - Period length: {period_length} days")

        try:
            start_price = self.data.loc[rebalance_date]
            end_price = self.data.loc[end_date]

            # Compute cumulative log return: log(P_end / P_start)
            actual_returns = np.log(end_price / start_price).values

            print(f"DEBUG - Actual returns range: [{np.min(actual_returns):.6f}, {np.max(actual_returns):.6f}]")
            print(f"DEBUG - Sample actual returns: {actual_returns[:5]}")

            return actual_returns, period_length

        except KeyError as e:
            print(f"ERROR - Could not find price data: {e}")
            return None, 0

    def validate_step4(self):
        """
        Validate step 4: Compute actual returns and create predicted vs actual comparison.
        """
        print("=" * 80)
        print("VALIDATION STEP 4: PREDICTED VS ACTUAL RETURNS")
        print("=" * 80)

        validation_results = []
        predicted_actual_pairs = []

        # Process only the last 3 rebalance dates for detailed debugging
        debug_dates = self.rebalance_dates[-3:] if len(self.rebalance_dates) >= 3 else self.rebalance_dates

        for i, rebalance_date in enumerate(self.rebalance_dates):
            if i % 50 == 0 or rebalance_date in debug_dates:  # Show progress every 50 dates
                print(f"\n{'=' * 60}")
                print(f"PROCESSING REBALANCE DATE {i + 1}/{len(self.rebalance_dates)}: {rebalance_date.date()}")
                print(f"{'=' * 60}")

            # Get data and compute all components
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

            # Compute predictions with detailed debugging for last 3 dates
            if rebalance_date in debug_dates:
                print(f"\nDETAILED DEBUG for {rebalance_date.date()}:")
                print(f"PC explained variance: {explained_var[:5]}")
                print(f"Regression results available for PCs: {list(regression_results.keys())}")

            sigma_pc_vec, delta_pc_pred_vec, pc_movement_weighted, r_hat, r_hat_weighted = self.compute_predictions(
                pc_series, factor_changes, regression_results, rebalance_date, V, u_centrality
            )

            # Compute actual returns for the same period
            actual_returns, actual_period = self.compute_actual_returns(rebalance_date)

            # Store results and compute comparison metrics
            correlation = np.nan
            mse = np.nan
            mae = np.nan

            if r_hat_weighted is not None and actual_returns is not None:
                if rebalance_date in debug_dates or i % 50 == 0:
                    print(f"\nCOMPARISON for {rebalance_date.date()}:")
                    print(f"Prediction period: {actual_period} days")
                    print(f"Predicted returns range: [{np.min(r_hat_weighted):.6f}, {np.max(r_hat_weighted):.6f}]")
                    print(f"Actual returns range: [{np.min(actual_returns):.6f}, {np.max(actual_returns):.6f}]")

                correlation = np.corrcoef(r_hat_weighted, actual_returns)[0, 1] if len(r_hat_weighted) > 1 else np.nan
                mse = np.mean((r_hat_weighted - actual_returns) ** 2)
                mae = np.mean(np.abs(r_hat_weighted - actual_returns))

                if rebalance_date in debug_dates:
                    print(f"Predicted vs Actual correlation: {correlation:.4f}")
                    print(f"Mean Squared Error: {mse:.6f}")
                    print(f"Mean Absolute Error: {mae:.6f}")

                    # Show individual stock comparisons for debug dates
                    print(f"\nINDIVIDUAL STOCK COMPARISON:")
                    print(f"{'Stock':<6} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
                    print("-" * 45)
                    for stock, pred, actual in zip(self.stocks, r_hat_weighted, actual_returns):
                        error = pred - actual
                        print(f"{stock:<6} {pred:>10.6f} {actual:>10.6f} {error:>10.6f}")

                # Store for scatter plot
                for stock_idx, (pred, actual) in enumerate(zip(r_hat_weighted, actual_returns)):
                    predicted_actual_pairs.append({
                        'date': rebalance_date,
                        'stock': self.stocks[stock_idx],
                        'predicted': pred,
                        'actual': actual,
                        'period_days': actual_period
                    })

            validation_results.append({
                'date': rebalance_date,
                'r_hat_weighted': r_hat_weighted.tolist() if r_hat_weighted is not None else None,
                'actual_returns': actual_returns.tolist() if actual_returns is not None else None,
                'actual_days': actual_period,
                'correlation': correlation,
                'mse': mse,
                'mae': mae
            })

        # Create comprehensive scatter plot and statistics
        if predicted_actual_pairs:
            print(f"\n{'=' * 60}")
            print("CREATING SCATTER PLOT AND FINAL STATISTICS")
            print(f"{'=' * 60}")

            df_pairs = pd.DataFrame(predicted_actual_pairs)

            # Overall statistics
            overall_corr = np.corrcoef(df_pairs['predicted'], df_pairs['actual'])[0, 1]
            overall_mse = np.mean((df_pairs['predicted'] - df_pairs['actual']) ** 2)
            overall_mae = np.mean(np.abs(df_pairs['predicted'] - df_pairs['actual']))

            print(f"OVERALL STATISTICS:")
            print(f"Total predictions: {len(df_pairs)}")
            print(f"Overall correlation: {overall_corr:.4f}")
            print(f"Overall MSE: {overall_mse:.6f}")
            print(f"Overall MAE: {overall_mae:.6f}")
            print(f"Predicted returns range: [{df_pairs['predicted'].min():.4f}, {df_pairs['predicted'].max():.4f}]")
            print(f"Actual returns range: [{df_pairs['actual'].min():.4f}, {df_pairs['actual'].max():.4f}]")

            # Create scatter plot
            plt.figure(figsize=(12, 8))

            # Color by stock for sample of stocks to avoid too much clutter
            sample_stocks = self.stocks[:10] if len(self.stocks) > 10 else self.stocks
            colors = plt.cm.Set3(np.linspace(0, 1, len(sample_stocks)))

            for i, stock in enumerate(sample_stocks):
                stock_data = df_pairs[df_pairs['stock'] == stock]
                plt.scatter(stock_data['predicted'], stock_data['actual'],
                            color=colors[i], label=stock, alpha=0.6, s=30)

            # Perfect prediction line
            min_val = min(df_pairs['predicted'].min(), df_pairs['actual'].min())
            max_val = max(df_pairs['predicted'].max(), df_pairs['actual'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Formatting
            plt.xlabel('Predicted Returns (Weighted)', fontsize=12)
            plt.ylabel('Actual Returns', fontsize=12)
            plt.title(f'Predicted vs Actual Returns\nCorrelation: {overall_corr:.4f}, MSE: {overall_mse:.6f}',
                      fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Period length analysis
            print(f"\nPERIOD LENGTH ANALYSIS:")
            period_groups = df_pairs.groupby('period_days')
            for period, group in period_groups:
                if len(group) > 10:  # Only show periods with sufficient data
                    corr = np.corrcoef(group['predicted'], group['actual'])[0, 1]
                    print(f"  {period} days: {len(group)} predictions, correlation: {corr:.4f}")

        else:
            print("No data available for predicted vs actual comparison.")

        print(f"\n{'=' * 60}")
        print("VALIDATION STEP 4 COMPLETE")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
    start_date = '2022-01-01'
    end_date = '2025-08-22'
    strategy = PCAFactorStrategy(stocks, start_date, end_date, LR_lookback=150)
    strategy.download_data()
    strategy.validate_step3()
    strategy.validate_step4()

