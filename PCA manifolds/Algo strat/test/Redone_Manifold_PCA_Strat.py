import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import pandas.tseries.offsets as offsets
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class PCAFactorStrategy:
    def __init__(self, start_date, end_date, rebalance_frequency='W-FRI', lookback=252,
                 min_trading_days=100, num_pcs=5, centrality_std=0.13,
                 regression_lookback=52, min_regression_weeks=42):
        """
        Initialize the PCAFactorStrategy with configurable parameters.

        Parameters:
        - start_date: Start date for data (str, e.g., '2022-01-01')
        - end_date: End date for data (str, e.g., '2025-08-22')
        - rebalance_frequency: Frequency for rebalancing (str, e.g., 'W-FRI')
        - lookback: Number of trading days for PCA/centrality (int, default 252)
        - min_trading_days: Minimum valid trading days for PCA/centrality (int, default 100)
        - num_pcs: Number of principal components (int, default 5)
        - centrality_std: Standard deviation for centrality vector (float, default 0.13)
        - regression_lookback: Number of weeks for regression training (int, default 52)
        - min_regression_weeks: Minimum weeks required for regression (int, default 42)
        """
        self.stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.lookback = lookback
        self.min_trading_days = min_trading_days
        self.num_pcs = num_pcs
        self.centrality_std = centrality_std
        self.regression_lookback = regression_lookback
        self.min_regression_weeks = min_regression_weeks
        self.factors = ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']
        self.rotation_pairs = {
            "Growth vs Value": ("VUG", "VTV"),
            "Large vs Small Cap": ("SPY", "IWM"),
            "Tech vs Market": ("XLK", "SPY"),
            "Financials vs Market": ("XLF", "SPY"),
            "Banking vs Financials": ("KBE", "XLF"),
            "Regional vs Banks": ("KRE", "KBE")
        }
        self.momentum_factors = {
            "High vs Low Beta": ("SPHB", "SPLV"),
            "Momentum vs Anti-momentum": ("MTUM", "VMOT"),
            "Quality vs Junk": ("QUAL", "SJNK")
        }
        self.macro_factors = {
            "Dollar Strength": ("UUP", "UDN"),
            "Inflation Expectation": ("SCHP", "VTEB"),
            "Credit Spread": ("LQD", "HYG"),
            "Yield Curve": ("SHY", "TLT"),
            "Real vs Nominal": ("VTEB", "VGIT")
        }
        self.sector_rotation_factors = {
            "Cyclical vs Defensive": ("XLI", "XLP"),
            "Risk-on vs Risk-off": ("XLY", "XLRE"),
            "Energy vs Utilities": ("XLE", "XLU"),
            "Healthcare vs Tech": ("XLV", "XLK"),
            "Materials vs Staples": ("XLB", "XLP")
        }
        self.volatility_factors = {
            "Vol Surface": ("VXX", "SVXY"),
            "Equity vs Bond Vol": ("^VIX", "^MOVE")
        }
        self.all_factor_categories = {
            **self.rotation_pairs,
            **self.momentum_factors,
            **self.macro_factors,
            **self.sector_rotation_factors,
            **self.volatility_factors
        }
        self.stock_data = None
        self.factor_data = None
        self.rotation_data = None
        self.rebalance_dates = None
        self.rebalance_data = {}
        self.regression_results = {}
        # Store predictions and actual PC returns by next rebalance date
        self.predictions = {i: {} for i in range(num_pcs)}
        self.actuals = {i: {} for i in range(num_pcs)}

    def download_data(self):
        """Download stock and factor data, compute weekly factor returns, and set rebalance dates."""
        nominal_start = pd.to_datetime(self.start_date)
        earliest_data_start = nominal_start - offsets.BDay(max(self.lookback + 15, self.regression_lookback * 5 + 10))

        raw_stock_data = yf.download(self.stocks, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        self.stock_data = raw_stock_data['Close'] if isinstance(raw_stock_data.columns,
                                                                pd.MultiIndex) else raw_stock_data
        self.stock_data = self.stock_data.dropna(axis=0, how='any')

        raw_factor_data = yf.download(self.factors, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        factor_data_daily = raw_factor_data['Close'] if isinstance(raw_factor_data.columns,
                                                                   pd.MultiIndex) else raw_factor_data
        factor_data_daily = factor_data_daily.dropna(axis=1, how='all').dropna(axis=0, how='any')
        self.factor_data = factor_data_daily.resample(self.rebalance_frequency).last()
        self.factor_data = self.compute_log_returns(self.factor_data)

        all_factor_tickers = list(set(sum(self.all_factor_categories.values(), ())))
        raw_rotation_data = yf.download(all_factor_tickers, start=earliest_data_start, end=self.end_date,
                                        auto_adjust=True)
        rotation_data_daily = raw_rotation_data['Close'] if isinstance(raw_rotation_data.columns,
                                                                       pd.MultiIndex) else raw_rotation_data
        rotation_data_daily = rotation_data_daily.dropna(axis=1, how='all').dropna(axis=0, how='any')
        self.rotation_data = rotation_data_daily.resample(self.rebalance_frequency).last()

        computed_factors = self.compute_all_factor_categories()
        if not computed_factors.empty:
            common_dates = self.factor_data.index.intersection(computed_factors.index)
            self.factor_data = self.factor_data.loc[common_dates]
            computed_factors = computed_factors.loc[common_dates]
            self.factor_data = pd.concat([self.factor_data, computed_factors], axis=1)

        if self.factor_data.empty or len(self.factor_data.columns) == 0:
            raise ValueError("No valid factor data available")

        self.factors = list(self.factor_data.columns)

        all_dates = self.stock_data.index
        rebalance_dates = self.stock_data.resample(self.rebalance_frequency).last().index
        self.rebalance_dates = rebalance_dates[rebalance_dates.isin(all_dates)]
        first_possible_rebalance = all_dates[all_dates >= (nominal_start + offsets.BDay(self.lookback))][0]
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates >= first_possible_rebalance]

        print(f"Stock data shape: {self.stock_data.shape}, Stocks: {len(self.stocks)}")
        print(f"Factor data shape: {self.factor_data.shape}, Total factors: {len(self.factors)}")
        print(f"Rebalance dates: {len(self.rebalance_dates)}")
        print(f"Successfully added factor categories: {len(self.all_factor_categories)}")

    def compute_log_returns(self, prices):
        """Compute log returns from prices."""
        prices_clean = prices.where(prices > 0)
        returns = np.log(prices_clean / prices_clean.shift(1)).dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def compute_all_factor_categories(self):
        """Compute factor categories as differences in weekly log returns."""
        if not hasattr(self, 'rotation_data') or self.rotation_data.empty:
            print("Warning: No factor data available")
            return pd.DataFrame()

        all_factors = pd.DataFrame(index=self.rotation_data.index)
        factor_returns = self.compute_log_returns(self.rotation_data)

        successful_factors = []
        failed_factors = []

        for factor_name, (asset1, asset2) in self.all_factor_categories.items():
            if asset1 in factor_returns.columns and asset2 in factor_returns.columns:
                all_factors[factor_name] = factor_returns[asset1] - factor_returns[asset2]
                successful_factors.append(factor_name)
            else:
                missing_assets = [asset for asset in [asset1, asset2] if asset not in factor_returns.columns]
                failed_factors.append((factor_name, missing_assets))

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

        return all_factors

    def compute_pca_for_rebalance(self, rebalance_date):
        """Compute PCA loadings matrix for a rebalance date."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None, None

        end_idx = self.stock_data.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback + 1)
        if end_idx - start_idx + 1 < self.min_trading_days:
            print(
                f"Warning: Insufficient data for {rebalance_date.date()} ({end_idx - start_idx + 1} days < {self.min_trading_days})")
            return None, None

        prices = self.stock_data.iloc[start_idx:end_idx + 1]
        returns = self.compute_log_returns(prices)
        if len(returns) < self.min_trading_days:
            print(f"Warning: Insufficient valid returns for {rebalance_date.date()} ({len(returns)} days)")
            return None, None

        returns_std = (returns - returns.mean()) / returns.std()
        returns_std = returns_std.dropna(axis=1, how='any')

        cov_matrix = returns_std.T @ returns_std / (len(returns_std) - 1)

        pca = PCA(n_components=min(self.num_pcs, len(self.stocks)))
        pca.fit(returns_std)
        loadings = pca.components_.T
        explained_variance = pca.explained_variance_ratio_

        for i in range(loadings.shape[1]):
            if np.sum(loadings[:, i]) < 0:
                loadings[:, i] = -loadings[:, i]

        return loadings, explained_variance

    def compute_centrality_for_rebalance(self, rebalance_date):
        """Compute centrality vector for a rebalance date."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None

        end_idx = self.stock_data.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback + 1)
        if end_idx - start_idx + 1 < self.min_trading_days:
            print(f"Warning: Insufficient data for centrality at {rebalance_date.date()}")
            return None

        prices = self.stock_data.iloc[start_idx:end_idx + 1]
        returns = self.compute_log_returns(prices)
        if len(returns) < self.min_trading_days:
            print(f"Warning: Insufficient valid returns for centrality at {rebalance_date.date()}")
            return None

        corr_matrix = returns.corr()

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        max_eigen_idx = np.argmax(eigenvalues)
        centrality_vector = eigenvectors[:, max_eigen_idx]

        centrality_vector = centrality_vector / centrality_vector.std() * self.centrality_std
        centrality_vector = centrality_vector - centrality_vector.mean() + 1.0

        return centrality_vector

    def compute_actual_returns_for_rebalance(self, rebalance_date):
        """Compute actual weekly stock returns for a rebalance date."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None

        rebalance_idx = self.rebalance_dates.get_loc(rebalance_date)
        if rebalance_idx + 1 >= len(self.rebalance_dates):
            print(f"Warning: No next rebalance date for {rebalance_date.date()}")
            return None

        next_rebalance_date = self.rebalance_dates[rebalance_idx + 1]
        if next_rebalance_date not in self.stock_data.index:
            next_dates = self.stock_data.index[self.stock_data.index > rebalance_date]
            if len(next_dates) == 0:
                print(f"Warning: No valid next date for {rebalance_date.date()}")
                return None
            next_rebalance_date = next_dates[0]

        try:
            start_price = self.stock_data.loc[rebalance_date]
            end_price = self.stock_data.loc[next_rebalance_date]
            returns = np.log(end_price / start_price).replace([np.inf, -np.inf], np.nan)
            if returns.isna().any():
                print(f"Warning: Missing returns for {rebalance_date.date()} to {next_rebalance_date.date()}")
                return None
            return returns.values
        except KeyError as e:
            print(f"Warning: Missing price data for {rebalance_date.date()} or {next_rebalance_date.date()}: {e}")
            return None


    def train_regression_models(self, rebalance_date):
        """Train Linear Regression models for each PC."""
        pc_returns, next_pc_returns = self.compute_weekly_pc_returns(rebalance_date)
        if pc_returns is None:
            print(f"Warning: Cannot train regression for {rebalance_date.date()} (insufficient data)")
            return None

        factor_returns = self.factor_data.loc[self.factor_data.index <= rebalance_date]
        factor_returns = factor_returns.iloc[-self.regression_lookback:]
        common_dates = pc_returns.index.intersection(factor_returns.index)
        if len(common_dates) < self.min_regression_weeks:
            print(f"Warning: Insufficient common dates for {rebalance_date.date()} ({len(common_dates)} weeks)")
            return None

        X = factor_returns.loc[common_dates].values
        y = pc_returns.loc[common_dates].values

        # Get next rebalance date for storing predictions
        rebalance_idx = self.rebalance_dates.get_loc(rebalance_date)
        next_rebalance_date = self.rebalance_dates[rebalance_idx + 1] if rebalance_idx + 1 < len(
            self.rebalance_dates) else None

        results = {}
        for pc_idx in range(min(self.num_pcs, y.shape[1])):  # Ensure we don't exceed available PCs
            lin_reg = LinearRegression()
            lin_reg.fit(X, y[:, pc_idx])
            train_r2 = r2_score(y[:, pc_idx], lin_reg.predict(X))

            # Initialize prediction and actual values
            pred_value = None
            actual_value = None
            test_error = None

            if next_pc_returns is not None and next_rebalance_date is not None:
                next_factor_returns = self.factor_data.loc[self.factor_data.index > rebalance_date]
                if not next_factor_returns.empty:
                    X_next = next_factor_returns.iloc[0:1].values
                    pred_value = lin_reg.predict(X_next)[0]
                    actual_value = next_pc_returns[pc_idx]
                    epsilon = 1e-10
                    test_error = abs(pred_value - actual_value) / (abs(actual_value) + epsilon)

                    # Store predictions and actuals by current rebalance date (not next)
                    # This ensures we can look back at historical predictions
                    if next_rebalance_date not in self.predictions[pc_idx]:
                        self.predictions[pc_idx][next_rebalance_date] = pred_value
                        self.actuals[pc_idx][next_rebalance_date] = actual_value

            results[pc_idx] = {
                'model': lin_reg,
                'train_r2': train_r2,
                'test_error': test_error,
                'prediction': pred_value,
                'actual': actual_value
            }

        return results

    def compute_rolling_r2(self, pc_idx, rebalance_date, window=10):
        """Compute rolling R^2 for predictions up to (but not including) rebalance_date."""
        # Get dates strictly before rebalance_date to avoid look-ahead bias
        valid_dates = [d for d in self.predictions[pc_idx].keys() if d < rebalance_date]

        if len(valid_dates) < 2:
            return None

        # Sort and take the most recent 'window' predictions
        valid_dates = sorted(valid_dates)[-window:] if len(valid_dates) >= window else sorted(valid_dates)

        if len(valid_dates) < 2:
            return None

        preds = [self.predictions[pc_idx][d] for d in valid_dates]
        actuals = [self.actuals[pc_idx][d] for d in valid_dates]

        # Check for sufficient variance in actuals to compute meaningful R²
        if np.var(actuals) < 1e-10:
            return None

        try:
            r2 = r2_score(actuals, preds)
            # Cap extreme negative R² values for better visualization
            return max(r2, -10.0)
        except:
            return None

    def plot_r2_scores(self):
        """Create two plots: Training R^2 and Testing R^2 over time for each PC."""
        train_r2_data = {i: [] for i in range(self.num_pcs)}
        test_r2_data = {i: [] for i in range(self.num_pcs)}
        dates = []

        for rebalance_date in self.rebalance_dates:
            results = self.regression_results.get(rebalance_date)
            if results:
                dates.append(rebalance_date)
                for pc_idx in range(self.num_pcs):
                    if pc_idx in results:
                        train_r2_data[pc_idx].append(results[pc_idx]['train_r2'])
                        # Compute rolling R² using only past data
                        test_r2 = self.compute_rolling_r2(pc_idx, rebalance_date, window=10)
                        test_r2_data[pc_idx].append(test_r2 if test_r2 is not None else np.nan)
                    else:
                        train_r2_data[pc_idx].append(np.nan)
                        test_r2_data[pc_idx].append(np.nan)

        # Training R^2 Plot
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(train_r2_data[pc_idx])
            if np.any(valid_mask):
                plt.plot(np.array(dates)[valid_mask], np.array(train_r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)

        plt.title("Training R² Scores Over Time", fontsize=14)
        plt.xlabel("Rebalance Date", fontsize=12)
        plt.ylabel("R² Score", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Testing R^2 Plot
        plt.figure(figsize=(12, 6))
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(test_r2_data[pc_idx])
            if np.any(valid_mask):
                plt.plot(np.array(dates)[valid_mask], np.array(test_r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)

        plt.title("Testing R² Scores Over Time (Rolling Window = 10)", fontsize=14)
        plt.xlabel("Rebalance Date", fontsize=12)
        plt.ylabel("R² Score", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(-2, 1)  # Set reasonable y-axis limits
        plt.tight_layout()
        plt.show()

    def compute_weekly_pc_returns(self, rebalance_date):
        """Compute weekly PC returns for the lookback period and the next week."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None, None

        weekly_prices = self.stock_data.resample(self.rebalance_frequency).last()
        weekly_dates = weekly_prices.index[weekly_prices.index <= rebalance_date]
        start_idx = max(0, len(weekly_dates) - self.regression_lookback)
        if len(weekly_dates[start_idx:]) < self.min_regression_weeks:
            print(
                f"Warning: Insufficient weekly data for {rebalance_date.date()} ({len(weekly_dates[start_idx:])} weeks)")
            return None, None

        weekly_stock_returns = self.compute_log_returns(weekly_prices.iloc[start_idx:])

        pca_matrix = self.rebalance_data.get(rebalance_date, {}).get('pca_matrix')
        if pca_matrix is None:
            print(f"Warning: No PCA matrix for {rebalance_date.date()}")
            return None, None

        # Ensure dimensions match
        num_stocks = min(weekly_stock_returns.shape[1], pca_matrix.shape[0])
        num_pcs = min(self.num_pcs, pca_matrix.shape[1])

        pc_returns = weekly_stock_returns.iloc[:, :num_stocks] @ pca_matrix[:num_stocks, :num_pcs]

        actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
        next_pc_returns = None
        if actual_returns is not None:
            # Ensure dimensions match for actual returns too
            actual_returns_array = np.array(actual_returns)[:num_stocks]
            next_pc_returns = np.dot(actual_returns_array, pca_matrix[:num_stocks, :num_pcs])

        return pc_returns, next_pc_returns

    def save_rebalance_data(self):
        """Save rebalance_data to a JSON file."""
        serializable_data = {}
        for date, data in self.rebalance_data.items():
            date_str = date.strftime('%Y-%m-%d')
            serializable_data[date_str] = {
                'pca_matrix': data['pca_matrix'].tolist() if data['pca_matrix'] is not None else None,
                'centrality_vector': data['centrality_vector'].tolist() if data[
                                                                               'centrality_vector'] is not None else None,
                'actual_returns': data['actual_returns'].tolist() if data['actual_returns'] is not None else None,
                'explained_variance': data['explained_variance'].tolist() if data[
                                                                                 'explained_variance'] is not None else None
            }

        with open('rebalance_data.json', 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print("Saved rebalance data to rebalance_data.json")

    def main(self):
        """Main method with regression and plotting."""
        self.download_data()

        for rebalance_date in self.rebalance_dates:
            pca_matrix, explained_variance = self.compute_pca_for_rebalance(rebalance_date)
            centrality_vector = self.compute_centrality_for_rebalance(rebalance_date)
            actual_returns = self.compute_actual_returns_for_rebalance(rebalance_date)

            self.rebalance_data[rebalance_date] = {
                'pca_matrix': pca_matrix,
                'centrality_vector': centrality_vector,
                'actual_returns': actual_returns,
                'explained_variance': explained_variance
            }

            regression_results = self.train_regression_models(rebalance_date)
            if regression_results:
                self.regression_results[rebalance_date] = regression_results

        self.plot_r2_scores()

        print("\nSummary of Factor Data:")
        print(f"Total Factors: {len(self.factors)}")
        print(f"Date Range: {self.factor_data.index[0].date()} to {self.factor_data.index[-1].date()}")
        print(f"Rebalance Frequency: {self.rebalance_frequency}")
        print(f"Rebalance Dates: {len(self.rebalance_dates)}")
        print(f"\nList of Stocks: {self.stocks}")
        print("\nFactor Categories (Weekly Returns):")
        for category, factors in [
            ("Rotation", self.rotation_pairs),
            ("Momentum", self.momentum_factors),
            ("Macro", self.macro_factors),
            ("Sector Rotation", self.sector_rotation_factors),
            ("Volatility", self.volatility_factors)
        ]:
            print(f"  {category}:")
            for factor_name in factors.keys():
                if factor_name in self.factor_data.columns:
                    mean_return = self.factor_data[factor_name].mean()
                    std_return = self.factor_data[factor_name].std()
                    print(f"    {factor_name}: Mean Weekly Return = {mean_return:.6f}, Std = {std_return:.6f}")

        print("\nRebalancing Dates (Showing first/last 5):")
        display_dates = self.rebalance_dates[:5].append(self.rebalance_dates[-5:]) if len(
            self.rebalance_dates) > 10 else self.rebalance_dates
        for i, date in enumerate(display_dates, 1):
            next_date_idx = self.rebalance_dates.get_loc(date) + 1
            days_between = (self.rebalance_dates[next_date_idx] - date).days if next_date_idx < len(
                self.rebalance_dates) else None
            print(f"  {i}. {date.date()} (Days to next: {days_between if days_between else 'N/A'})")

        print("\nActual Weekly Stock Returns, Explained Variance, and Regression R² (First/Last 5 Dates):")
        for date in display_dates:
            data = self.rebalance_data.get(date, {})
            print(f"\n{date.date()}:")
            if data['actual_returns'] is not None:
                print("  Actual Returns:")
                for stock, ret in zip(self.stocks, data['actual_returns']):
                    print(f"    {stock}: {ret:.6f}")
            if data['explained_variance'] is not None:
                print("  Explained Variance Ratios:")
                for i, var in enumerate(data['explained_variance'], 1):
                    print(f"    PC{i}: {var:.4f} ({var * 100:.2f}%)")
            regression = self.regression_results.get(date, {})
            if regression:
                print("  Regression R² Scores:")
                print("    Linear:")
                for pc_idx in range(self.num_pcs):
                    train_r2 = regression[pc_idx]['train_r2']
                    test_error = regression[pc_idx]['test_error']
                    test_r2 = self.compute_rolling_r2(pc_idx, date, window=10)
                    print(
                        f"      PC{pc_idx + 1}: Train R² = {train_r2:.4f}, Test Error = {test_error if test_error is not None else 'N/A'}, Rolling Test R² = {test_r2 if test_r2 is not None else 'N/A'}")
            else:
                print("  No regression data available")

        self.save_rebalance_data()


if __name__ == "__main__":
    strategy = PCAFactorStrategy(
        start_date='2022-01-01',
        end_date='2025-08-22',
        rebalance_frequency='W-FRI',
        lookback=252,
        min_trading_days=100,
        num_pcs=5,
        centrality_std=0.13,
        regression_lookback=52,
        min_regression_weeks=42
    )
    strategy.main()