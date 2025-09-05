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
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, ttest_1samp, kstest
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class PCAFactorStrategy:

    def __init__(self, start_date, end_date, rebalance_frequency='W-FRI', lookback=252,
                 min_trading_days=100, num_pcs=5,
                 regression_lookback=52, min_regression_weeks=42):
        """
        Initialize the PCAFactorStrategy with configurable parameters.
        Parameters:
        - start_date: Start date for data (str, e.g., '2022-01-01')
        - end_date: End date for data (str, e.g., '2025-08-22')
        - rebalance_frequency: Frequency for rebalancing (str, e.g., 'W-FRI')
        - lookback: Number of trading days for PCA (int, default 252)
        - min_trading_days: Minimum valid trading days for PCA (int, default 100)
        - num_pcs: Number of principal components (int, default 5)
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
        self.regression_lookback = regression_lookback
        self.min_regression_weeks = min_regression_weeks
        self.lags = 3  # Number of lags for multiple lookbacks (including lag 0)
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
        self.sector_r2_history = {}  # Store sector R² values for each rebalance date
        self.stock_data = None
        self.factor_data = None
        self.rotation_data = None
        self.rebalance_dates = None
        self.rebalance_data = {}
        self.regression_results = {}
        self.predictions = {i: {} for i in range(num_pcs)}
        self.actuals = {i: {} for i in range(num_pcs)}
        self.actuals_future = {i: {} for i in range(num_pcs)}
        self.sector_proxy = 'XLF'  # Sector proxy for beta calculation
        self.stock_variance_r2_history = {}

        self.initial_capital = 10000  # Starting capital
        self.current_capital = self.initial_capital
        self.portfolio_history = {}  # Track portfolio composition over time
        self.portfolio_values = {}  # Track portfolio value over time
        self.daily_returns = {}  # Track daily returns
        self.optimization_success = {}  # Track optimization success/failure
        self.stock_betas = {}  # Store calculated betas for each rebalance date

    def download_data(self):
        """Download stock and factor data, compute weekly factor returns, and set rebalance dates."""
        nominal_start = pd.to_datetime(self.start_date)
        earliest_data_start = nominal_start - offsets.BDay(max(self.lookback + 15, self.regression_lookback * 5 + 10))
        latest_data_end = pd.to_datetime(self.end_date) + offsets.BDay(10)
        raw_stock_data = yf.download(self.stocks, start=earliest_data_start, end=latest_data_end, auto_adjust=True)
        self.stock_data = raw_stock_data['Close'] if isinstance(raw_stock_data.columns,
                                                                pd.MultiIndex) else raw_stock_data
        self.stock_data = self.stock_data.dropna(axis=0, how='any')
        raw_factor_data = yf.download(self.factors, start=earliest_data_start, end=latest_data_end, auto_adjust=True)
        self.factor_data_daily = raw_factor_data['Close'] if isinstance(raw_factor_data.columns, pd.MultiIndex) else raw_factor_data
        self.factor_data_daily = self.factor_data_daily.dropna(axis=1, how='all').dropna(axis=0, how='any')
        self.factor_data = self.factor_data_daily.resample(self.rebalance_frequency).last()
        self.factor_data = self.compute_returns(self.factor_data)
        all_factor_tickers = list(set(sum(self.all_factor_categories.values(), ())))
        raw_rotation_data = yf.download(all_factor_tickers, start=earliest_data_start, end=latest_data_end,
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

    def compute_returns(self, prices):
        """Compute simple (percent) returns from prices."""
        prices_clean = prices.where(prices > 0)
        returns = prices_clean.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def compute_all_factor_categories(self):
        """Compute factor categories as differences in weekly returns."""
        if not hasattr(self, 'rotation_data') or self.rotation_data.empty:
            print("Warning: No factor data available")
            return pd.DataFrame()
        all_factors = pd.DataFrame(index=self.rotation_data.index)
        factor_returns = self.compute_returns(self.rotation_data)
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
                print(f" {category}: {successful_in_category}")
        if failed_factors:
            print(f"\nFailed to compute {len(failed_factors)} factors (missing data):")
            for factor_name, missing in failed_factors:
                print(f" {factor_name}: Missing {missing}")
        return all_factors

    def compute_pca_for_rebalance(self, rebalance_date):
        """Compute PCA loadings matrix for a rebalance date, retaining stock return scales, and calculate percentage of sector movement explained by each PC."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None, None, None

        end_idx = self.stock_data.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback + 1)

        if end_idx - start_idx + 1 < self.min_trading_days:
            print(
                f"Warning: Insufficient data for {rebalance_date.date()} ({end_idx - start_idx + 1} days < {self.min_trading_days})")
            return None, None, None

        prices = self.stock_data.iloc[start_idx:end_idx + 1]
        returns = self.compute_returns(prices)

        if len(returns) < self.min_trading_days:
            print(f"Warning: Insufficient valid returns for {rebalance_date.date()} ({len(returns)} days)")
            return None, None, None

        # Compute mean and std for rescaling
        returns_mean = returns.mean()
        returns_std = returns.std()
        returns_std = returns_std.where(returns_std > 0, 1e-10)  # Avoid division by zero

        returns_standardized = (returns - returns_mean) / returns_std
        returns_standardized = returns_standardized.dropna(axis=1, how='any')

        cov_matrix = returns_standardized.T @ returns_standardized / (len(returns_standardized) - 1)
        pca = PCA(n_components=min(self.num_pcs, len(self.stocks)))
        pca.fit(returns_standardized)

        loadings = pca.components_.T
        explained_variance = pca.explained_variance_ratio_

        num_stocks = min(len(self.stocks), loadings.shape[0])
        stock_std = returns_std.values[:num_stocks]

        for i in range(loadings.shape[1]):
            if np.sum(loadings[:, i]) < 0:
                loadings[:, i] = -loadings[:, i]

        # Compute sector returns and their variance
        sector_prices = self.factor_data_daily[self.sector_proxy].loc[prices.index]
        sector_returns = self.compute_returns(sector_prices)

        if sector_returns.empty:
            print(f"Warning: No valid sector returns for {rebalance_date.date()}")
        else:
            common_dates = returns.index.intersection(sector_returns.index)
            if len(common_dates) >= self.min_trading_days:
                sector_returns = sector_returns.loc[common_dates].values.flatten()  # Ensure 1D array
                returns_standardized = returns_standardized.loc[common_dates].values  # Convert to NumPy array

                # Compute PC scores
                pc_scores = returns_standardized @ pca.components_.T  # Ensure NumPy array

                if not isinstance(pc_scores, np.ndarray):
                    print(f"Warning: pc_scores is not a NumPy array at {rebalance_date.date()}")
                    pc_scores = np.array(pc_scores)

                # Regress sector returns on PC scores to get R²
                sector_r2 = []
                for pc_idx in range(min(self.num_pcs, len(pca.components_))):
                    pc_scores_pc = pc_scores[:, pc_idx].reshape(-1, 1)  # Ensure 2D array for sklearn
                    if len(pc_scores_pc) != len(sector_returns):
                        print(f"Warning: Mismatch in data length for PC{pc_idx + 1} at {rebalance_date.date()}")
                        sector_r2.append(0.0)
                        continue
                    try:
                        model = LinearRegression().fit(pc_scores_pc, sector_returns)
                        r2 = r2_score(sector_returns, model.predict(pc_scores_pc))
                        sector_r2.append(r2 * 100)  # Convert to percentage
                    except Exception as e:
                        print(f"Warning: Regression failed for PC{pc_idx + 1} at {rebalance_date.date()}: {e}")
                        sector_r2.append(0.0)

                # Store sector R² values
                self.sector_r2_history[rebalance_date] = sector_r2

                # Print percentage of sector movement explained by each PC
                print(
                    f"\nPercentage of sector ({self.sector_proxy}) movement explained by each PC for {rebalance_date.date()}:")
                for pc_idx, r2 in enumerate(sector_r2):
                    print(f" PC{pc_idx + 1}: {r2:.2f}%")
            else:
                print(f"Warning: Insufficient common dates for sector R² calculation at {rebalance_date.date()}")
                self.sector_r2_history[rebalance_date] = [0.0] * self.num_pcs  # Store zeros if calculation fails

        # NEW: Calculate percentage of stock movement variance explained by each PC
        stock_variance_r2 = self.compute_stock_variance_explained(returns_standardized, pca, rebalance_date)
        self.stock_variance_r2_history[rebalance_date] = stock_variance_r2

        return loadings, explained_variance, stock_std

    # Add this new method:
    def compute_stock_variance_explained(self, returns_standardized, pca, rebalance_date):
        """Compute the percentage of cross-sectional stock movement variance explained by each PC."""
        if returns_standardized is None or len(returns_standardized) == 0:
            return [0.0] * self.num_pcs

        # Convert to numpy array if needed
        if isinstance(returns_standardized, pd.DataFrame):
            returns_standardized = returns_standardized.values

        # The explained variance ratio from PCA already tells us how much variance each PC explains
        # across the cross-section of stocks over the entire time period
        explained_variance_ratios = pca.explained_variance_ratio_

        # Convert to percentages
        stock_variance_r2 = []
        for pc_idx in range(min(self.num_pcs, len(explained_variance_ratios))):
            variance_explained = explained_variance_ratios[pc_idx] * 100
            stock_variance_r2.append(variance_explained)

        # Fill remaining PCs with 0 if we have fewer components than num_pcs
        while len(stock_variance_r2) < self.num_pcs:
            stock_variance_r2.append(0.0)

        # Print results
        print(
            f"\nPercentage of cross-sectional stock movement variance explained by each PC for {rebalance_date.date()}:")
        for pc_idx, r2 in enumerate(stock_variance_r2):
            print(f" PC{pc_idx + 1}: {r2:.2f}%")

        return stock_variance_r2

    # Add this new method:
    def plot_stock_variance_r2_over_time(self):
        """Plot the percentage of cross-sectional stock movement variance explained by each PC over time."""
        if not hasattr(self, 'stock_variance_r2_history') or not self.stock_variance_r2_history:
            print("No stock variance R² data available for plotting")
            return

        dates = sorted(self.stock_variance_r2_history.keys())
        r2_data = {i: [] for i in range(self.num_pcs)}

        for date in dates:
            r2_values = self.stock_variance_r2_history[date]
            for pc_idx in range(self.num_pcs):
                r2_value = r2_values[pc_idx] if pc_idx < len(r2_values) else 0.0
                r2_data[pc_idx].append(r2_value)

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        plt.figure(figsize=(12, 6))

        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(r2_data[pc_idx])
            if np.any(valid_mask):
                plt.plot(np.array(dates)[valid_mask], np.array(r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)

        plt.title("Percentage of Cross-Sectional Stock Movement Variance Explained by Each PC Over Time")
        plt.xlabel("Rebalance Date")
        plt.ylabel("Percentage of Variance Explained (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

        # Output average R² over time for each PC
        print("\nAverage Percentage of Cross-Sectional Stock Movement Variance Explained Over Time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.2f}%")

    # Add this new method:
    def print_avg_stock_variance_r2(self):
        """Print the average percentage of cross-sectional stock movement variance explained by each PC over time."""
        if not hasattr(self, 'stock_variance_r2_history') or not self.stock_variance_r2_history:
            print("No stock variance R² data available")
            return

        dates = sorted(self.stock_variance_r2_history.keys())
        r2_data = {i: [] for i in range(self.num_pcs)}

        for date in dates:
            r2_values = self.stock_variance_r2_history[date]
            for pc_idx in range(self.num_pcs):
                r2_value = r2_values[pc_idx] if pc_idx < len(r2_values) else 0.0
                r2_data[pc_idx].append(r2_value)

        # Output average R² over time for each PC
        print("\nAverage Percentage of Cross-Sectional Stock Movement Variance Explained Over Time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.2f}%")


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
            returns = (end_price / start_price) - 1
            returns = returns.replace([np.inf, -np.inf], np.nan)
            if returns.isna().any():
                print(f"Warning: Missing returns for {rebalance_date.date()} to {next_rebalance_date.date()}")
                return None
            # Sanity check: ensure returns are reasonable (e.g., within -50% to +50% weekly)
            if np.any(np.abs(returns) > 0.5):
                print(f"Warning: Unrealistic returns detected for {rebalance_date.date()}: {returns}")
                return None
            return returns.values
        except KeyError as e:
            print(f"Warning: Missing price data for {rebalance_date.date()} or {next_rebalance_date.date()}: {e}")
            return None

    def compute_actuals_future(self):
        """Compute actual PC returns using the PCA matrix from the next rebalance date."""
        for i in range(len(self.rebalance_dates) - 1):
            date = self.rebalance_dates[i]
            next_date = self.rebalance_dates[i + 1]
            data = self.rebalance_data.get(date, {})
            actual_returns = data.get('actual_returns')
            if actual_returns is None:
                continue
            next_data = self.rebalance_data.get(next_date, {})
            pca_matrix = next_data.get('pca_matrix')
            if pca_matrix is None:
                continue
            actual_returns = np.array(actual_returns)
            num_stocks = min(len(actual_returns), pca_matrix.shape[0])
            actual_returns = actual_returns[:num_stocks]
            pca_matrix = pca_matrix[:num_stocks, :]
            for pc_idx in range(self.num_pcs):
                pc_return = np.dot(actual_returns, pca_matrix[:, pc_idx])
                self.actuals_future[pc_idx][next_date] = pc_return

    def compute_rolling_r2(self, pc_idx, rebalance_date, window=10, use_future=False):
        """Compute rolling R^2 for predictions up to (but not including) rebalance_date."""
        actuals_dict = self.actuals_future if use_future else self.actuals
        valid_dates = [d for d in self.predictions[pc_idx].keys() if d < rebalance_date]
        if len(valid_dates) < 2:
            return None
        valid_dates = sorted(valid_dates)[-window:] if len(valid_dates) >= window else sorted(valid_dates)
        if len(valid_dates) < 2:
            return None
        preds = [self.predictions[pc_idx][d] for d in valid_dates]
        actuals = [actuals_dict[pc_idx][d] for d in valid_dates]
        if np.var(actuals) < 1e-10:
            return None
        try:
            r2 = r2_score(actuals, preds)
            return max(r2, -10.0)
        except:
            return None

    def compute_rolling_train_r2(self, pc_idx, rebalance_date, window=10):
        """Compute rolling average training R² up to (but not including) rebalance_date."""
        valid_dates = [d for d in self.regression_results.keys() if d < rebalance_date]
        if len(valid_dates) < 1:
            return None
        valid_dates = sorted(valid_dates)[-window:] if len(valid_dates) >= window else sorted(valid_dates)
        train_r2s = [self.regression_results[d][pc_idx]['train_r2'] for d in valid_dates if
                     pc_idx in self.regression_results[d] and 'train_r2' in self.regression_results[d][pc_idx]]
        if len(train_r2s) < 1:
            return None
        return np.mean(train_r2s)

    def plot_r2_scores(self):
        """Create two plots: Rolling Training R² and Rolling Test R²."""
        train_r2_data = {i: [] for i in range(self.num_pcs)}
        test_r2_data = {i: [] for i in range(self.num_pcs)}
        dates = []
        for rebalance_date in self.rebalance_dates:
            dates.append(rebalance_date)
            for pc_idx in range(self.num_pcs):
                train_r2 = self.compute_rolling_train_r2(pc_idx, rebalance_date, window=10)
                train_r2_data[pc_idx].append(train_r2 if train_r2 is not None else np.nan)
                test_r2 = self.compute_rolling_r2(pc_idx, rebalance_date, window=10, use_future=False)
                test_r2_data[pc_idx].append(test_r2 if test_r2 is not None else np.nan)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # First plot: Rolling Training R²
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(train_r2_data[pc_idx])
            if np.any(valid_mask):
                ax1.plot(np.array(dates)[valid_mask], np.array(train_r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)
        ax1.set_title("Rolling Training R² Over Time", fontsize=14)
        ax1.set_ylabel("R² Score", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(-2, 1)
        # Second plot: Rolling Test R²
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(test_r2_data[pc_idx])
            if np.any(valid_mask):
                ax2.plot(np.array(dates)[valid_mask], np.array(test_r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)
        ax2.set_title("Rolling Test R² Over Time", fontsize=14)
        ax2.set_xlabel("Rebalance Date", fontsize=12)
        ax2.set_ylabel("R² Score", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(-2, 1)
        plt.tight_layout()
        plt.show()
        # Output average R² over time for both plots
        print("Average Rolling Training R² over time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(train_r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.4f}")
        print("Average Rolling Test R² over time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(test_r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.4f}")

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
        weekly_stock_returns = self.compute_returns(weekly_prices.iloc[start_idx:])
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

    def train_regression_models(self, rebalance_date):
        """Train Lasso to select factors, then Linear Regression for predictions."""
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
        X = factor_returns.loc[common_dates]
        X = X.dropna(axis=0, how='any')
        if len(X) < self.min_regression_weeks:
            print(f"Warning: Insufficient data for {rebalance_date.date()} ({len(X)} weeks)")
            return None
        common_dates = X.index
        X_values = X.values
        pc_returns = pc_returns.loc[common_dates]
        # Get next rebalance date for predictions
        rebalance_idx = self.rebalance_dates.get_loc(rebalance_date)
        next_rebalance_date = self.rebalance_dates[rebalance_idx + 1] if rebalance_idx + 1 < len(
            self.rebalance_dates) else None
        X_next = None
        next_factor_returns = self.factor_data.loc[self.factor_data.index > rebalance_date]
        if not next_factor_returns.empty and next_rebalance_date is not None:
            X_next = next_factor_returns.iloc[0:1].values
        results = {}
        n_splits = min(5, len(common_dates) // 10)
        if n_splits < 2:
            n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for pc_idx in range(self.num_pcs):
            y = pc_returns.values[:, pc_idx]
            # Standardize features and target
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_values)
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            # Step 1: Lasso for feature selection
            param_grid = {'alpha': np.logspace(-4, -1, 6)}
            lasso = GridSearchCV(
                estimator=Lasso(fit_intercept=True, max_iter=5000, tol=1e-5),
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=1
            )
            lasso.fit(X_scaled, y_scaled)
            selected_features = lasso.best_estimator_.coef_ != 0
            selected_indices = np.where(selected_features)[0]
            if len(selected_indices) == 0:
                print(
                    f"Warning: No features selected for PC{pc_idx + 1} at {rebalance_date.date()}, using all features")
                selected_indices = np.arange(X_scaled.shape[1])
            # Step 2: Linear Regression on selected features
            X_selected = X_scaled[:, selected_indices]
            model = LinearRegression(fit_intercept=True)
            model.fit(X_selected, y_scaled)
            train_r2 = r2_score(y_scaled, model.predict(X_selected))
            # Compute CV R² and MSE
            cv_r2_scores = []
            cv_mse_scores = []
            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
                temp_model = LinearRegression(fit_intercept=True)
                temp_model.fit(X_train, y_train)
                val_pred = temp_model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                cv_r2_scores.append(val_r2)
                val_mse = mean_squared_error(y_val, val_pred)
                cv_mse_scores.append(val_mse)
            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)
            cv_score = -np.mean(cv_mse_scores)
            # Prediction and actual
            pred_value = None
            actual_value = None
            test_error = None
            if X_next is not None:
                X_next_scaled = scaler_X.transform(X_next)
                X_next_selected = X_next_scaled[:, selected_indices]
                pred_scaled = model.predict(X_next_selected)[0]
                # Rescale prediction to original PC return scale
                pred_value = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                actual_value = next_pc_returns[pc_idx] if next_pc_returns is not None else None
                if actual_value is not None:
                    epsilon = 1e-10
                    test_error = abs(pred_value - actual_value) / (abs(actual_value) + epsilon)
                    # Store predictions and actuals
                    if next_rebalance_date not in self.predictions[pc_idx]:
                        self.predictions[pc_idx][next_rebalance_date] = pred_value
                        self.actuals[pc_idx][next_rebalance_date] = actual_value
            # Coefficients (only for selected features)
            coef_dict = {}
            feature_idx = 0
            for col_idx, col in enumerate(self.factor_data.columns):
                if col_idx in selected_indices:
                    coef_dict[f'{col}_lag0'] = model.coef_[feature_idx]
                    feature_idx += 1
                else:
                    coef_dict[f'{col}_lag0'] = 0.0
            results[pc_idx] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'selected_features': selected_indices,
                'train_r2': train_r2,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'test_error': test_error,
                'prediction': pred_value,
                'actual': actual_value,
                'model_type': 'Linear (Lasso-selected features)',
                'cv_score': cv_score,
                'coefficients': coef_dict
            }
        return results

    def save_rebalance_data(self):
        """Save rebalance_data to a JSON file, including regression coefficients."""
        serializable_data = {}
        for date, data in self.rebalance_data.items():
            date_str = date.strftime('%Y-%m-%d')
            serializable_data[date_str] = {
                'pca_matrix': data['pca_matrix'].tolist() if data['pca_matrix'] is not None else None,
                'actual_returns': data['actual_returns'].tolist() if data['actual_returns'] is not None else None,
                'explained_variance': data['explained_variance'].tolist() if data[
                                                                                 'explained_variance'] is not None else None,
                'regression_coefficients': {
                    str(pc_idx): self.regression_results.get(date, {}).get(pc_idx, {}).get('coefficients', {})
                    for pc_idx in range(self.num_pcs)
                }
            }
        with open('rebalance_data.json', 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print("Saved rebalance data to rebalance_data.json")

    def compute_pc_std(self, rebalance_date):
        """Compute scaling factor based on PC-specific standard deviations."""
        weekly_prices = self.stock_data.resample(self.rebalance_frequency).last()
        weekly_dates = weekly_prices.index[weekly_prices.index <= rebalance_date]
        start_idx = max(0, len(weekly_dates) - self.regression_lookback)
        if len(weekly_dates[start_idx:]) < self.min_regression_weeks:
            print(
                f"Warning: Insufficient weekly data for {rebalance_date.date()} ({len(weekly_dates[start_idx:])} weeks)")
            return None
        weekly_stock_returns = self.compute_returns(weekly_prices.iloc[start_idx:])
        pca_matrix = self.rebalance_data.get(rebalance_date, {}).get('pca_matrix')
        if pca_matrix is None:
            print(f"Warning: No PCA matrix for {rebalance_date.date()}")
            return None
        # Ensure dimensions match
        num_stocks = min(weekly_stock_returns.shape[1], pca_matrix.shape[0])
        weekly_stock_returns = weekly_stock_returns.iloc[:, :num_stocks]
        pca_matrix = pca_matrix[:num_stocks, :self.num_pcs]
        # Compute PC returns
        pc_returns = weekly_stock_returns @ pca_matrix
        # Compute standard deviation for each PC
        sigma = pc_returns.std(ddof=1).values
        sigma = np.where(sigma == 0, 1e-10, sigma)  # Avoid division by zero
        return sigma

    def compute_predicted_pc_movement(self, rebalance_date):
        """Compute predicted PC movements."""
        results = self.regression_results.get(rebalance_date)
        if not results:
            return None
        sigma = self.compute_pc_std(rebalance_date)
        if sigma is None:
            return None
        pred_pct_change_pc = np.zeros(self.num_pcs)
        for pc_idx in range(self.num_pcs):
            pred_value = results[pc_idx].get('prediction')
            if pred_value is not None:
                pred_pct_change_pc[pc_idx] = pred_value
        return pred_pct_change_pc

    def compute_predicted_stock_returns(self, rebalance_date):
        """Compute predicted stock returns using loadings and predicted PC movements."""
        pred_pct_change_pc = self.compute_predicted_pc_movement(rebalance_date)
        if pred_pct_change_pc is None:
            return None

        pca_matrix = self.rebalance_data.get(rebalance_date, {}).get('pca_matrix')
        if pca_matrix is None:
            return None

        num_stocks = min(len(self.stocks), pca_matrix.shape[0])
        num_pcs = min(self.num_pcs, pca_matrix.shape[1])
        pca_matrix = pca_matrix[:num_stocks, :num_pcs]
        pred_pct_change_pc = pred_pct_change_pc[:num_pcs]

        # The PCA matrix already contains the proper loadings that map PC movements to stock movements
        # No additional scaling needed - the predictions are already in the correct PC return scale
        r_hat = pca_matrix @ pred_pct_change_pc

        return r_hat



    def plot_pred_vs_actual_scatter(self):
        """Scatter plot: predicted vs actual weekly returns for each stock and rebalance date."""
        predicted_data = {}
        actual_data = {}
        for rebalance_date in self.rebalance_dates[:-1]:  # Skip last if no actuals
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
            if r_hat is not None and actual_returns is not None:
                predicted_data[rebalance_date] = r_hat * 100  # To percent
                actual_data[rebalance_date] = actual_returns * 100  # To percent
        if not predicted_data:
            print("No data available for scatter plot")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.stocks)))
        all_preds = []
        all_acts = []
        for stock_idx, stock in enumerate(self.stocks):
            preds = [predicted_data[date][stock_idx] for date in predicted_data if
                     len(predicted_data[date]) > stock_idx]
            acts = [actual_data[date][stock_idx] for date in actual_data if len(actual_data[date]) > stock_idx]
            ax.scatter(preds, acts, color=colors[stock_idx], label=stock, alpha=0.6)
            all_preds.extend(preds)
            all_acts.extend(acts)
        # Compute and plot line of best fit
        if all_preds and all_acts:
            model = LinearRegression()
            X = np.array(all_preds).reshape(-1, 1)
            y = np.array(all_acts)
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            x_range = np.linspace(min(all_preds), max(all_preds), 100)
            ax.plot(x_range, model.predict(x_range.reshape(-1, 1)), color='black', linestyle='--',
                    label='Best Fit Line')
            print(f"Slope of the line of best fit: {slope:.4f}")
        ax.set_xlabel('Weekly Predicted Returns (%)')
        ax.set_ylabel('Weekly Actual Returns (%)')
        ax.set_title('Scatter: Predicted vs Actual Stock Returns Across Rebalance Dates (Percent)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Compute and output average R² for the prediction vs actual
        if all_preds:
            overall_r2 = r2_score(all_acts, all_preds)
            print(f"Overall R² between predicted and actual stock returns: {overall_r2:.4f}")

    def plot_predicted_over_time(self):
        """Line plot: predicted stock movement over time for each stock."""
        predicted_data = {}
        for rebalance_date in self.rebalance_dates:
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            if r_hat is not None:
                predicted_data[rebalance_date] = r_hat * 100  # To percent
        if not predicted_data:
            print("No data available for predicted over time plot")
            return
        dates = sorted(predicted_data.keys())
        fig, ax = plt.subplots(figsize=(12, 8))
        for stock_idx, stock in enumerate(self.stocks):
            preds = [predicted_data.get(date, np.full(len(self.stocks), np.nan))[stock_idx] for date in dates]
            ax.plot(dates, preds, marker='o', label=stock, alpha=0.7)
        ax.set_xlabel('Time (Rebalance Dates)')
        ax.set_ylabel('Predicted Stock Movement (%)')
        ax.set_title('Predicted Stock Movement Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_actual_over_time(self):
        """Line plot: actual stock movement over time for each stock."""
        actual_data = {}
        for rebalance_date in self.rebalance_dates[:-1]: # Skip last if no actuals
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
            if actual_returns is not None:
                actual_data[rebalance_date] = actual_returns * 100  # To percent
        if not actual_data:
            print("No data available for actual over time plot")
            return
        dates = sorted(actual_data.keys())
        fig, ax = plt.subplots(figsize=(12, 8))
        for stock_idx, stock in enumerate(self.stocks):
            acts = [actual_data.get(date, np.full(len(self.stocks), np.nan))[stock_idx] for date in dates]
            ax.plot(dates, acts, marker='o', label=stock, alpha=0.7)
        ax.set_xlabel('Time (Rebalance Dates)')
        ax.set_ylabel('Actual Stock Movement (%)')
        ax.set_title('Actual Stock Movement Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_sector_r2_over_time(self):
        """Plot the percentage of sector (XLF) movement explained by each PC over time."""
        if not hasattr(self, 'sector_r2_history') or not self.sector_r2_history:
            print("No sector R² data available for plotting")
            return
        dates = sorted(self.sector_r2_history.keys())
        r2_data = {i: [] for i in range(self.num_pcs)}
        for date in dates:
            r2_values = self.sector_r2_history[date]
            for pc_idx in range(self.num_pcs):
                r2_value = r2_values[pc_idx] if pc_idx < len(r2_values) else 0.0
                r2_data[pc_idx].append(r2_value)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        plt.figure(figsize=(12, 6))
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(r2_data[pc_idx])
            if np.any(valid_mask):
                plt.plot(np.array(dates)[valid_mask], np.array(r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)
        plt.title(f"Percentage of Sector ({self.sector_proxy}) Movement Explained by Each PC Over Time")
        plt.xlabel("Rebalance Date")
        plt.ylabel("Percentage of Variance Explained (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()
        # Output average R² over time for each PC


    def print_avg_sector_r2(self):
        """Print the average percentage of sector (XLF) movement explained by each PC over time."""
        if not hasattr(self, 'sector_r2_history') or not self.sector_r2_history:
            print("No sector R² data available")
            return
        dates = sorted(self.sector_r2_history.keys())
        r2_data = {i: [] for i in range(self.num_pcs)}
        for date in dates:
            r2_values = self.sector_r2_history[date]
            for pc_idx in range(self.num_pcs):
                r2_value = r2_values[pc_idx] if pc_idx < len(r2_values) else 0.0
                r2_data[pc_idx].append(r2_value)
        # Output average R² over time for each PC
        print("\nAverage Percentage of Sector Movement Explained Over Time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.2f}%")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import spearmanr, pearsonr, ttest_1samp, kstest
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def compute_comprehensive_metrics(self):
        """Compute comprehensive mathematical metrics for prediction performance."""

        # Collect all data
        all_predictions = []
        all_actuals = []
        correlations_pearson = []
        correlations_spearman = []
        hit_rates = []
        rank_correlations = []
        maes = []
        rmses = []
        prediction_errors = []

        # Market regime data
        high_vol_periods = []
        low_vol_periods = []
        bull_periods = []
        bear_periods = []

        for rebalance_date in self.rebalance_dates[:-1]:
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')

            if r_hat is not None and actual_returns is not None:
                min_len = min(len(r_hat), len(actual_returns))
                r_hat_trimmed = r_hat[:min_len]
                actual_trimmed = actual_returns[:min_len]

                all_predictions.extend(r_hat_trimmed)
                all_actuals.extend(actual_trimmed)

                # Correlations
                if min_len > 1:
                    pearson_corr = np.corrcoef(r_hat_trimmed, actual_trimmed)[0, 1]
                    spearman_corr = spearmanr(r_hat_trimmed, actual_trimmed)[0]

                    if not np.isnan(pearson_corr):
                        correlations_pearson.append(pearson_corr)
                    if not np.isnan(spearman_corr):
                        correlations_spearman.append(spearman_corr)

                    # Hit rate (directional accuracy)
                    correct_direction = np.sum(np.sign(r_hat_trimmed) == np.sign(actual_trimmed))
                    hit_rate = correct_direction / min_len
                    hit_rates.append(hit_rate)

                    # Rank correlation
                    pred_ranks = stats.rankdata(r_hat_trimmed)
                    actual_ranks = stats.rankdata(actual_trimmed)
                    rank_corr = np.corrcoef(pred_ranks, actual_ranks)[0, 1]
                    if not np.isnan(rank_corr):
                        rank_correlations.append(rank_corr)

                    # Error metrics
                    mae = mean_absolute_error(actual_trimmed, r_hat_trimmed)
                    rmse = np.sqrt(mean_squared_error(actual_trimmed, r_hat_trimmed))
                    maes.append(mae)
                    rmses.append(rmse)

                    errors = r_hat_trimmed - actual_trimmed
                    prediction_errors.extend(errors)

                    # Market regime classification
                    market_return = np.mean(actual_trimmed)
                    market_vol = np.std(actual_trimmed)

                    # Simple regime classification (you can make this more sophisticated)
                    if market_vol > np.median([np.std(self.rebalance_data.get(d, {}).get('actual_returns', []))
                                               for d in self.rebalance_dates[:-1]
                                               if self.rebalance_data.get(d, {}).get('actual_returns') is not None]):
                        high_vol_periods.append((pearson_corr, hit_rate, mae))
                    else:
                        low_vol_periods.append((pearson_corr, hit_rate, mae))

                    if market_return > 0:
                        bull_periods.append((pearson_corr, hit_rate, mae))
                    else:
                        bear_periods.append((pearson_corr, hit_rate, mae))

        return {
            'all_predictions': all_predictions,
            'all_actuals': all_actuals,
            'correlations_pearson': correlations_pearson,
            'correlations_spearman': correlations_spearman,
            'hit_rates': hit_rates,
            'rank_correlations': rank_correlations,
            'maes': maes,
            'rmses': rmses,
            'prediction_errors': prediction_errors,
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'bull_periods': bull_periods,
            'bear_periods': bear_periods
        }

    def compute_factor_exposure_consistency(self):
        """Analyze stability of factor loadings over time."""
        factor_stability = {}

        for pc_idx in range(self.num_pcs):
            factor_usage = {}
            coefficient_history = {}

            for date, results in self.regression_results.items():
                if pc_idx in results and 'coefficients' in results[pc_idx]:
                    for factor, coef in results[pc_idx]['coefficients'].items():
                        if factor not in factor_usage:
                            factor_usage[factor] = 0
                            coefficient_history[factor] = []

                        if abs(coef) > 1e-12:
                            factor_usage[factor] += 1
                        coefficient_history[factor].append(coef)

            # Calculate stability metrics
            total_periods = len(self.regression_results)
            stability_metrics = {}

            for factor in factor_usage:
                usage_rate = factor_usage[factor] / total_periods
                coef_std = np.std(coefficient_history[factor])
                coef_mean = np.mean([abs(c) for c in coefficient_history[factor]])

                stability_metrics[factor] = {
                    'usage_rate': usage_rate,
                    'coefficient_volatility': coef_std,
                    'avg_magnitude': coef_mean,
                    'stability_score': usage_rate * (1 / (1 + coef_std)) if coef_std > 0 else usage_rate
                }

            factor_stability[pc_idx] = stability_metrics

        return factor_stability

    def _print_enhanced_summary_statistics(self):
        """Enhanced summary statistics with comprehensive mathematical analysis."""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE MATHEMATICAL ANALYSIS")
        print("=" * 100)

        # Compute comprehensive metrics
        metrics = self.compute_comprehensive_metrics()

        # 1. CORRELATION ANALYSIS
        print(f"\n1. CORRELATION ANALYSIS:")
        print(
            f" Pearson Correlation: {np.mean(metrics['correlations_pearson']):.4f} ± {np.std(metrics['correlations_pearson']):.4f}")
        print(
            f" Spearman Correlation: {np.mean(metrics['correlations_spearman']):.4f} ± {np.std(metrics['correlations_spearman']):.4f}")
        print(
            f" Rank Correlation: {np.mean(metrics['rank_correlations']):.4f} ± {np.std(metrics['rank_correlations']):.4f}")

        # Information Coefficient (IC) statistics
        ic_mean = np.mean(metrics['correlations_spearman'])
        ic_std = np.std(metrics['correlations_spearman'])
        information_ratio = ic_mean / ic_std if ic_std > 0 else 0
        print(f" Information Ratio (IC/IC_std): {information_ratio:.4f}")

        # Statistical significance
        if len(metrics['correlations_pearson']) > 1:
            t_stat, p_value = ttest_1samp(metrics['correlations_pearson'], 0)
            print(f" T-test (H0: correlation = 0): t={t_stat:.4f}, p={p_value:.4f}")
            print(f" Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")

        # 2. DIRECTIONAL ACCURACY
        print(f"\n2. DIRECTIONAL ACCURACY (HIT RATE):")
        print(f" Overall Hit Rate: {np.mean(metrics['hit_rates']):.4f} ± {np.std(metrics['hit_rates']):.4f}")
        print(f" Random Baseline: 0.5000")
        print(
            f" Hit Rate > 50%: {np.sum(np.array(metrics['hit_rates']) > 0.5) / len(metrics['hit_rates']) * 100:.1f}% of periods")

        # Hit rate by prediction confidence (magnitude quintiles)
        if len(metrics['all_predictions']) > 0:
            pred_magnitudes = np.abs(metrics['all_predictions'])
            quintile_thresholds = np.percentile(pred_magnitudes, [20, 40, 60, 80])

            print(f" Hit Rate by Prediction Confidence:")
            for i, (low, high) in enumerate(zip([0] + list(quintile_thresholds), list(quintile_thresholds) + [np.inf])):
                mask = (pred_magnitudes >= low) & (pred_magnitudes < high)
                if np.sum(mask) > 0:
                    hit_rate_quintile = np.mean(
                        np.sign(np.array(metrics['all_predictions'])[mask]) ==
                        np.sign(np.array(metrics['all_actuals'])[mask])
                    )
                    print(f"   Q{i + 1} ({low:.4f} - {high:.4f}): {hit_rate_quintile:.4f}")

        # 3. ERROR METRICS
        print(f"\n3. ERROR METRICS:")
        print(f" Mean Absolute Error (MAE): {np.mean(metrics['maes']):.6f} ± {np.std(metrics['maes']):.6f}")
        print(f" Root Mean Square Error (RMSE): {np.mean(metrics['rmses']):.6f} ± {np.std(metrics['rmses']):.6f}")

        # Error distribution analysis
        if len(metrics['prediction_errors']) > 0:
            errors = np.array(metrics['prediction_errors'])
            print(f" Error Skewness: {stats.skew(errors):.4f}")
            print(f" Error Kurtosis: {stats.kurtosis(errors):.4f}")

            # Kolmogorov-Smirnov test for normality
            ks_stat, ks_p = kstest(errors, 'norm', args=(np.mean(errors), np.std(errors)))
            print(f" Normality Test (KS): statistic={ks_stat:.4f}, p={ks_p:.4f}")
            print(f" Errors normally distributed: {'Yes' if ks_p > 0.05 else 'No'}")

            # Error consistency (Sharpe ratio of errors)
            if np.std(errors) > 0:
                error_sharpe = np.mean(np.abs(errors)) / np.std(errors)
                print(f" Error Consistency Ratio: {error_sharpe:.4f}")

        # 4. REGIME-DEPENDENT PERFORMANCE
        print(f"\n4. REGIME-DEPENDENT PERFORMANCE:")

        if len(metrics['high_vol_periods']) > 0 and len(metrics['low_vol_periods']) > 0:
            high_vol_corr = np.mean([x[0] for x in metrics['high_vol_periods']])
            low_vol_corr = np.mean([x[0] for x in metrics['low_vol_periods']])
            high_vol_hit = np.mean([x[1] for x in metrics['high_vol_periods']])
            low_vol_hit = np.mean([x[1] for x in metrics['low_vol_periods']])

            print(f" High Volatility Periods:")
            print(f"   Correlation: {high_vol_corr:.4f}")
            print(f"   Hit Rate: {high_vol_hit:.4f}")
            print(f"   Count: {len(metrics['high_vol_periods'])}")

            print(f" Low Volatility Periods:")
            print(f"   Correlation: {low_vol_corr:.4f}")
            print(f"   Hit Rate: {low_vol_hit:.4f}")
            print(f"   Count: {len(metrics['low_vol_periods'])}")

        if len(metrics['bull_periods']) > 0 and len(metrics['bear_periods']) > 0:
            bull_corr = np.mean([x[0] for x in metrics['bull_periods']])
            bear_corr = np.mean([x[0] for x in metrics['bear_periods']])
            bull_hit = np.mean([x[1] for x in metrics['bull_periods']])
            bear_hit = np.mean([x[1] for x in metrics['bear_periods']])

            print(f" Bull Market Periods:")
            print(f"   Correlation: {bull_corr:.4f}")
            print(f"   Hit Rate: {bull_hit:.4f}")
            print(f"   Count: {len(metrics['bull_periods'])}")

            print(f" Bear Market Periods:")
            print(f"   Correlation: {bear_corr:.4f}")
            print(f"   Hit Rate: {bear_hit:.4f}")
            print(f"   Count: {len(metrics['bear_periods'])}")

        # 5. FACTOR EXPOSURE CONSISTENCY
        print(f"\n5. FACTOR EXPOSURE CONSISTENCY:")
        factor_stability = self.compute_factor_exposure_consistency()

        for pc_idx in range(self.num_pcs):
            print(f"\n PC{pc_idx + 1} - Most Stable Factors:")
            if pc_idx in factor_stability:
                stable_factors = sorted(factor_stability[pc_idx].items(),
                                        key=lambda x: x[1]['stability_score'], reverse=True)[:5]

                for factor, metrics in stable_factors:
                    print(f"   {factor}:")
                    print(f"     Usage Rate: {metrics['usage_rate']:.3f}")
                    print(f"     Coefficient Volatility: {metrics['coefficient_volatility']:.6f}")
                    print(f"     Stability Score: {metrics['stability_score']:.3f}")

        # 6. CROSS-SECTIONAL ANALYSIS
        print(f"\n6. CROSS-SECTIONAL ANALYSIS:")

        # Calculate average correlation by stock
        stock_correlations = {stock: [] for stock in self.stocks}

        for rebalance_date in self.rebalance_dates[:-1]:
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')

            if r_hat is not None and actual_returns is not None:
                min_len = min(len(r_hat), len(actual_returns), len(self.stocks))

                for i in range(min_len):
                    if not np.isnan(r_hat[i]) and not np.isnan(actual_returns[i]):
                        stock_correlations[self.stocks[i]].append((r_hat[i], actual_returns[i]))

        print(f" Individual Stock Performance:")
        for stock in self.stocks:
            if len(stock_correlations[stock]) > 10:  # Need sufficient data
                preds, actuals = zip(*stock_correlations[stock])
                if np.std(preds) > 1e-10 and np.std(actuals) > 1e-10:
                    corr = np.corrcoef(preds, actuals)[0, 1]
                    mae = mean_absolute_error(actuals, preds)
                    print(f"   {stock}: Correlation={corr:.3f}, MAE={mae:.6f}, N={len(stock_correlations[stock])}")

    def calculate_stock_betas(self, rebalance_date):
        """Calculate beta for each stock relative to sector proxy (XLF) using 52-week lookback."""
        # Get weekly prices for the lookback period
        weekly_prices = self.stock_data.resample(self.rebalance_frequency).last()
        weekly_dates = weekly_prices.index[weekly_prices.index <= rebalance_date]
        start_idx = max(0, len(weekly_dates) - self.regression_lookback)

        if len(weekly_dates[start_idx:]) < self.min_regression_weeks:
            print(f"Warning: Insufficient weekly data for beta calculation at {rebalance_date.date()}")
            return None

        # Get stock returns for the period
        stock_prices_period = weekly_prices.iloc[start_idx:]
        stock_returns = self.compute_returns(stock_prices_period)

        # Get sector proxy returns
        sector_prices = self.factor_data_daily[self.sector_proxy].resample(self.rebalance_frequency).last()
        sector_prices_period = sector_prices.loc[stock_prices_period.index]
        sector_returns = self.compute_returns(sector_prices_period)

        # Calculate betas
        betas = {}
        for stock in self.stocks:
            if stock in stock_returns.columns:
                common_dates = stock_returns.index.intersection(sector_returns.index)
                if len(common_dates) >= self.min_regression_weeks:
                    stock_ret = stock_returns[stock].loc[common_dates]
                    sector_ret = sector_returns.loc[common_dates]

                    covariance = np.cov(stock_ret, sector_ret)[0, 1]
                    sector_variance = np.var(sector_ret, ddof=1)

                    beta = covariance / sector_variance if sector_variance > 0 else 1.0
                    betas[stock] = beta
                else:
                    betas[stock] = 1.0  # Default beta
            else:
                betas[stock] = 1.0  # Default beta

        return betas

    def optimize_portfolio(self, rebalance_date):
        """Optimize portfolio weights based on predicted returns and constraints."""
        from scipy.optimize import linprog

        # Get predicted returns
        r_hat = self.compute_predicted_stock_returns(rebalance_date)
        if r_hat is None:
            return None, False

        # Get betas
        betas = self.calculate_stock_betas(rebalance_date)
        if betas is None:
            return None, False

        # Store betas for this rebalance date
        self.stock_betas[rebalance_date] = betas

        # Ensure we have the right number of stocks and betas
        n_stocks = min(len(self.stocks), len(r_hat))
        r_hat = r_hat[:n_stocks]
        beta_array = np.array([betas.get(self.stocks[i], 1.0) for i in range(n_stocks)])

        # Set up optimization problem: maximize r_hat^T * w
        # Convert to minimization: minimize -r_hat^T * w
        c = -r_hat  # Coefficients for minimization

        # Constraints:
        # 1. Dollar neutral: sum(w) = 0
        # 2. Beta neutral: sum(beta * w) = 0
        # 3. Exposure constraint: sum(|w|) <= current_capital

        A_eq = np.array([
            np.ones(n_stocks),  # Dollar neutral
            beta_array  # Beta neutral
        ])
        b_eq = np.array([0, 0])

        # Bounds: allow both long and short positions
        # Use exposure constraint through bounds
        max_position = self.current_capital / n_stocks  # Simple position sizing
        bounds = [(-max_position, max_position) for _ in range(n_stocks)]

        # Solve optimization
        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if result.success:
                weights = result.x

                # Apply exposure constraint post-optimization
                total_exposure = np.sum(np.abs(weights))
                if total_exposure > self.current_capital:
                    # Scale down to meet exposure constraint
                    weights = weights * (self.current_capital / total_exposure)

                return weights, True
            else:
                print(f"Optimization failed at {rebalance_date.date()}: {result.message}")
                return None, False

        except Exception as e:
            print(f"Optimization error at {rebalance_date.date()}: {e}")
            return None, False

    def rebalance_portfolio(self, rebalance_date):
        """Execute portfolio rebalancing for a given date."""
        # Get optimal weights
        weights, success = self.optimize_portfolio(rebalance_date)

        if not success or weights is None:
            # Keep previous portfolio if optimization fails
            prev_idx = self.rebalance_dates.get_loc(rebalance_date)
            if prev_idx > 0:
                prev_date = self.rebalance_dates[prev_idx - 1]
                if prev_date in self.portfolio_history:
                    weights = self.portfolio_history[prev_date].copy()
                    print(f"Using previous portfolio weights due to optimization failure")
                else:
                    weights = np.zeros(len(self.stocks))
            else:
                weights = np.zeros(len(self.stocks))
            success = False

        # Store portfolio composition
        n_stocks = min(len(self.stocks), len(weights))
        portfolio_dict = {self.stocks[i]: weights[i] for i in range(n_stocks)}
        self.portfolio_history[rebalance_date] = portfolio_dict
        self.optimization_success[rebalance_date] = success

        # Calculate and store portfolio value and P&L
        if rebalance_date in self.rebalance_dates[:-1]:  # Not the last date
            # Get actual returns for P&L calculation
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')

            if actual_returns is not None:
                # Calculate portfolio return
                portfolio_return = 0
                for i in range(min(len(weights), len(actual_returns))):
                    portfolio_return += weights[i] * actual_returns[i]

                # Update capital
                pnl = portfolio_return
                self.current_capital += pnl

                # Store results
                self.portfolio_values[rebalance_date] = self.current_capital
                self.daily_returns[rebalance_date] = portfolio_return
            else:
                self.portfolio_values[rebalance_date] = self.current_capital
                self.daily_returns[rebalance_date] = 0
        else:
            # For the last date, just record current capital
            self.portfolio_values[rebalance_date] = self.current_capital

        return weights, success

    def plot_portfolio_value(self):
        """Plot portfolio value over time starting from initial capital."""
        if not self.portfolio_values:
            print("No portfolio value data available for plotting")
            return

        dates = sorted(self.portfolio_values.keys())
        values = [self.portfolio_values[date] for date in dates]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, linewidth=2, color='blue', marker='o', markersize=3)
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')

        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Add some statistics to the plot
        final_value = values[-1] if values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        plt.text(0.02, 0.98, f'Total Return: {total_return:.2%}', transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total P&L: ${final_value - self.initial_capital:,.2f}")
        print(f"Total Return: {total_return:.2%}")

    def plot_portfolio_composition(self):
        """Plot portfolio composition over time with longs above zero and shorts below."""
        if not self.portfolio_history:
            print("No portfolio composition data available for plotting")
            return

        dates = sorted(self.portfolio_history.keys())

        # Prepare data
        composition_data = {stock: [] for stock in self.stocks}

        for date in dates:
            portfolio = self.portfolio_history[date]
            for stock in self.stocks:
                composition_data[stock].append(portfolio.get(stock, 0))

        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Separate longs and shorts
        longs_data = {stock: [max(0, val) for val in values] for stock, values in composition_data.items()}
        shorts_data = {stock: [min(0, val) for val in values] for stock, values in composition_data.items()}

        # Colors for each stock
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.stocks)))

        # Plot longs (positive values)
        bottom_long = np.zeros(len(dates))
        for i, stock in enumerate(self.stocks):
            values = longs_data[stock]
            if any(v > 0 for v in values):
                ax.fill_between(dates, bottom_long, bottom_long + values,
                                label=f'{stock} (Long)', alpha=0.7, color=colors[i])
                bottom_long += values

        # Plot shorts (negative values)
        bottom_short = np.zeros(len(dates))
        for i, stock in enumerate(self.stocks):
            values = shorts_data[stock]
            if any(v < 0 for v in values):
                ax.fill_between(dates, bottom_short, bottom_short + values,
                                label=f'{stock} (Short)', alpha=0.7,
                                color=colors[i], hatch='///')
                bottom_short += values

        # Add zero line
        ax.axhline(y=0, color='black', linewidth=1)

        ax.set_title('Portfolio Composition Over Time\n(Longs Above Zero, Shorts Below Zero)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Position Size ($)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def print_portfolio_metrics(self):
        """Print comprehensive portfolio performance metrics."""
        print("\n" + "=" * 80)
        print("PORTFOLIO PERFORMANCE METRICS")
        print("=" * 80)

        if not self.portfolio_values or not self.daily_returns:
            print("No portfolio data available for metrics calculation")
            return

        # Basic metrics
        dates = sorted(self.portfolio_values.keys())
        returns = [self.daily_returns.get(date, 0) for date in dates if date in self.daily_returns]

        if not returns:
            print("No return data available")
            return

        returns = np.array(returns)

        # 1. Basic Performance
        print(f"\n1. BASIC PERFORMANCE:")
        initial_value = self.initial_capital
        final_value = list(self.portfolio_values.values())[-1]
        total_return = (final_value - initial_value) / initial_value
        total_pnl = final_value - initial_value

        print(f" Initial Capital: ${initial_value:,.2f}")
        print(f" Final Capital: ${final_value:,.2f}")
        print(f" Total P&L: ${total_pnl:,.2f}")
        print(f" Total Return: {total_return:.2%}")

        # 2. Risk-Adjusted Returns
        print(f"\n2. RISK-ADJUSTED RETURNS:")
        avg_return = np.mean(returns)
        return_std = np.std(returns, ddof=1)

        # Sharpe Ratio (assuming weekly data, annualize appropriately)
        periods_per_year = 52  # Weekly data
        annualized_return = avg_return * periods_per_year
        annualized_vol = return_std * np.sqrt(periods_per_year)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        print(f" Average Weekly Return: {avg_return:.6f} ({avg_return:.4%})")
        print(f" Weekly Volatility: {return_std:.6f} ({return_std:.4%})")
        print(f" Annualized Return: {annualized_return:.4%}")
        print(f" Annualized Volatility: {annualized_vol:.4%}")
        print(f" Sharpe Ratio: {sharpe_ratio:.4f}")

        # 3. Optimization Success Rate
        print(f"\n3. OPTIMIZATION STATISTICS:")
        successful_optimizations = sum(self.optimization_success.values())
        total_optimizations = len(self.optimization_success)
        success_rate = successful_optimizations / total_optimizations if total_optimizations > 0 else 0

        print(f" Successful Optimizations: {successful_optimizations}/{total_optimizations}")
        print(f" Success Rate: {success_rate:.1%}")

        # 4. Return Distribution
        print(f"\n4. RETURN DISTRIBUTION:")
        profitable_periods = np.sum(returns > 0)
        total_periods = len(returns)
        win_rate = profitable_periods / total_periods if total_periods > 0 else 0

        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

        print(f" Profitable Periods: {profitable_periods}/{total_periods}")
        print(f" Win Rate: {win_rate:.1%}")
        print(f" Average Winning Return: {avg_win:.4%}")
        print(f" Average Losing Return: {avg_loss:.4%}")
        if avg_loss != 0:
            print(f" Win/Loss Ratio: {abs(avg_win / avg_loss):.2f}")

        # 5. Long vs Short Performance
        print(f"\n5. LONG VS SHORT PERFORMANCE:")
        long_returns_by_date = []
        short_returns_by_date = []

        for date in dates[:-1]:  # Skip last date
            if date in self.portfolio_history and date in self.daily_returns:
                portfolio = self.portfolio_history[date]
                actual_returns = self.rebalance_data.get(date, {}).get('actual_returns', [])

                if actual_returns is not None and len(actual_returns) > 0:
                    long_return = 0
                    short_return = 0
                    long_weight = 0
                    short_weight = 0

                    for i, stock in enumerate(self.stocks):
                        if i < len(actual_returns) and stock in portfolio:
                            weight = portfolio[stock]
                            ret = actual_returns[i]

                            if weight > 0:  # Long position
                                long_return += weight * ret
                                long_weight += abs(weight)
                            elif weight < 0:  # Short position
                                short_return += weight * ret
                                short_weight += abs(weight)

                    if long_weight > 0:
                        long_returns_by_date.append(long_return / long_weight)
                    if short_weight > 0:
                        short_returns_by_date.append(short_return / short_weight)

        if long_returns_by_date:
            avg_long_return = np.mean(long_returns_by_date)
            print(f" Average Long-side Return per Period: {avg_long_return:.4%}")
            print(f" Long-side Win Rate: {np.sum(np.array(long_returns_by_date) > 0) / len(long_returns_by_date):.1%}")

        if short_returns_by_date:
            avg_short_return = np.mean(short_returns_by_date)
            print(f" Average Short-side Return per Period: {avg_short_return:.4%}")
            print(
                f" Short-side Win Rate: {np.sum(np.array(short_returns_by_date) > 0) / len(short_returns_by_date):.1%}")

        # 6. Additional Risk Metrics
        print(f"\n6. ADDITIONAL RISK METRICS:")
        if len(returns) > 1:
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns) - 1
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max)
            max_drawdown = np.min(drawdown)

            print(f" Maximum Drawdown: {max_drawdown:.4%}")

            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
            annualized_downside_dev = downside_deviation * np.sqrt(periods_per_year)
            sortino_ratio = annualized_return / annualized_downside_dev if annualized_downside_dev > 0 else 0

            print(f" Downside Deviation (Annualized): {annualized_downside_dev:.4%}")
            print(f" Sortino Ratio: {sortino_ratio:.4f}")

    def main(self):
        """Main function to run the PCA factor strategy with portfolio rebalancing."""
        # Download data
        print("Downloading data...")
        self.download_data()

        # Initialize portfolio tracking
        self.current_capital = self.initial_capital
        self.portfolio_values[self.rebalance_dates[0]] = self.initial_capital

        # Process each rebalance date
        for i, rebalance_date in enumerate(self.rebalance_dates):
            print(f"\nProcessing rebalance date: {rebalance_date.date()}")

            # Find training period start and end dates
            weekly_prices = self.stock_data.resample(self.rebalance_frequency).last()
            weekly_dates = weekly_prices.index[weekly_prices.index <= rebalance_date]
            start_idx = max(0, len(weekly_dates) - self.regression_lookback)

            if len(weekly_dates) > start_idx:
                training_start = weekly_dates[start_idx] - offsets.BDay(1)
                training_end = rebalance_date - offsets.BDay(1)
                print(f"Training period: {training_start.date()} to {training_end.date()}")
            else:
                print(f"Training period: Insufficient data")

            # Find prediction period start and end dates
            rebalance_idx = self.rebalance_dates.get_loc(rebalance_date)
            next_rebalance_date = self.rebalance_dates[rebalance_idx + 1] if rebalance_idx + 1 < len(
                self.rebalance_dates) else None

            if next_rebalance_date is not None:
                print(f"Prediction period: {rebalance_date.date()} to {next_rebalance_date.date()}")
            else:
                print(f"Prediction period: {rebalance_date.date()} (no next rebalance date)")

            # Compute PCA and centrality
            pca_matrix, explained_variance, stock_std = self.compute_pca_for_rebalance(rebalance_date)
            actual_returns = self.compute_actual_returns_for_rebalance(rebalance_date)

            # Store rebalance data
            self.rebalance_data[rebalance_date] = {
                'pca_matrix': pca_matrix,
                'actual_returns': actual_returns,
                'explained_variance': explained_variance,
                'stock_std': stock_std,
            }

            # Train regression models
            self.regression_results[rebalance_date] = self.train_regression_models(rebalance_date)

            # Print results for each PC
            print(f"\n=== Regression Results for {rebalance_date.date()} ===")
            for pc_idx, pc_result in self.regression_results[rebalance_date].items():
                print(f"\nPC{pc_idx + 1} Results:")
                print(f" Train R²: {pc_result['train_r2']:.4f}")
                print(f" CV R² Mean: {pc_result['cv_r2_mean']:.4f} ± {pc_result['cv_r2_std']:.4f}")
                print(f" CV Score (−MSE): {pc_result['cv_score']:.4f}")

                if pc_result['prediction'] is not None and pc_result['actual'] is not None:
                    print(f" Prediction: {pc_result['prediction']:.6f}")
                    print(f" Actual: {pc_result['actual']:.6f}")
                    print(f" Test Error: {pc_result['test_error']:.4%}")
                else:
                    print(" No prediction available for this date.")

                # Print only non-zero coefficients
                print(" Significant Coefficients:")
                for factor, coef in pc_result['coefficients'].items():
                    if abs(coef) > 1e-12:  # tiny threshold to avoid floating-point noise
                        print(f" {factor}: {coef:.6f}")

            # Compute and display predicted vs actual returns
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')

            print("\nPredicted Returns:")
            if r_hat is not None:
                for i, stock in enumerate(self.stocks):
                    if i < len(r_hat):
                        print(f" {stock:4}: {r_hat[i]:8.6f}")
            else:
                print(" No predictions available")

            print("\nActual Returns:")
            if actual_returns is not None:
                for i, stock in enumerate(self.stocks):
                    if i < len(actual_returns):
                        print(f" {stock:4}: {actual_returns[i]:8.6f}")
            else:
                print(" No actual returns available")

            # Execute portfolio rebalancing
            weights, optimization_success = self.rebalance_portfolio(rebalance_date)

            print(f"\nPortfolio Rebalancing for {rebalance_date.date()}:")
            print(f"Optimization Success: {'Yes' if optimization_success else 'No'}")
            print("Portfolio Weights:")
            if weights is not None:
                for i, stock in enumerate(self.stocks):
                    if i < len(weights):
                        position_type = "Long" if weights[i] > 0 else "Short" if weights[i] < 0 else "None"
                        print(f" {stock:4}: {weights[i]:10.2f} ({position_type})")
            else:
                print(" No weights available")

            # Print current portfolio value and P&L
            if rebalance_date in self.portfolio_values:
                current_value = self.portfolio_values[rebalance_date]
                daily_return = self.daily_returns.get(rebalance_date, 0)
                print(f"Portfolio Value: ${current_value:,.2f}")
                print(f"Period P&L: ${daily_return:,.2f}")

        # Compute actual future PC returns
        self.compute_actuals_future()
        #hi
        # Save rebalance data
        self.save_rebalance_data()

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_r2_scores()
        self.plot_pred_vs_actual_scatter()
        self.plot_predicted_over_time()
        self.plot_actual_over_time()
        self.plot_sector_r2_over_time()
        self.print_avg_sector_r2()
        self.plot_stock_variance_r2_over_time()
        self.print_avg_stock_variance_r2()

        # Portfolio-specific visualizations and metrics
        self.plot_portfolio_value()
        self.plot_portfolio_composition()
        self.print_portfolio_metrics()

        # Print enhanced comprehensive summary
        self._print_enhanced_summary_statistics()
    def _print_summary_statistics(self):
        """Print comprehensive summary statistics at the end of processing."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Collect all correlations between r_hat and actual returns
        correlations = []
        valid_predictions = 0
        total_predictions = 0

        for rebalance_date in self.rebalance_dates[:-1]:  # Skip last date
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
            total_predictions += 1

            if r_hat is not None and actual_returns is not None:
                min_len = min(len(r_hat), len(actual_returns))
                if min_len > 1:  # Need at least 2 points for correlation
                    corr = np.corrcoef(r_hat[:min_len], actual_returns[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        valid_predictions += 1

        # 1. Average correlation between predicted and actual returns
        print(f"\n1. PREDICTION ACCURACY:")
        print(f" Average correlation (r_hat vs actual): {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f" Valid predictions: {valid_predictions}/{total_predictions}")
        print(f" Prediction success rate: {valid_predictions / total_predictions * 100:.1f}%")

        # 2. Average R² for each PC regression
        print(f"\n2. REGRESSION MODEL PERFORMANCE:")
        all_train_r2 = {pc: [] for pc in range(self.num_pcs)}
        all_cv_r2 = {pc: [] for pc in range(self.num_pcs)}
        all_test_errors = {pc: [] for pc in range(self.num_pcs)}

        for date, results in self.regression_results.items():
            for pc_idx, pc_result in results.items():
                if isinstance(pc_result, dict):
                    all_train_r2[pc_idx].append(pc_result.get('train_r2', 0))
                    all_cv_r2[pc_idx].append(pc_result.get('cv_r2_mean', 0))
                    if pc_result.get('test_error') is not None:
                        all_test_errors[pc_idx].append(pc_result['test_error'])

        for pc in range(self.num_pcs):
            train_avg = np.mean(all_train_r2[pc]) if all_train_r2[pc] else 0
            cv_avg = np.mean(all_cv_r2[pc]) if all_cv_r2[pc] else 0
            test_avg = np.mean(all_test_errors[pc]) if all_test_errors[pc] else 0
            print(f" PC{pc + 1}: Train R² = {train_avg:.4f}, CV R² = {cv_avg:.4f}, Test Error = {test_avg:.4%}")

        # 3. Most commonly selected features
        print(f"\n3. FEATURE SELECTION ANALYSIS:")
        feature_counts = {pc: {} for pc in range(self.num_pcs)}

        for date, results in self.regression_results.items():
            for pc_idx, pc_result in results.items():
                if isinstance(pc_result, dict) and 'coefficients' in pc_result:
                    for feature, coef in pc_result['coefficients'].items():
                        if abs(coef) > 1e-12:  # Feature was selected
                            if feature not in feature_counts[pc_idx]:
                                feature_counts[pc_idx][feature] = 0
                            feature_counts[pc_idx][feature] += 1

        for pc in range(self.num_pcs):
            print(f"\n PC{pc + 1} - Top 5 Most Selected Features:")
            sorted_features = sorted(feature_counts[pc].items(), key=lambda x: x[1], reverse=True)[:5]
            total_dates = len(self.regression_results)
            for feature, count in sorted_features:
                percentage = count / total_dates * 100
                print(f" {feature}: {count}/{total_dates} times ({percentage:.1f}%)")

        # 4. Model stability metrics
        print(f"\n4. MODEL STABILITY:")
        # Calculate average number of features selected per PC
        avg_features_selected = {pc: [] for pc in range(self.num_pcs)}

        for date, results in self.regression_results.items():
            for pc_idx, pc_result in results.items():
                if isinstance(pc_result, dict) and 'coefficients' in pc_result:
                    selected_count = sum(1 for coef in pc_result['coefficients'].values() if abs(coef) > 1e-12)
                    avg_features_selected[pc_idx].append(selected_count)

        for pc in range(self.num_pcs):
            if avg_features_selected[pc]:
                avg_count = np.mean(avg_features_selected[pc])
                std_count = np.std(avg_features_selected[pc])
                print(f" PC{pc + 1}: Avg features selected = {avg_count:.1f} ± {std_count:.1f}")

        # 5. Prediction magnitude analysis
        print(f"\n5. PREDICTION MAGNITUDE ANALYSIS:")
        pred_magnitudes = []
        actual_magnitudes = []

        for rebalance_date in self.rebalance_dates[:-1]:
            r_hat = self.compute_predicted_stock_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')

            if r_hat is not None and actual_returns is not None:
                min_len = min(len(r_hat), len(actual_returns))
                pred_magnitudes.extend([abs(x) for x in r_hat[:min_len]])
                actual_magnitudes.extend([abs(x) for x in actual_returns[:min_len]])

        if pred_magnitudes and actual_magnitudes:
            print(f" Average predicted magnitude: {np.mean(pred_magnitudes):.6f}")
            print(f" Average actual magnitude: {np.mean(actual_magnitudes):.6f}")
            print(f" Prediction/Actual ratio: {np.mean(pred_magnitudes) / np.mean(actual_magnitudes):.4f}")

        # 6. Stock Movement Variance Analysis (NEW SECTION)
        print(f"\n6. STOCK MOVEMENT VARIANCE ANALYSIS:")
        if hasattr(self, 'stock_variance_r2_history') and self.stock_variance_r2_history:
            dates = sorted(self.stock_variance_r2_history.keys())
            r2_data = {i: [] for i in range(self.num_pcs)}

            for date in dates:
                r2_values = self.stock_variance_r2_history[date]
                for pc_idx in range(self.num_pcs):
                    r2_value = r2_values[pc_idx] if pc_idx < len(r2_values) else 0.0
                    r2_data[pc_idx].append(r2_value)

            for pc_idx in range(self.num_pcs):
                avg_r2 = np.nanmean(r2_data[pc_idx])
                std_r2 = np.nanstd(r2_data[pc_idx])
                print(f" PC{pc_idx + 1}: {avg_r2:.2f}% ± {std_r2:.2f}% variance explained")
        else:
            print(" No stock variance data available")

if __name__ == "__main__":
    strategy = PCAFactorStrategy(
        start_date='2019-01-01',
        end_date='2025-08-22',
        rebalance_frequency='W-FRI',
        lookback=252,
        min_trading_days=100,
        num_pcs=5,
        regression_lookback=52,
        min_regression_weeks=42
    )
    strategy.main()