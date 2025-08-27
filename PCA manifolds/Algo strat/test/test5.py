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
import numpy as np
from scipy.optimize import linprog
import yfinance as yf
import pandas as pd
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
        # self.stocks = [
        #     'ROKU',  # Roku
        #     'PTON',  # Peloton
        #     'ZM',  # Zoom
        #     'TDOC',  # Teladoc
        #     'AFRM',  # Affirm
        #     'RBLX',  # Roblox
        #     'COIN',  # Coinbase
        #     'CVNA',  # Carvana
        #     'OPEN',  # Opendoor
        #     'SHOP',  # Shopify
        #     'UPST',  # Upstart
        #     'DOCU',  # DocuSign
        #     'DOCN',  # DigitalOcean
        #     'HOOD',  # Robinhood
        #     'SNAP'  # Snap
        # ]
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.lookback = lookback
        self.min_trading_days = min_trading_days
        self.num_pcs = num_pcs
        self.centrality_std = centrality_std
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
        self.stock_data = None
        self.factor_data = None
        self.rotation_data = None
        self.rebalance_dates = None
        self.rebalance_data = {}
        self.regression_results = {}
        self.predictions = {i: {} for i in range(num_pcs)}
        self.actuals = {i: {} for i in range(num_pcs)}
        self.actuals_future = {i: {} for i in range(num_pcs)}
        # New attributes for portfolio optimization
        self.initial_capital = 10000.0  # Initial capital in dollars
        self.portfolio_values = {}  # Store portfolio value (C_total) at each rebalance
        self.weights = {}  # Store portfolio weights (v_i) at each rebalance
        self.pnl_history = {}  # Store P&L at each rebalance
        self.transaction_cost_rate = 0.001  # 0.1% transaction cost per dollar traded
        self.sector_proxy = 'XLF'  # Sector proxy for beta calculation
        self.betas = None  # Will store stock betas
        self.sectors = {
            'JPM': 'Banking', 'BAC': 'Banking', 'WFC': 'Banking', 'C': 'Banking',
            'GS': 'Investment Banking', 'MS': 'Investment Banking',
            'V': 'Payments', 'MA': 'Payments', 'AXP': 'Payments',
            'PNC': 'Regional Banking', 'TFC': 'Regional Banking', 'USB': 'Regional Banking',
            'ALL': 'Insurance', 'MET': 'Insurance', 'PRU': 'Insurance'
        }
        # self.sectors = {
        #     'ROKU': 'Communication Services',  # Streaming media
        #     'PTON': 'Consumer Discretionary',  # Fitness equipment
        #     'ZM': 'Technology',  # Video conferencing
        #     'TDOC': 'Healthcare',  # Telehealth
        #     'AFRM': 'Financials',  # Buy now, pay later
        #     'RBLX': 'Technology',  # Gaming platform
        #     'COIN': 'Financials',  # Cryptocurrency exchange
        #     'CVNA': 'Consumer Discretionary',  # Online car sales
        #     'OPEN': 'Real Estate',  # Real estate tech
        #     'SHOP': 'Technology',  # E-commerce platform
        #     'UPST': 'Financials',  # AI lending platform
        #     'DOCU': 'Technology',  # E-signature software
        #     'DOCN': 'Technology',  # Cloud infrastructure
        #     'HOOD': 'Financials',  # Trading platform
        #     'SNAP': 'Communication Services'  # Social media
        # }

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
        self.factor_data = self.compute_returns(self.factor_data)
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
        """Compute PCA loadings matrix for a rebalance date, retaining stock return scales."""
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
        returns_std = returns_std.where(returns_std > 0, 1e-10) # Avoid division by zero
        returns_standardized = (returns - returns_mean) / returns_std
        returns_standardized = returns_standardized.dropna(axis=1, how='any')
        cov_matrix = returns_standardized.T @ returns_standardized / (len(returns_standardized) - 1)
        pca = PCA(n_components=min(self.num_pcs, len(self.stocks)))
        pca.fit(returns_standardized)
        loadings = pca.components_.T
        explained_variance = pca.explained_variance_ratio_
        # Rescale loadings back to original return space
        num_stocks = min(len(self.stocks), loadings.shape[0])
        stock_std = returns_std.values[:num_stocks]
        loadings = loadings / stock_std[:, np.newaxis]
        for i in range(loadings.shape[1]):
            if np.sum(loadings[:, i]) < 0:
                loadings[:, i] = -loadings[:, i]
        return loadings, explained_variance, stock_std
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
        returns = self.compute_returns(prices)
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
    def plot_r2_scores(self):
        """Create two plots: Rolling Test R^2 using same PCA and using next PCA."""
        test_r2_data = {i: [] for i in range(self.num_pcs)}
        test_r2_future_data = {i: [] for i in range(self.num_pcs)}
        dates = []
        for rebalance_date in self.rebalance_dates:
            dates.append(rebalance_date)
            for pc_idx in range(self.num_pcs):
                test_r2 = self.compute_rolling_r2(pc_idx, rebalance_date, window=10, use_future=False)
                test_r2_data[pc_idx].append(test_r2 if test_r2 is not None else np.nan)
                test_r2_future = self.compute_rolling_r2(pc_idx, rebalance_date, window=10, use_future=True)
                test_r2_future_data[pc_idx].append(test_r2_future if test_r2_future is not None else np.nan)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # First plot: Rolling Test R² (Same PCA)
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(test_r2_data[pc_idx])
            if np.any(valid_mask):
                ax1.plot(np.array(dates)[valid_mask], np.array(test_r2_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)
        ax1.set_title("Rolling Test R² Over Time (Same PCA)", fontsize=14)
        ax1.set_ylabel("R² Score", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(-2, 1)
        # Second plot: Rolling Test R² (Next PCA)
        for pc_idx in range(self.num_pcs):
            valid_mask = ~np.isnan(test_r2_future_data[pc_idx])
            if np.any(valid_mask):
                ax2.plot(np.array(dates)[valid_mask], np.array(test_r2_future_data[pc_idx])[valid_mask],
                         label=f'PC{pc_idx + 1}', marker='o', markersize=3,
                         color=colors[pc_idx % len(colors)], alpha=0.7)
        ax2.set_title("Rolling Test R² Over Time (Next PCA)", fontsize=14)
        ax2.set_xlabel("Rebalance Date", fontsize=12)
        ax2.set_ylabel("R² Score", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(-2, 1)
        plt.tight_layout()
        plt.show()
        # Output average R² over time for both plots
        print("Average Rolling R² over time (Same PCA):")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(test_r2_data[pc_idx])
            print(f"PC{pc_idx+1}: {avg_r2:.4f}")
        print("Average Rolling R² over time (Next PCA):")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(test_r2_future_data[pc_idx])
            print(f"PC{pc_idx+1}: {avg_r2:.4f}")
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
                'centrality_vector': data['centrality_vector'].tolist() if data[
                                                                               'centrality_vector'] is not None else None,
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
    def debug_scaling(self, rebalance_date):
        """Debug function to print ranges of key variables for scaling analysis."""
        print(f"\nDebugging scaling for {rebalance_date.date()}:")
        # Factor returns scale
        factor_returns = self.factor_data.loc[self.factor_data.index <= rebalance_date]
        factor_returns = factor_returns.iloc[-self.regression_lookback:]
        if not factor_returns.empty:
            print(
                f" Factor returns range (min, max): {factor_returns.min().min():.4f}, {factor_returns.max().max():.4f}")
        # PC returns and sigma
        pc_returns, next_pc_returns = self.compute_weekly_pc_returns(rebalance_date)
        if pc_returns is not None:
            print(f" PC returns range (min, max): {pc_returns.min().min():.4f}, {pc_returns.max().max():.4f}")
        sigma = self.compute_pc_std(rebalance_date)
        if sigma is not None:
            print(f" Sigma: {sigma}")
        # Predicted PC movements
        pred_pct_change_pc = self.compute_predicted_pc_movement(rebalance_date)
        if pred_pct_change_pc is not None:
            print(f" Predicted PC movements range: {pred_pct_change_pc.min():.4f}, {pred_pct_change_pc.max():.4f}")
        # Predicted stock returns
        r_hat = self.compute_predicted_stock_returns(rebalance_date)
        if r_hat is not None:
            print(f" Predicted stock returns (r_hat) range: {r_hat.min():.4f}, {r_hat.max():.4f}")
        # Weighted predicted returns
        r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
        if r_hat_weighted is not None:
            print(f" Weighted predicted returns range: {r_hat_weighted.min():.4f}, {r_hat_weighted.max():.4f}")
        # Actual stock returns
        actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
        if actual_returns is not None:
            print(f" Actual stock returns range: {actual_returns.min():.4f}, {actual_returns.max():.4f}")
        # Centrality vector
        centrality_vector = self.rebalance_data.get(rebalance_date, {}).get('centrality_vector')
        if centrality_vector is not None:
            print(f" Centrality vector range: {centrality_vector.min():.4f}, {centrality_vector.max():.4f}")
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
        # Ensure dimensions match
        num_stocks = min(len(self.stocks), pca_matrix.shape[0])
        num_pcs = min(self.num_pcs, pca_matrix.shape[1])
        pca_matrix = pca_matrix[:num_stocks, :num_pcs]
        pred_pct_change_pc = pred_pct_change_pc[:num_pcs]
        # Compute predicted stock returns
        r_hat = pca_matrix @ pred_pct_change_pc
        stock_std = self.rebalance_data.get(rebalance_date, {}).get('stock_std')
        r_hat = stock_std[:num_stocks] * r_hat * stock_std[:num_stocks]
        return r_hat
    def compute_weighted_predicted_returns(self, rebalance_date):
        """Apply centrality weighting to predicted stock returns without compression."""
        r_hat = self.compute_predicted_stock_returns(rebalance_date)
        if r_hat is None:
            return None
        centrality_vector = self.rebalance_data.get(rebalance_date, {}).get('centrality_vector')
        if centrality_vector is None:
            return None
        # Ensure dimensions match
        num_stocks = min(len(r_hat), len(centrality_vector))
        r_hat = r_hat[:num_stocks]
        centrality_vector = centrality_vector[:num_stocks]
        # Normalize centrality vector to mean=1.0
        centrality_vector = centrality_vector / centrality_vector.mean()
        r_hat_weighted = r_hat * centrality_vector
        return r_hat_weighted
    def plot_pred_vs_actual_scatter(self):
        """Scatter plot: predicted vs actual weekly returns for each stock and rebalance date."""
        predicted_data = {}
        actual_data = {}
        for rebalance_date in self.rebalance_dates[:-1]: # Skip last if no actuals
            r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
            if r_hat_weighted is not None and actual_returns is not None:
                predicted_data[rebalance_date] = r_hat_weighted * 100  # To percent
                actual_data[rebalance_date] = actual_returns * 100  # To percent
        if not predicted_data:
            print("No data available for scatter plot")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.stocks)))
        for stock_idx, stock in enumerate(self.stocks):
            preds = [predicted_data[date][stock_idx] for date in predicted_data if
                     len(predicted_data[date]) > stock_idx]
            acts = [actual_data[date][stock_idx] for date in actual_data if len(actual_data[date]) > stock_idx]
            ax.scatter(preds, acts, color=colors[stock_idx], label=stock, alpha=0.6)
        ax.set_xlabel('Weekly Predicted Returns (%)')
        ax.set_ylabel('Weekly Actual Returns (%)')
        ax.set_title('Scatter: Predicted vs Actual Stock Returns Across Rebalance Dates (Percent)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Compute and output average R² for the prediction vs actual
        all_preds = []
        all_acts = []
        for date in predicted_data:
            all_preds.extend(predicted_data[date])
            all_acts.extend(actual_data[date])
        if all_preds:
            overall_r2 = r2_score(all_acts, all_preds)
            print(f"Overall R² between predicted and actual stock returns: {overall_r2:.4f}")
    def plot_predicted_over_time(self):
        """Line plot: predicted stock movement over time for each stock."""
        predicted_data = {}
        for rebalance_date in self.rebalance_dates:
            r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
            if r_hat_weighted is not None:
                predicted_data[rebalance_date] = r_hat_weighted * 100  # To percent
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

    def compute_betas(self, rebalance_date):
        """Compute stock betas relative to sector proxy (XLF) over the lookback period."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None

        end_idx = self.stock_data.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback + 1)
        if end_idx - start_idx + 1 < self.min_trading_days:
            print(
                f"Warning: Insufficient data for beta calculation at {rebalance_date.date()} ({end_idx - start_idx + 1} days)")
            return None

        # Fetch stock prices for the lookback period
        prices = self.stock_data.iloc[start_idx:end_idx + 1]
        if prices.empty:
            print(f"Warning: Empty stock price data for {rebalance_date.date()}")
            return None

        # Fetch sector proxy (XLF) data for the same period
        try:
            sector_data = yf.download(
                self.sector_proxy,
                start=prices.index[0],
                end=prices.index[-1] + pd.Timedelta(days=1),  # Ensure end date inclusivity
                auto_adjust=True
            )['Close']
            # Ensure sector_data is a Series, not a DataFrame
            if isinstance(sector_data, pd.DataFrame):
                if sector_data.shape[1] == 1:
                    sector_data = sector_data.iloc[:, 0]
                else:
                    print(
                        f"Warning: Sector data for {self.sector_proxy} at {rebalance_date.date()} has unexpected shape {sector_data.shape}")
                    return None
        except Exception as e:
            print(f"Warning: Failed to download sector proxy {self.sector_proxy} data at {rebalance_date.date()}: {e}")
            return None

        if sector_data.empty or len(sector_data) < self.min_trading_days:
            print(
                f"Warning: Insufficient or empty sector data for {self.sector_proxy} at {rebalance_date.date()} (rows: {len(sector_data)})")
            return None

        # Compute returns for stocks and sector
        returns = self.compute_returns(prices)
        sector_returns = self.compute_returns(sector_data)

        # Align dates
        common_dates = returns.index.intersection(sector_returns.index)
        if len(common_dates) < self.min_trading_days:
            print(
                f"Warning: Insufficient common dates for beta calculation at {rebalance_date.date()} ({len(common_dates)} days)")
            return None

        returns = returns.loc[common_dates]
        sector_returns = sector_returns.loc[common_dates]

        # Ensure sector_returns is a Series
        if isinstance(sector_returns, pd.DataFrame):
            if sector_returns.shape[1] == 1:
                sector_returns = sector_returns.iloc[:, 0]
            else:
                print(f"Warning: sector_returns has unexpected shape {sector_returns.shape} at {rebalance_date.date()}")
                return None

        # Debugging: Print shapes to diagnose dimension issues
        print(
            f"Debug: returns shape = {returns.shape}, sector_returns shape = {sector_returns.shape} at {rebalance_date.date()}")

        betas = []
        for stock in self.stocks:
            if stock in returns.columns:
                try:
                    stock_returns = returns[stock]
                    if len(stock_returns) != len(sector_returns):
                        print(
                            f"Warning: Mismatch in return lengths for {stock} ({len(stock_returns)}) vs sector ({len(sector_returns)}) at {rebalance_date.date()}")
                        betas.append(0.0)
                        continue
                    cov = stock_returns.cov(sector_returns)
                    var = sector_returns.var()
                    beta = cov / var if var > 0 else 0.0
                    betas.append(beta)
                except Exception as e:
                    print(f"Warning: Error computing beta for {stock} at {rebalance_date.date()}: {e}")
                    betas.append(0.0)
            else:
                print(f"Warning: Stock {stock} not in returns data at {rebalance_date.date()}")
                betas.append(0.0)

        return np.array(betas)

    def optimize_portfolio(self, rebalance_date):
        """Optimize portfolio weights using linear programming with specified constraints, ensuring market neutrality."""
        r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
        if r_hat_weighted is None:
            print(f"Warning: No predicted returns for {rebalance_date.date()}")
            return None
        betas = self.compute_betas(rebalance_date)
        if betas is None:
            print(f"Warning: No betas for {rebalance_date.date()}")
            return None
        prev_date = self.rebalance_dates[self.rebalance_dates < rebalance_date][-1] if rebalance_date != \
                                                                                       self.rebalance_dates[0] else None
        C_total = self.portfolio_values.get(prev_date, self.initial_capital)
        n = len(self.stocks)
        # Decision variables: v_i (long and short positions handled by positive/negative values)
        c = -r_hat_weighted  # Objective: maximize sum(v_i * r_hat_weighted_i)
        # Constraints
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        bounds = []
        # 1. Market neutrality: sum(v_i) = 0
        A_eq.append(np.ones(n))
        b_eq.append(0.0)
        # 2. Long and short allocations
        long_mask = r_hat_weighted >= 0
        short_mask = r_hat_weighted < 0
        # Sum of long positions = 0.65 * C_total
        long_eq = np.zeros(n)
        long_eq[long_mask] = 1.0
        A_eq.append(long_eq)
        b_eq.append(0.65 * C_total)
        # Sum of abs(short positions) = 0.35 * C_total
        short_eq = np.zeros(n)
        short_eq[short_mask] = -1.0
        A_eq.append(short_eq)
        b_eq.append(0.35 * C_total)
        # 3. Individual stock limits and short only negative returns
        for i in range(n):
            if r_hat_weighted[i] >= 0:
                bounds.append((0, 0.15 * C_total))
            else:
                bounds.append((-0.12 * C_total, 0))
        # 4. Short concentration limit: |v_i_short| <= 0.3 * total_short_allocation
        total_short_allocation = 0.35 * C_total
        for i in range(n):
            if r_hat_weighted[i] < 0:
                row = np.zeros(n)
                row[i] = -1.0
                A_ub.append(row)
                b_ub.append(0.3 * total_short_allocation)
        # 5. Beta exposure constraint: |v · β| <= 0.15
        A_ub.append(betas)
        A_ub.append(-betas)
        b_ub.append(0.15 * C_total)
        b_ub.append(0.15 * C_total)
        # 6. Total allocation: sum(|v_i|) <= C_total
        A_ub.append(np.ones(n))
        A_ub.append(-np.ones(n))
        b_ub.append(C_total)
        b_ub.append(C_total)
        # 7. Sector exposure: limit exposure to any sector to 40% of C_total
        sector_limits = 0.4 * C_total
        for sector in set(self.sectors.values()):
            sector_mask = np.array([1 if self.sectors[stock] == sector else 0 for stock in self.stocks])
            A_ub.append(sector_mask)
            A_ub.append(-sector_mask)
            b_ub.append(sector_limits)
            b_ub.append(sector_limits)
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        # Track optimization method
        if not hasattr(self, 'optimization_counts'):
            self.optimization_counts = {'linprog_success': 0, 'least_squares_fallback': 0}
        # Try linear programming
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                weights = res.x
                # Verify market neutrality
                if abs(np.sum(weights)) > 1e-6:
                    print(
                        f"Warning: Linear programming solution not market neutral at {rebalance_date.date()}. Adjusting weights.")
                    weights -= np.sum(weights) / n
                self.optimization_counts['linprog_success'] += 1
                print(f"Optimization Debug: Used linprog successfully for {rebalance_date.date()}")
                return weights
        except Exception as e:
            print(f"Linear programming failed for {rebalance_date.date()}: {e}")
        # Fallback to least squares optimization
        print(f"Falling back to least squares optimization for {rebalance_date.date()}")
        self.optimization_counts['least_squares_fallback'] += 1
        weights = np.zeros(n)
        long_indices = np.where(r_hat_weighted >= 0)[0]
        short_indices = np.where(r_hat_weighted < 0)[0]
        if len(long_indices) > 0:
            long_weights = np.ones(len(long_indices)) * (0.65 * C_total / max(len(long_indices), 1))
            weights[long_indices] = long_weights
        if len(short_indices) > 0:
            short_weights = -np.ones(len(short_indices)) * (0.35 * C_total / max(len(short_indices), 1))
            weights[short_indices] = short_weights
        # Ensure market neutrality
        net_position = np.sum(weights)
        if abs(net_position) > 1e-6:
            adjustment = net_position / n
            weights -= adjustment
        # Verify short only negative returns
        for i in range(n):
            if r_hat_weighted[i] >= 0 and weights[i] < 0:
                weights[i] = 0
            elif r_hat_weighted[i] < 0 and weights[i] > 0:
                weights[i] = 0
        # Re-normalize to maintain long/short allocation constraints
        long_sum = np.sum(weights[weights > 0])
        short_sum = np.abs(np.sum(weights[weights < 0]))
        if long_sum > 0:
            weights[weights > 0] *= (0.65 * C_total) / long_sum
        if short_sum > 0:
            weights[weights < 0] *= (0.35 * C_total) / short_sum
        # Final market neutrality adjustment
        net_position = np.sum(weights)
        if abs(net_position) > 1e-6:
            weights -= net_position / n
        return weights

    def compute_portfolio_metrics(self, rebalance_date, weights, prev_weights=None):
        """Compute P&L, transaction costs, and portfolio metrics without P&L cap."""
        actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
        if actual_returns is None or weights is None:
            print(f"Warning: No actual returns or weights for {rebalance_date.date()}")
            return None, None, None, None
        prev_date = self.rebalance_dates[self.rebalance_dates < rebalance_date][-1] if rebalance_date != \
                                                                                       self.rebalance_dates[0] else None
        C_total = self.portfolio_values.get(prev_date, self.initial_capital)
        # Ensure dimensions match
        actual_returns = actual_returns[:len(weights)]
        # Calculate P&L: weights (in dollars) * actual_returns (simple returns)
        pnl = np.sum(weights * actual_returns)
        # Calculate transaction costs
        turnover = 0.0
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
        transaction_costs = turnover * self.transaction_cost_rate
        # Net P&L
        net_pnl = pnl - transaction_costs
        # Long and short exposures
        long_exposure = np.sum(weights[weights > 0])
        short_exposure = np.abs(np.sum(weights[weights < 0]))
        # Debugging: Print P&L components
        print(
            f"Debug: {rebalance_date.date()} - P&L: {pnl:.2f}, Transaction Costs: {transaction_costs:.2f}, "
            f"Net P&L: {net_pnl:.2f}, Portfolio Value: {C_total + net_pnl:.2f}")
        return net_pnl, transaction_costs, long_exposure, short_exposure

    def compute_sharpe_ratio(self):
        """Compute annualized Sharpe ratio based on weekly P&L."""
        pnl_values = [self.pnl_history[date][0] for date in sorted(self.pnl_history.keys()) if
                      self.pnl_history[date][0] is not None]
        if len(pnl_values) < 2:
            return None
        weekly_returns = np.array(pnl_values) / self.initial_capital
        mean_return = np.mean(weekly_returns)
        std_return = np.std(weekly_returns, ddof=1)
        if std_return == 0:
            return None
        annualized_return = mean_return * 52
        annualized_std = std_return * np.sqrt(52)
        sharpe_ratio = annualized_return / annualized_std
        return sharpe_ratio, annualized_return, annualized_std

    def compute_sortino_ratio(self):
        """Compute annualized Sortino ratio using downside deviation."""
        pnl_values = [self.pnl_history[date][0] for date in sorted(self.pnl_history.keys()) if
                      self.pnl_history[date][0] is not None]
        if len(pnl_values) < 2:
            return None
        weekly_returns = np.array(pnl_values) / self.initial_capital
        mean_return = np.mean(weekly_returns)
        downside_returns = weekly_returns[weekly_returns < 0]
        if len(downside_returns) == 0:
            return None
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return None
        annualized_return = mean_return * 52
        annualized_downside_std = downside_std * np.sqrt(52)
        sortino_ratio = annualized_return / annualized_downside_std
        return sortino_ratio

    def compute_max_drawdown(self):
        """Compute maximum drawdown based on portfolio values with debugging output."""
        dates = sorted(self.portfolio_values.keys())
        if not dates:
            print("No portfolio value data available for max drawdown calculation")
            return None
        portfolio_values = [self.portfolio_values[date] for date in dates]
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        # Debugging output
        if len(drawdowns) > 0:
            max_drawdown_idx = np.argmax(drawdowns)
            peak_date = dates[max_drawdown_idx]
            peak_value = running_max[max_drawdown_idx]
            trough_value = portfolio_values[max_drawdown_idx]
            print(f"Max Drawdown Debug: Peak at {peak_date.date()} (${peak_value:.2f}), "
                  f"Trough at {peak_date.date()} (${trough_value:.2f}), Drawdown = {max_drawdown * 100:.2f}%")
        else:
            print("Max Drawdown Debug: No drawdowns calculated (insufficient data)")
        return max_drawdown

    def compute_profit_factor(self):
        """Compute profit factor (gross profits / gross losses)."""
        pnl_values = [self.pnl_history[date][0] for date in sorted(self.pnl_history.keys()) if
                      self.pnl_history[date][0] is not None]
        if len(pnl_values) == 0:
            return None
        gross_profits = sum(p for p in pnl_values if p > 0)
        gross_losses = abs(sum(p for p in pnl_values if p < 0))
        if gross_losses == 0:
            return None if gross_profits == 0 else float('inf')
        return gross_profits / gross_losses

    def compute_long_short_profits(self, rebalance_date):
        """Compute P&L contributions from long and short positions for a rebalance date."""
        actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
        weights = self.weights.get(rebalance_date)
        if actual_returns is None or weights is None:
            return None, None
        actual_returns = actual_returns[:len(weights)]
        long_mask = weights > 0
        short_mask = weights < 0
        long_pnl = np.sum(weights[long_mask] * actual_returns[long_mask]) if np.any(long_mask) else 0.0
        short_pnl = np.sum(weights[short_mask] * actual_returns[short_mask]) if np.any(short_mask) else 0.0
        return long_pnl, short_pnl


    def plot_profits_over_time(self):
        """Plot portfolio value (initial capital + cumulative P&L) over time."""
        dates = sorted(self.portfolio_values.keys())
        if not dates:
            print("No portfolio value data available for plotting")
            return
        portfolio_values = []
        for date in dates:
            value = self.portfolio_values.get(date, self.initial_capital)
            portfolio_values.append(value)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, portfolio_values, marker='o', color='blue', label='Portfolio Value')
        ax.set_xlabel('Time (Rebalance Dates)')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Over Time (Starting at $10,000)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_portfolio_composition(self):
        """Plot portfolio composition (long/short dollar allocations) over time as a stacked area chart."""
        dates = sorted(self.weights.keys())
        if not dates:
            print("No portfolio weights available for plotting")
            return
        long_data = []
        short_data = []
        for date in dates:
            weights = self.weights[date]
            C_total = self.portfolio_values.get(date, self.initial_capital)
            # Use dollar amounts directly for market neutrality visualization
            long_dollars = np.where(weights > 0, weights, 0)
            short_dollars = np.where(weights < 0, -weights, 0)
            long_data.append(long_dollars)
            short_data.append(short_dollars)
        long_data = np.array(long_data).T  # Shape: (n_stocks, n_dates)
        short_data = np.array(short_data).T
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.stocks)))
        # Plot long positions (above y=0)
        ax.stackplot(dates, long_data, labels=[f'{stock} (Long)' for stock in self.stocks], colors=colors, alpha=0.6)
        # Plot short positions (below y=0)
        ax.stackplot(dates, -short_data, labels=[f'{stock} (Short)' for stock in self.stocks], colors=colors, alpha=0.6)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (Rebalance Dates)')
        ax.set_ylabel('Portfolio Composition ($)')
        ax.set_title('Portfolio Composition Over Time (Longs Above, Shorts Below, Market Neutral)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_beta_exposure(self):
        """Plot portfolio beta exposure over time."""
        dates = sorted(self.weights.keys())
        if not dates:
            print("No portfolio weights available for beta exposure plotting")
            return
        beta_exposures = []
        for date in dates:
            weights = self.weights[date]
            betas = self.compute_betas(date)
            if betas is None:
                beta_exposures.append(np.nan)
                continue
            beta_exposure = np.abs(np.dot(weights, betas))
            beta_exposures.append(beta_exposure)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, beta_exposures, marker='o', color='purple', label='Portfolio Beta Exposure')
        ax.axhline(0.15, color='red', linestyle='--', label='Beta Constraint (±0.15)')
        ax.axhline(-0.15, color='red', linestyle='--')
        ax.set_xlabel('Time (Rebalance Dates)')
        ax.set_ylabel('Portfolio Beta Exposure')
        ax.set_title('Portfolio Beta Exposure Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    def debug_and_validate_last_two_days(self):
        """Debug and validate data for the last two rebalance dates, including regression, returns, and portfolio metrics."""
        if len(self.rebalance_dates) < 2:
            print("Insufficient rebalance dates for debugging (need at least 2)")
            return
        last_two_dates = self.rebalance_dates[-2:]
        for date in last_two_dates:
            print(f"\n=== Debugging and Validation for {date.date()} ===")
            # 1. Validate Data Availability
            data = self.rebalance_data.get(date, {})
            pca_matrix = data.get('pca_matrix')
            centrality_vector = data.get('centrality_vector')
            actual_returns = data.get('actual_returns')
            explained_variance = data.get('explained_variance')
            stock_std = data.get('stock_std')
            regression_results = self.regression_results.get(date, {})
            weights = self.weights.get(date)
            net_pnl, transaction_costs, long_exposure, short_exposure = self.pnl_history.get(date,
                                                                                             (None, None, None, None))
            print("\nData Availability Check:")
            print(f" PCA Matrix: {'Available' if pca_matrix is not None else 'Missing'}")
            print(f" Centrality Vector: {'Available' if centrality_vector is not None else 'Missing'}")
            print(f" Actual Returns: {'Available' if actual_returns is not None else 'Missing'}")
            print(f" Explained Variance: {'Available' if explained_variance is not None else 'Missing'}")
            print(f" Stock Std: {'Available' if stock_std is not None else 'Missing'}")
            print(f" Regression Results: {'Available' if regression_results else 'Missing'}")
            print(f" Weights: {'Available' if weights is not None else 'Missing'}")
            print(f" Portfolio Metrics: {'Available' if net_pnl is not None else 'Missing'}")
            # 2. Validate Dimensions
            if pca_matrix is not None:
                print(f"\nPCA Matrix Shape: {pca_matrix.shape} (Expected: {len(self.stocks)} x {self.num_pcs})")
                if pca_matrix.shape[0] != len(self.stocks) or pca_matrix.shape[1] != self.num_pcs:
                    print(" Warning: PCA matrix dimensions mismatch")
            if centrality_vector is not None:
                print(f"Centrality Vector Length: {len(centrality_vector)} (Expected: {len(self.stocks)})")
                if len(centrality_vector) != len(self.stocks):
                    print(" Warning: Centrality vector length mismatch")
            if actual_returns is not None:
                print(f"Actual Returns Length: {len(actual_returns)} (Expected: {len(self.stocks)})")
                if len(actual_returns) != len(self.stocks):
                    print(" Warning: Actual returns length mismatch")
            if weights is not None:
                print(f"Weights Length: {len(weights)} (Expected: {len(self.stocks)})")
                if len(weights) != len(self.stocks):
                    print(" Warning: Weights length mismatch")
            # 3. Check for NaNs and Unrealistic Values
            if actual_returns is not None:
                if np.any(np.isnan(actual_returns)):
                    print(" Warning: NaNs detected in actual returns")
                if np.any(np.abs(actual_returns) > 0.5):
                    print(" Warning: Unrealistic actual returns detected")
            if weights is not None:
                if np.any(np.isnan(weights)):
                    print(" Warning: NaNs detected in weights")
                if np.abs(np.sum(weights)) > 1e-6:
                    print(f" Warning: Portfolio not market neutral (Net position: {np.sum(weights):.6f})")
            # 4. Regression Details
            print("\nRegression Details:")
            for pc_idx in range(self.num_pcs):
                result = regression_results.get(pc_idx, {})
                if not result:
                    print(f" PC{pc_idx + 1}: No regression data")
                    continue
                print(f" PC{pc_idx + 1}:")
                print(f"  Train R²: {result.get('train_r2', 'N/A'):.4f}")
                print(f"  CV R² Mean: {result.get('cv_r2_mean', 'N/A'):.4f}")
                print(f"  CV R² Std: {result.get('cv_r2_std', 'N/A'):.4f}")
                print(f"  Test Error: {result.get('test_error', 'N/A')}")
                print(f"  Predicted PC Return: {result.get('prediction', 'N/A'):.6f}")
                print(f"  Actual PC Return: {result.get('actual', 'N/A'):.6f}")
                print("  Selected Factors and Coefficients:")
                coef_dict = result.get('coefficients', {})
                for factor, coef in coef_dict.items():
                    if abs(coef) > 0:
                        print(f"   {factor}: {coef:.6f}")
            # 5. Predicted and Actual Returns
            r_hat = self.compute_predicted_stock_returns(date)
            r_hat_weighted = self.compute_weighted_predicted_returns(date)
            print("\nStock Returns:")
            print(" Stock | Predicted | Weighted Predicted | Actual")
            print("-" * 50)
            for i, stock in enumerate(self.stocks):
                pred = r_hat[i] if r_hat is not None and i < len(r_hat) else np.nan
                pred_weighted = r_hat_weighted[i] if r_hat_weighted is not None and i < len(r_hat_weighted) else np.nan
                actual = actual_returns[i] if actual_returns is not None and i < len(actual_returns) else np.nan
                print(f" {stock:<4} | {pred:.6f} | {pred_weighted:.6f} | {actual:.6f}")
            # 6. Portfolio Metrics
            print("\nPortfolio Metrics:")
            print(f" Portfolio Value: ${self.portfolio_values.get(date, self.initial_capital):.2f}")
            print(f" Net P&L: ${net_pnl:.2f}" if net_pnl is not None else " Net P&L: N/A")
            print(
                f" Transaction Costs: ${transaction_costs:.2f}" if transaction_costs is not None else " Transaction Costs: N/A")
            print(f" Long Exposure: ${long_exposure:.2f}" if long_exposure is not None else " Long Exposure: N/A")
            print(f" Short Exposure: ${short_exposure:.2f}" if short_exposure is not None else " Short Exposure: N/A")
            if weights is not None:
                print(" Portfolio Weights:")
                for stock, w in zip(self.stocks, weights):
                    print(
                        f"  {stock}: ${w:.2f} ({w / self.portfolio_values.get(date, self.initial_capital) * 100:.2f}%)")
            # 7. Additional Debug Info
            print("\nAdditional Debug Info:")
            factor_returns = self.factor_data.loc[self.factor_data.index <= date].iloc[-1]
            print(" Latest Factor Returns:")
            for factor, ret in factor_returns.items():
                print(f"  {factor}: {ret:.6f}")
            if pca_matrix is not None:
                print(" PCA Loadings (First 5 stocks, all PCs):")
                for i, stock in enumerate(self.stocks[:5]):
                    loadings = pca_matrix[i, :] if i < pca_matrix.shape[0] else [np.nan] * self.num_pcs
                    print(f"  {stock}: {loadings}")
            if explained_variance is not None:
                print(" Explained Variance Ratios:")
                for i, var in enumerate(explained_variance, 1):
                    print(f"  PC{i}: {var:.4f} ({var * 100:.2f}%)")

    def plot_returns_boxplots(self):
        """Create box plots for actual and predicted returns on the same page."""
        # Collect actual and predicted returns
        actual_data = {}
        predicted_data = {}
        for rebalance_date in self.rebalance_dates[:-1]:
            r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
            actual_returns = self.rebalance_data.get(rebalance_date, {}).get('actual_returns')
            if r_hat_weighted is not None:
                predicted_data[rebalance_date] = r_hat_weighted * 100  # To percent
            if actual_returns is not None:
                actual_data[rebalance_date] = actual_returns * 100  # To percent
        if not actual_data or not predicted_data:
            print("No data available for box plots")
            return
        # Prepare data for box plots
        actual_returns_by_stock = {stock: [] for stock in self.stocks}
        predicted_returns_by_stock = {stock: [] for stock in self.stocks}
        for date in actual_data:
            for i, stock in enumerate(self.stocks):
                if i < len(actual_data[date]):
                    actual_returns_by_stock[stock].append(actual_data[date][i])
        for date in predicted_data:
            for i, stock in enumerate(self.stocks):
                if i < len(predicted_data[date]):
                    predicted_returns_by_stock[stock].append(predicted_data[date][i])
        # Convert to lists for plotting
        actual_data_list = [actual_returns_by_stock[stock] for stock in self.stocks]
        predicted_data_list = [predicted_returns_by_stock[stock] for stock in self.stocks]
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # Box plot for actual returns
        ax1.boxplot(actual_data_list, vert=True, patch_artist=True, labels=self.stocks)
        ax1.set_title('Actual Weekly Returns by Stock', fontsize=14)
        ax1.set_ylabel('Returns (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        # Box plot for predicted returns
        ax2.boxplot(predicted_data_list, vert=True, patch_artist=True, labels=self.stocks)
        ax2.set_title('Predicted Weekly Returns by Stock', fontsize=14)
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.set_xlabel('Stocks', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        # Print summary statistics
        print("\nActual Returns Summary Statistics:")
        for i, stock in enumerate(self.stocks):
            data = actual_data_list[i]
            if data:
                print(f"{stock}:")
                print(f" Mean: {np.mean(data):.6f}")
                print(f" Median: {np.median(data):.6f}")
                print(f" Q1: {np.percentile(data, 25):.6f}")
                print(f" Q3: {np.percentile(data, 75):.6f}")
        print("\nPredicted Returns Summary Statistics:")
        for i, stock in enumerate(self.stocks):
            data = predicted_data_list[i]
            if data:
                print(f"{stock}:")
                print(f" Mean: {np.mean(data):.6f}")
                print(f" Median: {np.median(data):.6f}")
                print(f" Q1: {np.percentile(data, 25):.6f}")
                print(f" Q3: {np.percentile(data, 75):.6f}")

    def plot_stock_returns_and_composition(self, num_plots=6):
        """Plot for each stock: actual returns overlaid with portfolio composition percentage over time.
        Both series scaled to [-1, 1]. Limited to first num_plots stocks."""
        dates = sorted(self.rebalance_dates)
        if not dates or len(dates) < 2:
            print("Insufficient data for stock returns and composition plots")
            return
        # Collect data: for each stock, lists of returns and compositions over dates
        returns_data = {stock: [] for stock in self.stocks}
        composition_data = {stock: [] for stock in self.stocks}
        valid_dates = []  # Only dates with both returns and weights
        for i in range(len(dates) - 1):  # Returns are forward-looking
            date = dates[i]
            if date not in self.weights or date not in self.rebalance_data:
                continue
            weights = self.weights[date]
            actual_returns = self.rebalance_data[date].get('actual_returns')
            if actual_returns is None:
                continue
            C_total = self.portfolio_values.get(date, self.initial_capital)
            for stock_idx, stock in enumerate(self.stocks):
                if stock_idx < len(weights) and stock_idx < len(actual_returns):
                    # Composition: signed fraction (positive long, negative short)
                    composition = weights[stock_idx] / C_total if C_total != 0 else 0.0
                    returns_data[stock].append(actual_returns[stock_idx])
                    composition_data[stock].append(composition)
            valid_dates.append(date)
        if not valid_dates:
            print("No valid data for stock returns and composition plots")
            return
        # Plot for each stock up to num_plots
        num_plots = min(num_plots, len(self.stocks))
        for stock_idx in range(num_plots):
            stock = self.stocks[stock_idx]
            ret_series = np.array(returns_data[stock])
            comp_series = np.array(composition_data[stock])
            if len(ret_series) == 0 or len(comp_series) == 0:
                print(f"No data for {stock}")
                continue

            # Scale each series to [-1, 1] independently
            def scale_to_minus1_1(y):
                if np.max(y) == np.min(y):
                    return np.zeros_like(y)  # All zero if constant
                return 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

            ret_scaled = scale_to_minus1_1(ret_series)
            comp_scaled = scale_to_minus1_1(comp_series)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(valid_dates, ret_scaled, marker='o', color='blue', label='Scaled Actual Returns')
            ax.plot(valid_dates, comp_scaled, marker='x', color='red', label='Scaled Composition Fraction (signed)')
            ax.set_xlabel('Rebalance Dates')
            ax.set_ylabel('Scaled Value [-1, 1]')
            ax.set_title(f'{stock}: Scaled Returns vs Portfolio Composition Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

    def main(self):
        """Main method with regression, optimization, and detailed profitability report."""
        self.download_data()
        self.portfolio_values = {date: self.initial_capital for date in self.rebalance_dates}
        prev_weights = None
        long_pnls = []
        short_pnls = []
        for rebalance_date in self.rebalance_dates:
            pca_matrix, explained_variance, stock_std = self.compute_pca_for_rebalance(rebalance_date)
            centrality_vector = self.compute_centrality_for_rebalance(rebalance_date)
            actual_returns = self.compute_actual_returns_for_rebalance(rebalance_date)
            self.rebalance_data[rebalance_date] = {
                'pca_matrix': pca_matrix,
                'centrality_vector': centrality_vector,
                'actual_returns': actual_returns,
                'explained_variance': explained_variance,
                'stock_std': stock_std
            }
            regression_results = self.train_regression_models(rebalance_date)
            if regression_results:
                self.regression_results[rebalance_date] = regression_results
            weights = self.optimize_portfolio(rebalance_date)
            if weights is not None:
                self.weights[rebalance_date] = weights
                net_pnl, transaction_costs, long_exposure, short_exposure = self.compute_portfolio_metrics(
                    rebalance_date, weights, prev_weights
                )
                self.pnl_history[rebalance_date] = (net_pnl, transaction_costs, long_exposure, short_exposure)
                if net_pnl is not None and rebalance_date != self.rebalance_dates[-1]:
                    next_date_idx = self.rebalance_dates.get_loc(rebalance_date) + 1
                    next_date = self.rebalance_dates[next_date_idx]
                    current_value = self.portfolio_values.get(rebalance_date, self.initial_capital)
                    self.portfolio_values[next_date] = current_value + net_pnl
                    print(
                        f"Debug: Updated portfolio value for {next_date.date()}: {self.portfolio_values[next_date]:.2f}")
                long_pnl, short_pnl = self.compute_long_short_profits(rebalance_date)
                if long_pnl is not None:
                    long_pnls.append(long_pnl)
                    short_pnls.append(short_pnl)
                prev_weights = weights
            self.debug_scaling(rebalance_date)
        self.compute_actuals_future()
        self.plot_r2_scores()
        self.plot_pred_vs_actual_scatter()
        self.plot_predicted_over_time()
        self.plot_actual_over_time()
        self.plot_profits_over_time()
        self.plot_portfolio_composition()
        self.plot_beta_exposure()
        self.plot_returns_boxplots()
        self.plot_stock_returns_and_composition(num_plots=6)

        # Detailed Profitability Report
        print("\n=== Detailed Portfolio Performance Report ===")
        print(f"Date Range: {self.rebalance_dates[0].date()} to {self.rebalance_dates[-1].date()}")
        print(f"Rebalance Frequency: {self.rebalance_frequency}")
        print(f"Number of Rebalance Periods: {len(self.rebalance_dates)}")
        print(f"Initial Capital: ${self.initial_capital:.2f}")

        # P&L Metrics
        cumulative_pnl = sum(pnl[0] for pnl in self.pnl_history.values() if pnl[0] is not None)
        print(f"\nCumulative P&L: ${cumulative_pnl:.2f}")
        avg_weekly_pnl = cumulative_pnl / len(self.pnl_history) if self.pnl_history else 0.0
        print(f"Average Weekly P&L: ${avg_weekly_pnl:.2f}")

        # Long and Short P&L
        avg_long_pnl = np.mean(long_pnls) if long_pnls else 0.0
        avg_short_pnl = np.mean(short_pnls) if short_pnls else 0.0
        print(f"Average Weekly Long P&L: ${avg_long_pnl:.2f}")
        print(f"Average Weekly Short P&L: ${avg_short_pnl:.2f}")
        total_long_pnl = sum(long_pnls) if long_pnls else 0.0
        total_short_pnl = sum(short_pnls) if short_pnls else 0.0
        print(f"Total Long P&L: ${total_long_pnl:.2f}")
        print(f"Total Short P&L: ${total_short_pnl:.2f}")

        # Transaction Costs and Turnover
        total_transaction_costs = sum(pnl[1] for pnl in self.pnl_history.values() if pnl[1] is not None)
        print(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
        avg_turnover = np.mean([pnl[1] / self.transaction_cost_rate for pnl in self.pnl_history.values()
                                if pnl[1] is not None and self.transaction_cost_rate != 0])
        print(f"Average Weekly Turnover: ${avg_turnover:.2f}")

        # Exposure Metrics
        avg_long_exposure = np.mean([pnl[2] for pnl in self.pnl_history.values() if pnl[2] is not None])
        avg_short_exposure = np.mean([pnl[3] for pnl in self.pnl_history.values() if pnl[3] is not None])
        print(f"Average Long Exposure: ${avg_long_exposure:.2f}")
        print(f"Average Short Exposure: ${avg_short_exposure:.2f}")

        # Optimization Method Counts
        print(f"\nOptimization Method Usage:")
        print(f" Linear Programming (linprog) Success: {self.optimization_counts.get('linprog_success', 0)} times")
        print(f" Least Squares Fallback: {self.optimization_counts.get('least_squares_fallback', 0)} times")

        # Performance Ratios
        sharpe_ratio, annualized_return, annualized_std = self.compute_sharpe_ratio() or (None, None, None)
        print(f"\nAnnualized Return: {annualized_return * 100:.2f}%")
        print(f"Annualized Volatility: {annualized_std * 100:.2f}%")
        print(
            f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}" if sharpe_ratio is not None else "Annualized Sharpe Ratio: N/A")

        sortino_ratio = self.compute_sortino_ratio()
        print(
            f"Annualized Sortino Ratio: {sortino_ratio:.4f}" if sortino_ratio is not None else "Annualized Sortino Ratio: N/A")

        max_drawdown = self.compute_max_drawdown()
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%" if max_drawdown is not None else "Maximum Drawdown: N/A")

        calmar_ratio = annualized_return / max_drawdown if max_drawdown and max_drawdown > 0 else None
        print(f"Calmar Ratio: {calmar_ratio:.4f}" if calmar_ratio is not None else "Calmar Ratio: N/A")

        # Win Rate and Profit Factor
        pnl_values = [pnl[0] for pnl in self.pnl_history.values() if pnl[0] is not None]
        win_rate = len([p for p in pnl_values if p > 0]) / len(pnl_values) * 100 if pnl_values else 0.0
        print(f"Win Rate: {win_rate:.2f}%")

        profit_factor = self.compute_profit_factor()
        print(f"Profit Factor: {profit_factor:.4f}" if profit_factor is not None else "Profit Factor: N/A")

        # Sector Exposure Analysis
        print("\nAverage Sector Exposure ($):")
        sector_exposures = {sector: [] for sector in set(self.sectors.values())}
        for date in self.weights:
            weights = self.weights[date]
            for sector in sector_exposures:
                sector_mask = np.array([1 if self.sectors[stock] == sector else 0 for stock in self.stocks])
                exposure = np.abs(np.sum(weights * sector_mask))
                sector_exposures[sector].append(exposure)
        for sector, exposures in sector_exposures.items():
            avg_exposure = np.mean(exposures) if exposures else 0.0
            print(f" {sector}: ${avg_exposure:.2f}")

        # Stock Contribution Analysis
        print("\nAverage Stock P&L Contribution ($):")
        stock_pnls = {stock: [] for stock in self.stocks}
        for date in self.rebalance_dates[:-1]:
            actual_returns = self.rebalance_data.get(date, {}).get('actual_returns')
            weights = self.weights.get(date)
            if actual_returns is not None and weights is not None:
                for i, stock in enumerate(self.stocks):
                    if i < len(actual_returns) and i < len(weights):
                        stock_pnls[stock].append(weights[i] * actual_returns[i])
        for stock, pnls in stock_pnls.items():
            avg_pnl = np.mean(pnls) if pnls else 0.0
            print(f" {stock}: ${avg_pnl:.2f}")

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
            print(f" {category}:")
            for factor_name in factors.keys():
                if factor_name in self.factor_data.columns:
                    mean_return = self.factor_data[factor_name].mean()
                    std_return = self.factor_data[factor_name].std()
                    print(f"  {factor_name}: Mean Weekly Return = {mean_return:.6f}, Std = {std_return:.6f}")

        print("\nRebalancing Dates (Showing first/last 5):")
        display_dates = self.rebalance_dates[:5].append(self.rebalance_dates[-5:]) if len(
            self.rebalance_dates) > 10 else self.rebalance_dates
        for i, date in enumerate(display_dates, 1):
            next_date_idx = self.rebalance_dates.get_loc(date) + 1
            days_between = (self.rebalance_dates[next_date_idx] - date).days if next_date_idx < len(
                self.rebalance_dates) else None
            print(f" {i}. {date.date()} (Days to next: {days_between if days_between else 'N/A'})")

        print(
            "\nActual Weekly Stock Returns, Explained Variance, Portfolio Metrics, and Regression R² (First/Last 5 Dates):")
        for date in display_dates:
            data = self.rebalance_data.get(date, {})
            print(f"\n{date.date()}:")
            if data['actual_returns'] is not None:
                print(" Actual Returns:")
                for stock, ret in zip(self.stocks, data['actual_returns']):
                    print(f"  {stock}: {ret:.6f}")
            if data['explained_variance'] is not None:
                print(" Explained Variance Ratios:")
                for i, var in enumerate(data['explained_variance'], 1):
                    print(f"  PC{i}: {var:.4f} ({var * 100:.2f}%)")
            if date in self.pnl_history:
                net_pnl, transaction_costs, long_exposure, short_exposure = self.pnl_history[date]
                print(" Portfolio Metrics:")
                print(f"  Net P&L: ${net_pnl:.2f}" if net_pnl is not None else "  Net P&L: N/A")
                print(
                    f"  Transaction Costs: ${transaction_costs:.2f}" if transaction_costs is not None else "  Transaction Costs: N/A")
                print(f"  Long Exposure: ${long_exposure:.2f}" if long_exposure is not None else "  Long Exposure: N/A")
                print(
                    f"  Short Exposure: ${short_exposure:.2f}" if short_exposure is not None else "  Short Exposure: N/A")
            if date in self.weights:
                print(" Portfolio Weights:")
                for stock, w in zip(self.stocks, self.weights[date]):
                    print(
                        f"  {stock}: ${w:.2f} ({w / self.portfolio_values.get(date, self.initial_capital) * 100:.2f}%)")
            regression = self.regression_results.get(date, {})
            if regression:
                print(" Regression R² Scores:")
                print(" Lasso:")
                for pc_idx in range(self.num_pcs):
                    train_r2 = regression[pc_idx]['train_r2']
                    test_error = regression[pc_idx]['test_error']
                    test_r2 = self.compute_rolling_r2(pc_idx, date, window=10, use_future=False)
                    print(
                        f"  PC{pc_idx + 1}: Train R² = {train_r2:.4f}, Test Error = {test_error if test_error is not None else 'N/A'}, Rolling Test R² = {test_r2 if test_r2 is not None else 'N/A'}")
            else:
                print(" No regression data available")
            self.debug_scaling(date)
        self.save_rebalance_data()

if __name__ == "__main__":
    strategy = PCAFactorStrategy(
        start_date='2021-07-01',
        end_date='2023-08-22',
        rebalance_frequency='W-FRI',
        lookback=252,
        min_trading_days=100,
        num_pcs=5,
        centrality_std=0.13,
        regression_lookback=52,
        min_regression_weeks=42
    )
    strategy.main()