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
        self.factor_data_daily = raw_factor_data['Close'] if isinstance(raw_factor_data.columns,
                                                                       pd.MultiIndex) else raw_factor_data
        self.factor_data_daily = self.factor_data_daily.dropna(axis=1, how='all').dropna(axis=0, how='any')
        self.factor_data = self.factor_data_daily.resample(self.rebalance_frequency).last()
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
                    print(f"  PC{pc_idx + 1}: {r2:.2f}%")
            else:
                print(f"Warning: Insufficient common dates for sector R² calculation at {rebalance_date.date()}")
                self.sector_r2_history[rebalance_date] = [0.0] * self.num_pcs  # Store zeros if calculation fails
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
        sigma = self.compute_pc_std(rebalance_date)
        if sigma is None:
            return None
        num_stocks = min(len(self.stocks), pca_matrix.shape[0])
        num_pcs = min(self.num_pcs, pca_matrix.shape[1])
        pca_matrix = pca_matrix[:num_stocks, :num_pcs]
        pred_pct_change_pc = pred_pct_change_pc[:num_pcs]
        sigma = sigma[:num_pcs]
        # Compute r_hat = V * (ΔPC_pred ⊙ σ_PC)
        r_hat = pca_matrix @ (pred_pct_change_pc * sigma)
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
        for rebalance_date in self.rebalance_dates[:-1]:  # Skip last if no actuals
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
        """Compute stock betas relative to the sector proxy using the lookback period."""
        if rebalance_date not in self.stock_data.index:
            print(f"Warning: Rebalance date {rebalance_date.date()} not in stock data")
            return None
        end_idx = self.stock_data.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback + 1)
        if end_idx - start_idx + 1 < self.min_trading_days:
            print(f"Warning: Insufficient data for betas at {rebalance_date.date()}")
            return None
        prices = self.stock_data.iloc[start_idx:end_idx + 1]
        returns = self.compute_returns(prices)
        if len(returns) < self.min_trading_days:
            print(f"Warning: Insufficient valid returns for betas at {rebalance_date.date()}")
            return None
        sector_prices = self.factor_data_daily[self.sector_proxy].loc[prices.index]
        sector_returns = self.compute_returns(sector_prices)
        common_dates = returns.index.intersection(sector_returns.index)
        if len(common_dates) < self.min_trading_days:
            print(f"Warning: Insufficient common dates for betas at {rebalance_date.date()}")
            return None
        returns = returns.loc[common_dates]
        sector_returns = sector_returns.loc[common_dates].values.reshape(-1, 1)
        betas = []
        for stock in self.stocks:
            stock_returns = returns[stock].values
            model = LinearRegression().fit(sector_returns, stock_returns)
            betas.append(model.coef_[0])
        return np.array(betas)

    def optimize_portfolio(self, r_hat_weighted, C_total, betas):
        """Optimize portfolio weights to maximize expected return subject to market neutral constraints."""
        if r_hat_weighted is None or betas is None:
            return np.zeros(len(self.stocks)), False
        n = len(self.stocks)
        pos_idx = np.where(r_hat_weighted >= 0)[0]
        neg_idx = np.where(r_hat_weighted < 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            print("Warning: Cannot form market neutral portfolio (one side empty)")
            return np.zeros(n), False
        num_vars = n_pos + n_neg
        r_pos = r_hat_weighted[pos_idx]
        r_neg = r_hat_weighted[neg_idx]
        c = np.concatenate((-r_pos, r_neg))
        A_eq_dollar = np.concatenate((np.ones(n_pos), -np.ones(n_neg)))[np.newaxis, :]
        b_eq_dollar = np.array([0.0])
        beta_pos = betas[pos_idx]
        beta_neg = betas[neg_idx]
        A_eq_beta = np.concatenate((beta_pos, -beta_neg))[np.newaxis, :]
        b_eq_beta = np.array([0.0])
        A_eq = np.vstack((A_eq_dollar, A_eq_beta))
        b_eq = np.concatenate((b_eq_dollar, b_eq_beta))
        A_ub = np.ones((1, num_vars))
        b_ub = np.array([C_total])
        bounds = [(0, None)] * num_vars
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w = res.x
            v = np.zeros(n)
            v[pos_idx] = w[:n_pos]
            v[neg_idx] = -w[n_pos:]
            return v, True
        else:
            print(f"Optimization failed: {res.message}")
            return np.zeros(n), False

    def compute_transaction_cost(self, prev_v, new_v):
        """Compute transaction cost based on turnover."""
        turnover = np.sum(np.abs(new_v - prev_v))
        cost = turnover * self.transaction_cost_rate
        return cost

    def compute_profitability_metrics(self):
        """Compute and print profitability metrics."""
        if not self.portfolio_values:
            print("No portfolio data available")
            return
        dates = sorted(self.portfolio_values.keys())
        values = np.array([self.portfolio_values[d] for d in dates])
        weekly_returns = np.diff(values) / values[:-1]
        if len(weekly_returns) == 0:
            print("Insufficient data for metrics")
            return
        # Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        # Sharpe (annualized, assuming risk-free=0)
        mean_ret = np.mean(weekly_returns)
        std_ret = np.std(weekly_returns)
        sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0
        # Calmar
        annual_ret = mean_ret * 52
        calmar = annual_ret / -max_drawdown if max_drawdown < 0 else 0
        # Percent profitable weeks
        pct_profitable = np.mean(weekly_returns > 0) * 100
        # Per-side metrics
        long_pnls = []
        short_pnls = []
        long_amounts = []
        short_amounts = []
        long_returns = []
        short_returns = []
        for i in range(len(dates) - 1):
            date = dates[i]
            next_date = dates[i + 1]
            v = self.weights.get(date, np.zeros(len(self.stocks)))
            actual_returns = self.rebalance_data.get(date, {}).get('actual_returns')
            if actual_returns is None:
                continue
            pnl = np.dot(v, actual_returns)
            long_mask = v > 0
            short_mask = v < 0
            long_pnl = np.sum(v[long_mask] * actual_returns[long_mask])
            short_pnl = np.sum(v[short_mask] * actual_returns[short_mask])
            long_amt = np.sum(v[long_mask])
            short_amt = np.sum(np.abs(v[short_mask]))
            long_ret = long_pnl / long_amt if long_amt > 0 else 0
            short_ret = short_pnl / short_amt if short_amt > 0 else 0
            long_pnls.append(long_pnl)
            short_pnls.append(short_pnl)
            long_returns.append(long_ret)
            short_returns.append(short_ret)
        pct_prof_long = np.mean([p > 0 for p in long_returns if abs(p) > 1e-6]) * 100 if long_returns else 0
        pct_prof_short = np.mean([p > 0 for p in short_returns if abs(p) > 1e-6]) * 100 if short_returns else 0
        avg_weekly_ret = mean_ret * 100
        avg_weekly_long_ret = np.mean(long_returns) * 100 if long_returns else 0
        avg_weekly_short_ret = np.mean(short_returns) * 100 if short_returns else 0
        total_profit = values[-1] - self.initial_capital
        # Print
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Sharpe Ratio (annualized): {sharpe:.2f}")
        print(f"Calmar Ratio: {calmar:.2f}")
        print(f"Percent Profitable Weeks: {pct_profitable:.2f}%")
        print(f"Percent Profitable Long Weeks: {pct_prof_long:.2f}%")
        print(f"Percent Profitable Short Weeks: {pct_prof_short:.2f}%")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Average Weekly Returns: {avg_weekly_ret:.2f}%")
        print(f"Average Weekly Long Returns: {avg_weekly_long_ret:.2f}%")
        print(f"Average Weekly Short Returns: {avg_weekly_short_ret:.2f}%")
        print(f"Optimization Failures: {self.num_opt_fails}")
        # Other info
        total_costs = sum([self.compute_transaction_cost(self.weights.get(dates[i], np.zeros(len(self.stocks))), self.weights.get(dates[i+1], np.zeros(len(self.stocks)))) for i in range(len(dates)-1)])
        print(f"Total Transaction Costs: ${total_costs:.2f}")
        avg_turnover = np.mean([np.sum(np.abs(self.weights.get(dates[i], np.zeros(len(self.stocks))) - self.weights.get(dates[i+1], np.zeros(len(self.stocks))))) / self.portfolio_values.get(dates[i], 1) for i in range(len(dates)-1)]) * 100
        print(f"Average Weekly Turnover: {avg_turnover:.2f}%")

    def plot_portfolio_value_over_time(self):
        """Plot portfolio value over time starting at $10,000."""
        if not self.portfolio_values:
            print("No portfolio data available")
            return
        dates = sorted(self.portfolio_values.keys())
        values = [self.portfolio_values[d] for d in dates]
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, marker='o')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_portfolio_composition_over_time(self):
        """Plot portfolio composition over time with longs above y=0 and shorts below."""
        if not self.weights:
            print("No weights data available")
            return
        dates = sorted(self.weights.keys())
        pos_values = {stock: [] for stock in self.stocks}
        neg_values = {stock: [] for stock in self.stocks}
        for d in dates:
            v = self.weights[d]
            for j, stock in enumerate(self.stocks):
                if v[j] > 0:
                    pos_values[stock].append(v[j])
                    neg_values[stock].append(0)
                elif v[j] < 0:
                    pos_values[stock].append(0)
                    neg_values[stock].append(v[j])
                else:
                    pos_values[stock].append(0)
                    neg_values[stock].append(0)
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.stocks)))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.stackplot(dates, list(pos_values.values()), labels=self.stocks, colors=colors)
        ax.stackplot(dates, list(neg_values.values()), colors=colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title('Portfolio Composition Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Dollar Allocation ($)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
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
        print("\nAverage Percentage of Sector Movement Explained Over Time:")
        for pc_idx in range(self.num_pcs):
            avg_r2 = np.nanmean(r2_data[pc_idx])
            print(f"PC{pc_idx + 1}: {avg_r2:.2f}%")

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

    def main(self):
        """Main function to run the PCA factor strategy."""
        # Download data
        print("Downloading data...")
        self.download_data()

        # Process each rebalance date
        C_total = self.initial_capital
        prev_v = np.zeros(len(self.stocks))
        self.num_opt_fails = 0
        for i, rebalance_date in enumerate(self.rebalance_dates):
            print(f"\nProcessing rebalance date: {rebalance_date.date()}")

            # Compute PCA and centrality
            pca_matrix, explained_variance, stock_std = self.compute_pca_for_rebalance(rebalance_date)
            centrality_vector = self.compute_centrality_for_rebalance(rebalance_date)
            actual_returns = self.compute_actual_returns_for_rebalance(rebalance_date)
            betas = self.compute_betas(rebalance_date)

            # Store rebalance data
            self.rebalance_data[rebalance_date] = {
                'pca_matrix': pca_matrix,
                'centrality_vector': centrality_vector,
                'actual_returns': actual_returns,
                'explained_variance': explained_variance,
                'stock_std': stock_std,
                'betas': betas
            }

            # Apply PnL from previous period if applicable
            if i > 0:
                prev_date = self.rebalance_dates[i - 1]
                actual_returns_prev = self.rebalance_data[prev_date].get('actual_returns')
                if actual_returns_prev is not None:
                    pnl = np.dot(prev_v, actual_returns_prev)
                    self.pnl_history[rebalance_date] = pnl
                    C_total += pnl

            # Train regression models
            self.regression_results[rebalance_date] = self.train_regression_models(rebalance_date)

            # Debug scaling
            self.debug_scaling(rebalance_date)

            # Compute predicted returns and optimize portfolio
            r_hat_weighted = self.compute_weighted_predicted_returns(rebalance_date)
            if r_hat_weighted is not None and betas is not None:
                v, success = self.optimize_portfolio(r_hat_weighted, C_total, betas)
                if not success:
                    self.num_opt_fails += 1
                    v = prev_v  # Hold previous positions if optimization fails
            else:
                v = prev_v
                self.num_opt_fails += 1

            # Compute and apply transaction cost
            cost = self.compute_transaction_cost(prev_v, v)
            C_total -= cost

            # Store portfolio data
            self.portfolio_values[rebalance_date] = C_total
            self.weights[rebalance_date] = v
            prev_v = v

        # Compute actual future PC returns
        self.compute_actuals_future()

        # Save rebalance data
        self.save_rebalance_data()

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_r2_scores()
        self.plot_pred_vs_actual_scatter()
        self.plot_predicted_over_time()
        self.plot_actual_over_time()
        self.plot_sector_r2_over_time()
        self.print_avg_sector_r2()  # Replace with new function name

        # Compute and print metrics
        print("\nProfitability Metrics:")
        self.compute_profitability_metrics()

        # Plot portfolio graphs
        self.plot_portfolio_value_over_time()
        self.plot_portfolio_composition_over_time()
if __name__ == "__main__":
    strategy = PCAFactorStrategy(
        start_date='2019-01-01',
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