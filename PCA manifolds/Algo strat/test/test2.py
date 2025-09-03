import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class FinancialDataLoader:
    def __init__(self, start_date="2022-01-01", end_date="2025-08-22"):
        self.start_date = start_date
        self.end_date = end_date
        self.financial_stocks = [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'BLK',
            'AXP', 'COF', 'USB', 'PNC', 'BK', 'STT', 'TFC'
        ]  # 15 major financial stocks

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

        # Data storage
        self.stock_prices = None
        self.factor_prices = None
        self.stock_daily_returns = None
        self.stock_weekly_returns = None
        self.factor_daily_returns = None
        self.factor_weekly_returns = None
        self.rotation_returns = None
        self.momentum_returns = None
        self.macro_returns = None
        self.sector_rotation_returns = None
        self.volatility_returns = None


    def load_price_data(self):
        """Load historical price data for all stocks and factors"""
        print("Loading stock price data...")
        stock_data = yf.download(
            self.financial_stocks,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True  # Use auto_adjust instead of Adj Close
        )
        self.stock_prices = stock_data['Close']  # Use 'Close' instead of 'Adj Close'

        print("Loading factor price data...")
        # Get all unique factor tickers
        all_factor_tickers = set(self.factors)
        for category in [self.rotation_pairs, self.momentum_factors,
                         self.macro_factors, self.sector_rotation_factors,
                         self.volatility_factors]:
            for pair in category.values():
                all_factor_tickers.update(pair)

        factor_data = yf.download(
            list(all_factor_tickers),
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True  # Use auto_adjust instead of Adj Close
        )
        self.factor_prices = factor_data['Close']  # Use 'Close' instead of 'Adj Close'

        # Handle VIX separately since it's not a price but an index value
        if '^VIX' in self.factor_prices.columns:
            vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, auto_adjust=True)['Close']
            self.factor_prices['^VIX'] = vix_data

        return self.stock_prices, self.factor_prices

    def calculate_returns(self):
        """Calculate daily and weekly returns for all instruments"""
        print("Calculating returns...")

        # Stock returns
        self.stock_daily_returns = self.stock_prices.pct_change().dropna()
        self.stock_weekly_returns = self.stock_prices.resample('W-FRI').last().pct_change().dropna()

        # Factor returns (both daily and weekly)
        self.factor_daily_returns = self.factor_prices.pct_change().dropna()
        self.factor_weekly_returns = self.factor_prices.resample('W-FRI').last().pct_change().dropna()

        # Calculate all factor categories using weekly returns
        self._calculate_rotation_returns(weekly=True)
        self._calculate_momentum_returns(weekly=True)
        self._calculate_macro_returns(weekly=True)
        self._calculate_sector_rotation_returns(weekly=True)
        self._calculate_volatility_returns(weekly=True)

        return self.stock_daily_returns, self.stock_weekly_returns

    def _calculate_rotation_returns(self, weekly=False):
        """Calculate rotation pair returns - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        self.rotation_returns = pd.DataFrame(index=source_data.index)

        for name, (long_ticker, short_ticker) in self.rotation_pairs.items():
            if long_ticker in source_data.columns and short_ticker in source_data.columns:
                self.rotation_returns[name] = (
                        source_data[long_ticker] - source_data[short_ticker]
                )

    def _calculate_momentum_returns(self, weekly=False):
        """Calculate momentum factor returns - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        self.momentum_returns = pd.DataFrame(index=source_data.index)

        for name, (long_ticker, short_ticker) in self.momentum_factors.items():
            if long_ticker in source_data.columns and short_ticker in source_data.columns:
                self.momentum_returns[name] = (
                        source_data[long_ticker] - source_data[short_ticker]
                )

    def _calculate_macro_returns(self, weekly=False):
        """Calculate macro factor returns - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        self.macro_returns = pd.DataFrame(index=source_data.index)

        for name, (long_ticker, short_ticker) in self.macro_factors.items():
            if long_ticker in source_data.columns and short_ticker in source_data.columns:
                self.macro_returns[name] = (
                        source_data[long_ticker] - source_data[short_ticker]
                )

    def _calculate_sector_rotation_returns(self, weekly=False):
        """Calculate sector rotation factor returns - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        self.sector_rotation_returns = pd.DataFrame(index=source_data.index)

        for name, (long_ticker, short_ticker) in self.sector_rotation_factors.items():
            if long_ticker in source_data.columns and short_ticker in source_data.columns:
                self.sector_rotation_returns[name] = (
                        source_data[long_ticker] - source_data[short_ticker]
                )

    def _calculate_volatility_returns(self, weekly=False):
        """Calculate volatility factor returns - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        self.volatility_returns = pd.DataFrame(index=source_data.index)

        for name, (long_ticker, short_ticker) in self.volatility_factors.items():
            if long_ticker in source_data.columns and short_ticker in source_data.columns:
                # For VIX, use the raw weekly returns
                if long_ticker == '^VIX' or short_ticker == '^VIX':
                    if long_ticker == '^VIX':
                        long_data = source_data[long_ticker]
                    else:
                        long_data = source_data[long_ticker]

                    if short_ticker == '^VIX':
                        short_data = source_data[short_ticker]
                    else:
                        short_data = source_data[short_ticker]

                    self.volatility_returns[name] = long_data - short_data
                else:
                    self.volatility_returns[name] = (
                            source_data[long_ticker] - source_data[short_ticker]
                    )

    def get_all_factor_data(self, weekly=False):
        """Combine all factor data into single DataFrame - weekly or daily"""
        source_data = self.factor_weekly_returns if weekly else self.factor_daily_returns
        all_factors = {}

        # Add basic factors
        for factor in self.factors:
            if factor in source_data.columns:
                all_factors[factor] = source_data[factor]

        # Add category factors (use the already calculated weekly versions)
        categories = [
            self.rotation_returns,
            self.momentum_returns,
            self.macro_returns,
            self.sector_rotation_returns,
            self.volatility_returns
        ]

        for category_df in categories:
            if category_df is not None:
                for col in category_df.columns:
                    all_factors[col] = category_df[col]

        return pd.DataFrame(all_factors).dropna()


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RebalanceManager:
    def __init__(self, stock_weekly_returns, lookback_weeks=52):
        self.stock_weekly_returns = stock_weekly_returns
        self.lookback_weeks = lookback_weeks
        self.rebalance_dates = None
        self.valid_rebalance_dates = None

    def generate_rebalance_dates(self):
        """Generate weekly rebalance dates (Fridays)"""
        # Get all Fridays from the weekly returns index
        fridays = self.stock_weekly_returns.index

        # Only include dates where we have sufficient lookback data
        self.valid_rebalance_dates = [
            date for date in fridays
            if self._has_sufficient_lookback(date)
        ]

        self.rebalance_dates = self.valid_rebalance_dates
        return self.rebalance_dates

    def _has_sufficient_lookback(self, date):
        """Check if sufficient lookback data exists for a given date"""
        date_index = self.stock_weekly_returns.index.get_loc(date)
        return date_index >= self.lookback_weeks

    def get_lookback_data(self, rebalance_date):
        """
        Extract point-in-time lookback data for a given rebalance date

        Returns:
        - lookback_returns: 52 weeks of weekly returns ending on rebalance_date
        - lookback_dates: The actual dates used for lookback
        """
        # Find the position of the rebalance date
        date_idx = self.stock_weekly_returns.index.get_loc(rebalance_date)

        # Extract lookback period (52 weeks including current week)
        lookback_start_idx = date_idx - self.lookback_weeks + 1
        lookback_end_idx = date_idx + 1  # Include current week

        lookback_returns = self.stock_weekly_returns.iloc[lookback_start_idx:lookback_end_idx]
        lookback_dates = self.stock_weekly_returns.index[lookback_start_idx:lookback_end_idx]

        return lookback_returns, lookback_dates

    def validate_rebalance_schedule(self, sample_date=None):
        """Validate the rebalance schedule and print comprehensive report"""
        print("\n" + "=" * 60)
        print("REBALANCE SCHEDULE VALIDATION")
        print("=" * 60)

        if sample_date is None:
            # Use a date from the middle of the range
            sample_date = self.rebalance_dates[len(self.rebalance_dates) // 2]

        print(f"\nSample rebalance date: {sample_date}")

        # Basic schedule info
        print(f"\n1. SCHEDULE INFO:")
        print(f"   Total available Fridays: {len(self.stock_weekly_returns)}")
        print(f"   Valid rebalance dates: {len(self.rebalance_dates)}")
        print(f"   First rebalance: {self.rebalance_dates[0]}")
        print(f"   Last rebalance: {self.rebalance_dates[-1]}")
        print(f"   Lookback weeks: {self.lookback_weeks}")

        # Sample lookback data validation
        print(f"\n2. SAMPLE LOOKBACK DATA for {sample_date}:")
        lookback_returns, lookback_dates = self.get_lookback_data(sample_date)

        print(f"   Lookback period: {lookback_dates[0]} to {lookback_dates[-1]}")
        print(f"   Lookback data shape: {lookback_returns.shape}")
        print(f"   Number of stocks in lookback: {lookback_returns.shape[1]}")

        # Check data completeness
        print(f"\n3. DATA COMPLETENESS CHECK:")
        missing_values = lookback_returns.isnull().sum().sum()
        print(f"   Total missing values in lookback: {missing_values}")

        if missing_values > 0:
            stocks_with_missing = lookback_returns.isnull().sum()
            stocks_with_missing = stocks_with_missing[stocks_with_missing > 0]
            print(f"   Stocks with missing data: {len(stocks_with_missing)}")
            for stock, count in stocks_with_missing.items():
                print(f"     {stock}: {count} missing values")

        # Statistical validation
        print(f"\n4. STATISTICAL VALIDATION:")
        print(f"   Mean returns by stock:")
        mean_returns = lookback_returns.mean().sort_values(ascending=False)
        for stock, mean_return in mean_returns.head(3).items():
            print(f"     {stock}: {mean_return:.6f}")
        print(f"   ...")
        for stock, mean_return in mean_returns.tail(3).items():
            print(f"     {stock}: {mean_return:.6f}")

        print(f"\n   Return volatility by stock:")
        std_returns = lookback_returns.std().sort_values(ascending=False)
        for stock, std_return in std_returns.head(3).items():
            print(f"     {stock}: {std_return:.6f}")

        # Point-in-time validation
        print(f"\n5. POINT-IN-TIME VALIDATION:")
        print(f"   Ensuring no lookahead bias...")

        # Check that all lookback dates are before rebalance date
        all_dates_before = all(date <= sample_date for date in lookback_dates)
        print(f"   All lookback dates <= rebalance date: {all_dates_before}")

        # Check that no future data is included
        future_dates = [date for date in lookback_dates if date > sample_date]
        print(f"   Future dates in lookback: {len(future_dates)}")

        if not all_dates_before or len(future_dates) > 0:
            print(f"   ⚠️  WARNING: Potential lookahead bias detected!")

        print("=" * 60)

        return lookback_returns, lookback_dates


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalyzer:
    def __init__(self, lookback_weeks=52):
        self.lookback_weeks = lookback_weeks
        self.pca_models = {}  # Store PCA models for each rebalance date
        self.pc_loadings = {}  # Store PC loadings for each date
        self.pc_returns = {}  # Store PC returns for each date

    def perform_pca_analysis(self, lookback_returns, rebalance_date):
        """
        Perform PCA analysis on standardized stock returns

        Returns:
        - pca: Fitted PCA model
        - loadings: PCA loadings matrix
        - pc_returns: Principal component returns
        - standardized_returns: Standardized returns used for PCA
        """
        # Standardize returns (mean=0, std=1)
        scaler = StandardScaler()
        standardized_returns = scaler.fit_transform(lookback_returns)

        # Perform PCA
        pca = PCA()
        pc_scores = pca.fit_transform(standardized_returns)

        # Get loadings (eigenvectors)
        loadings = pca.components_.T  # Transpose to get stocks x PCs

        # Create DataFrame for loadings
        loadings_df = pd.DataFrame(
            loadings,
            index=lookback_returns.columns,
            columns=[f'PC{i + 1}' for i in range(loadings.shape[1])]
        )

        # Rescale loadings to preserve original return magnitudes
        # Divide by stock return standard deviations
        stock_stds = lookback_returns.std()
        rescaled_loadings = loadings_df.div(stock_stds, axis=0)

        # Flip signs if sum of loadings is negative
        for pc in rescaled_loadings.columns:
            if rescaled_loadings[pc].sum() < 0:
                rescaled_loadings[pc] = -rescaled_loadings[pc]

        # Calculate PC returns by projecting returns onto loadings
        pc_returns = pd.DataFrame(
            pc_scores,
            index=lookback_returns.index,
            columns=[f'PC{i + 1}' for i in range(pc_scores.shape[1])]
        )

        # Store results
        self.pca_models[rebalance_date] = {
            'pca': pca,
            'scaler': scaler,
            'original_loadings': loadings_df,
            'rescaled_loadings': rescaled_loadings,
            'explained_variance': pca.explained_variance_ratio_,
            'eigenvalues': pca.explained_variance_
        }

        self.pc_loadings[rebalance_date] = rescaled_loadings
        self.pc_returns[rebalance_date] = pc_returns

        return pca, rescaled_loadings, pc_returns, standardized_returns
    def validate_pca_analysis(self, lookback_returns, rebalance_date, pca, loadings, pc_returns, standardized_returns):
        """Validate PCA results and print comprehensive report"""
        print(f"\nPCA VALIDATION for {rebalance_date}")
        print("=" * 50)

        # Basic PCA info
        print(f"1. PCA MODEL INFO:")
        print(f"   Number of components: {pca.n_components_}")
        print(f"   Number of stocks: {loadings.shape[0]}")
        print(f"   Lookback weeks: {self.lookback_weeks}")

        # Explained variance
        print(f"\n2. EXPLAINED VARIANCE:")
        cumulative_variance = 0
        for i, (variance, eigenvalue) in enumerate(zip(
                pca.explained_variance_ratio_[:10],  # First 10 components
                pca.explained_variance_[:10]
        )):
            cumulative_variance += variance
            print(f"   PC{i + 1}: {variance:.3%} (λ={eigenvalue:.4f})")
            if i == 4:  # Show first 5 in detail
                print(f"   ...")
                break

        print(f"   Cumulative (first 5): {cumulative_variance:.3%}")
        print(f"   Total variance explained: {pca.explained_variance_ratio_.sum():.3%}")

        # Loadings analysis
        print(f"\n3. LOADINGS ANALYSIS:")
        print(f"   PC1 - Most influential stocks:")
        pc1_loadings = loadings['PC1'].sort_values(ascending=False)
        for stock, loading in pc1_loadings.head(3).items():
            print(f"     {stock}: {loading:.4f}")
        print(f"   PC1 - Least influential stocks:")
        for stock, loading in pc1_loadings.tail(3).items():
            print(f"     {stock}: {loading:.4f}")

        # PC returns statistics
        print(f"\n4. PC RETURNS STATISTICS:")
        pc_stats = pc_returns.iloc[:, :5].describe()  # First 5 PCs
        for pc in pc_stats.columns:
            mean = pc_stats.loc['mean', pc]
            std = pc_stats.loc['std', pc]
            print(f"   {pc}: μ={mean:.6f}, σ={std:.6f}")

        # Orthogonality check (PCs should be uncorrelated)
        print(f"\n5. ORTHOGONALITY CHECK:")
        pc_corr = pc_returns.iloc[:, :5].corr()  # Correlation of first 5 PCs
        off_diag_corr = pc_corr.values[np.triu_indices_from(pc_corr, k=1)]
        max_off_diag = np.max(np.abs(off_diag_corr))
        print(f"   Max off-diagonal correlation: {max_off_diag:.6f}")
        print(f"   PCs are {'approximately orthogonal' if max_off_diag < 0.1 else 'correlated'}")

        # Reconstruction check
        # In the validate_pca_analysis method, replace the reconstruction check:
        print(f"\n6. RECONSTRUCTION CHECK:")
        # Reconstruct standardized returns using first 5 PCs
        reconstructed_standardized = np.dot(pc_returns.iloc[:, :5], loadings.iloc[:, :5].T)
        reconstruction_error = np.mean((standardized_returns - reconstructed_standardized) ** 2)
        print(f"   Mean squared reconstruction error (5 PCs): {reconstruction_error:.6f}")

        # Additional check: Compare to original scale
        scaler = StandardScaler()
        scaler.fit(lookback_returns)  # Fit the scaler again to get parameters
        reconstructed_original = reconstructed_standardized * scaler.scale_ + scaler.mean_
        original_reconstruction_error = np.mean((lookback_returns - reconstructed_original) ** 2)
        print(f"   Mean squared error on original scale: {original_reconstruction_error:.6f}")

        print("=" * 50)

        return True


import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np


class FactorRegression:
    def __init__(self, lookback_weeks=52):
        self.lookback_weeks = lookback_weeks
        self.regression_models = {}  # Store regression models for each date and PC
        self.selected_factors = {}  # Store selected factors for each PC
        self.regression_coefficients = {}  # Store regression coefficients

    def prepare_factor_data(self, all_factors, lookback_dates, weekly=False):
        """Prepare factor data for lookback period with proper weekly alignment"""
        if weekly:
            # For weekly factors, ensure we have proper weekly alignment
            # Convert lookback_dates to weekly frequency if needed
            weekly_lookback_dates = pd.DatetimeIndex(lookback_dates).normalize()

            # Find intersection with available weekly factor dates
            available_dates = weekly_lookback_dates.intersection(all_factors.index)

            if len(available_dates) < len(weekly_lookback_dates):
                missing_dates = len(weekly_lookback_dates) - len(available_dates)
                print(f"Warning: {missing_dates} weekly dates missing from factor data")

            # Get factor data for available dates
            factor_data = all_factors.loc[available_dates].copy()

        else:
            # Original daily logic (for backward compatibility)
            lookback_dates = pd.DatetimeIndex(lookback_dates).tz_localize(all_factors.index.tz)
            available_dates = lookback_dates.intersection(all_factors.index)

            if len(available_dates) < len(lookback_dates):
                missing_dates = len(lookback_dates) - len(available_dates)
                print(f"Warning: {missing_dates} dates missing from factor data")

            factor_data = all_factors.loc[available_dates].copy()

        return factor_data

    def run_lasso_linear_regression(self, pc_returns, factor_data, rebalance_date, pc_number):
        """
        Simple and robust Lasso + Linear Regression with proper regularization

        Returns:
        - model: Fitted Linear Regression model
        - selected_factors: List of selected factors
        - coefficients: Regression coefficients
        - r_squared: R-squared value
        - cv_score: Cross-validation score
        """
        # Align PC returns with factor data dates
        aligned_pc_returns = pc_returns.loc[factor_data.index]

        # Standardize both PC returns and factors
        pc_scaler = StandardScaler()
        factor_scaler = StandardScaler()

        y = aligned_pc_returns[f'PC{pc_number}'].values.reshape(-1, 1)
        X = factor_data.values

        y_standardized = pc_scaler.fit_transform(y).ravel()
        X_standardized = factor_scaler.fit_transform(X)

        # Use time-series cross-validation with fewer splits for stability
        n_splits = min(3, len(X_standardized) // 15)  # Fewer splits for more stable CV
        if n_splits < 2:
            n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Step 1: Lasso for feature selection with more conservative alpha range
        if pc_number == 1:
            # For PC1, use slightly more aggressive regularization
            alpha_range = np.logspace(-3, -1, 20)
        else:
            # For other PCs, use more conservative regularization
            alpha_range = np.logspace(-4, -2, 20)

        lasso_cv = LassoCV(
            alphas=alpha_range,
            cv=tscv,
            max_iter=5000,
            random_state=42,
            selection='random'
        )

        lasso_cv.fit(X_standardized, y_standardized)

        # Get selected features
        selected_mask = lasso_cv.coef_ != 0
        selected_factors = factor_data.columns[selected_mask].tolist()
        selected_indices = np.where(selected_mask)[0]

        # If too many features selected, use top features by coefficient magnitude
        if len(selected_indices) > 10:
            coef_magnitude = np.abs(lasso_cv.coef_[selected_mask])
            top_indices = np.argsort(coef_magnitude)[-10:]  # Top 10 features
            selected_indices = selected_indices[top_indices]
            selected_factors = [factor_data.columns[i] for i in selected_indices]
            selected_mask = np.zeros_like(lasso_cv.coef_, dtype=bool)
            selected_mask[selected_indices] = True

        # If no features selected, use top 5 features by correlation
        if len(selected_indices) == 0:
            correlations = []
            for i in range(X_standardized.shape[1]):
                corr = np.corrcoef(X_standardized[:, i], y_standardized)[0, 1]
                correlations.append((i, abs(corr)))

            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [correlations[i][0] for i in range(min(5, len(correlations)))]
            selected_factors = [factor_data.columns[i] for i in selected_indices]
            selected_mask = np.zeros_like(lasso_cv.coef_, dtype=bool)
            selected_mask[selected_indices] = True

        # Step 2: Linear Regression on selected features
        X_selected = X_standardized[:, selected_indices]

        linear_model = LinearRegression()
        linear_model.fit(X_selected, y_standardized)

        # Calculate R-squared
        y_pred = linear_model.predict(X_selected)
        r_squared = r2_score(y_standardized, y_pred)

        # Simple cross-validation - just use the last fold for stability
        cv_r2_scores = []
        cv_mse_scores = []

        for train_idx, val_idx in tscv.split(X_selected):
            # Use only the last fold to avoid over-optimistic CV
            if val_idx[-1] == len(X_selected) - 1:  # Last validation fold
                X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                y_train, y_val = y_standardized[train_idx], y_standardized[val_idx]

                temp_model = LinearRegression()
                temp_model.fit(X_train, y_train)

                val_pred = temp_model.predict(X_val)
                cv_r2_scores.append(r2_score(y_val, val_pred))
                cv_mse_scores.append(mean_squared_error(y_val, val_pred))
                break

        cv_r2_mean = np.mean(cv_r2_scores) if cv_r2_scores else 0
        cv_score = -np.mean(cv_mse_scores) if cv_mse_scores else 0

        # Get coefficients
        coefficients = linear_model.coef_

        # Store results
        if rebalance_date not in self.regression_models:
            self.regression_models[rebalance_date] = {}
            self.selected_factors[rebalance_date] = {}
            self.regression_coefficients[rebalance_date] = {}

        self.regression_models[rebalance_date][pc_number] = {
            'model': linear_model,
            'lasso_model': lasso_cv,
            'pc_scaler': pc_scaler,
            'factor_scaler': factor_scaler,
            'selected_indices': selected_indices,
            'optimal_alpha': lasso_cv.alpha_,
            'cv_r2_mean': cv_r2_mean,
            'cv_score': cv_score
        }

        self.selected_factors[rebalance_date][pc_number] = selected_factors
        self.regression_coefficients[rebalance_date][pc_number] = coefficients

        return linear_model, selected_factors, coefficients, r_squared, cv_score

    def validate_simple_regression(self, pc_returns, factor_data, rebalance_date,
                                   pc_number, model, selected_factors, coefficients, r_squared, cv_score):
        """Validate simple regression results"""
        print(f"\nSIMPLE REGRESSION VALIDATION for PC{pc_number} on {rebalance_date}")
        print("=" * 60)

        stored_data = self.regression_models[rebalance_date][pc_number]
        print(f"1. MODEL INFO:")
        print(f"   Target: PC{pc_number}")
        print(f"   Lookback weeks: {len(factor_data)}")
        print(f"   Optimal alpha: {stored_data['optimal_alpha']:.6f}")
        print(f"   Training R²: {r_squared:.4f}")
        print(f"   CV R²: {stored_data['cv_r2_mean']:.4f}")

        print(f"\n2. SELECTED FACTORS ({len(selected_factors)}):")
        # Show factors sorted by importance
        feature_importance = sorted(zip(selected_factors, coefficients),
                                    key=lambda x: abs(x[1]), reverse=True)

        for factor, coef in feature_importance[:8]:  # Show top 8
            significance = "✓" if abs(coef) > 0.1 else "○"
            print(f"   {significance} {factor}: {coef:.6f}")

        print(f"\n3. MODEL ROBUSTNESS:")
        print(f"   Features selected: {len(selected_factors)}")
        print(f"   CV stability: {'Good' if stored_data['cv_r2_mean'] > -0.5 else 'Poor'}")

        print("=" * 60)

        return True


class PCPredictor:
    def __init__(self):
        self.pc_predictions = {}  # Store PC predictions for each date
        self.prediction_errors = {}  # Store prediction errors

    def predict_pc_returns(self, regression_models, current_factors_weekly, rebalance_date, pc_volatility):
        """
        Predict next week's PC returns using weekly factor data

        Returns:
        - predicted_returns: Dictionary of predicted PC returns
        - prediction_details: Detailed prediction information
        """
        predicted_returns = {}
        prediction_details = {}

        for pc_number in range(1, 6):  # Predict for PC1 to PC5
            pc_key = f'PC{pc_number}'

            if (rebalance_date in regression_models and
                    pc_number in regression_models[rebalance_date]):
                model_data = regression_models[rebalance_date][pc_number]
                model = model_data['model']
                factor_scaler = model_data['factor_scaler']
                pc_scaler = model_data['pc_scaler']
                selected_indices = model_data['selected_indices']

                # Prepare current weekly factor data
                current_factor_values = current_factors_weekly.values.reshape(1, -1)
                current_factors_standardized = factor_scaler.transform(current_factor_values)

                # Select only the features used in training
                X_current = current_factors_standardized[:, selected_indices]

                # Make prediction
                pred_scaled = model.predict(X_current)[0]

                # Convert back to original scale
                pred_original = pc_scaler.inverse_transform([[pred_scaled]])[0][0]

                # Store prediction
                predicted_returns[pc_key] = pred_original
                prediction_details[pc_key] = {
                    'predicted_return': pred_original,
                    'predicted_std_move': pred_original / pc_volatility[pc_key],
                    'features_used': len(selected_indices),
                    'pc_volatility': pc_volatility[pc_key]
                }

        self.pc_predictions[rebalance_date] = predicted_returns
        return predicted_returns, prediction_details

    def validate_predictions(self, predicted_returns, actual_next_returns, pc_volatility, rebalance_date):
        """Validate prediction accuracy"""
        print(f"\nPC PREDICTION VALIDATION for {rebalance_date}")
        print("=" * 60)

        print(f"1. PREDICTED VS ACTUAL RETURNS:")
        for pc in range(1, 6):
            pc_key = f'PC{pc}'
            if pc_key in predicted_returns:
                pred = predicted_returns[pc_key]
                actual = actual_next_returns.get(pc_key, np.nan)
                error = pred - actual if not np.isnan(actual) else np.nan

                print(f"   {pc_key}:")
                print(f"     Predicted: {pred:.6f}")
                print(f"     Actual:    {actual:.6f}" if not np.isnan(actual) else "     Actual:    N/A")
                print(f"     Error:     {error:.6f}" if not np.isnan(error) else "     Error:     N/A")
                if not np.isnan(actual):
                    print(f"     Error (%):  {(error / abs(actual) * 100 if actual != 0 else 0):.1f}%")

        print(f"\n2. PREDICTION MAGNITUDE (in standard deviations):")
        for pc in range(1, 6):
            pc_key = f'PC{pc}'
            if pc_key in predicted_returns:
                std_move = predicted_returns[pc_key] / pc_volatility[pc_key]
                print(f"   {pc_key}: {std_move:.2f}σ movement predicted")

        print(f"\n3. PREDICTION CONFIDENCE:")
        large_moves = sum(1 for pc in range(1, 6)
                          if f'PC{pc}' in predicted_returns and
                          abs(predicted_returns[f'PC{pc}'] / pc_volatility[f'PC{pc}']) > 1.0)
        print(f"   PCs with >1σ predicted moves: {large_moves}/5")

        print("=" * 60)

        return True


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


class StockPredictor:
    def __init__(self):
        self.stock_predictions = {}
        self.centrality_vectors = {}
        self.actual_vs_predicted = {}  # Store actual vs predicted for each rebalance
        self.training_r2_history = {}  # Store training R² for each rebalance

    def calculate_centrality_vector(self, correlation_matrix):
        """Calculate eigenvector centrality from correlation matrix"""
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
        dominant_idx = np.argmax(np.abs(eigenvalues))
        centrality = np.abs(eigenvectors[:, dominant_idx])

        # Normalize to mean=1.0, std=0.13
        centrality = centrality / centrality.mean()
        if centrality.std() > 0:
            centrality = 1.0 + 0.13 * (centrality - centrality.mean()) / centrality.std()

        return pd.Series(centrality, index=correlation_matrix.index)

    def predict_stock_returns(self, pc_predictions, pc_loadings, pc_volatility,
                              correlation_matrix, rebalance_date):
        """
        Combine PC predictions with loadings and centrality weighting
        """
        # Calculate centrality vector
        centrality = self.calculate_centrality_vector(correlation_matrix)
        self.centrality_vectors[rebalance_date] = centrality

        # Initialize predictions
        stock_preds = pd.Series(0.0, index=pc_loadings.index)

        # Combine PC predictions
        for pc_num in range(1, 6):
            pc_key = f'PC{pc_num}'
            if pc_key in pc_predictions:
                # Add contribution from this PC
                stock_preds += pc_loadings[pc_key] * pc_predictions[pc_key]

        # Apply centrality weighting
        centrality_weighted = stock_preds * centrality

        self.stock_predictions[rebalance_date] = centrality_weighted
        return centrality_weighted

    def record_actual_vs_predicted(self, rebalance_date, predicted_returns, actual_returns):
        """Store actual vs predicted returns for validation"""
        aligned_data = pd.DataFrame({
            'predicted': predicted_returns,
            'actual': actual_returns
        }).dropna()

        self.actual_vs_predicted[rebalance_date] = aligned_data
        return aligned_data

    def record_training_r2(self, rebalance_date, training_r2_scores):
        """Store training R² scores for each PC"""
        self.training_r2_history[rebalance_date] = training_r2_scores

    def plot_predicted_vs_actual_scatter(self, rebalance_date, figsize=(10, 6)):
        """Plot predicted vs actual stock returns scatter plot"""
        if rebalance_date not in self.actual_vs_predicted:
            print(f"No actual vs predicted data for {rebalance_date}")
            return

        data = self.actual_vs_predicted[rebalance_date]

        plt.figure(figsize=figsize)
        plt.scatter(data['predicted'], data['actual'], alpha=0.6, s=50)

        # Add regression line
        slope, intercept, r_value, p_value, std_err = linregress(data['predicted'], data['actual'])
        x_range = np.linspace(data['predicted'].min(), data['predicted'].max(), 100)
        plt.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2,
                 label=f'R² = {r_value ** 2:.3f}')

        # Add identity line
        plt.plot(x_range, x_range, 'k--', alpha=0.5, label='Perfect prediction')

        plt.xlabel('Predicted Weekly Returns')
        plt.ylabel('Actual Weekly Returns')
        plt.title(f'Predicted vs Actual Stock Returns - {rebalance_date.strftime("%Y-%m-%d")}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return r_value ** 2

    def plot_out_of_sample_r2_over_time(self, stock_list=None, figsize=(12, 6)):
        """Plot R² between predicted and actual returns over time for selected stocks"""
        if not self.actual_vs_predicted:
            print("No out-of-sample data available")
            return

        # Prepare data
        r2_values = []
        dates = []

        for date, data in self.actual_vs_predicted.items():
            if stock_list:
                # Filter for selected stocks
                filtered_data = data[data.index.isin(stock_list)]
                if len(filtered_data) < 2:  # Need at least 2 points for R²
                    continue
                actual = filtered_data['actual']
                predicted = filtered_data['predicted']
            else:
                actual = data['actual']
                predicted = data['predicted']

            if len(actual) >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(predicted, actual)
                r2_values.append(r_value ** 2)
                dates.append(date)

        if not dates:
            print("Insufficient data for R² calculation")
            return

        plt.figure(figsize=figsize)
        plt.plot(dates, r2_values, 'o-', linewidth=2, markersize=4)
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Rebalance Date')
        plt.ylabel('R² (Out-of-Sample)')
        plt.title('Out-of-Sample Predictive Power Over Time')
        plt.grid(True, alpha=0.3)

        # Add rolling average
        if len(r2_values) > 10:
            rolling_r2 = pd.Series(r2_values, index=dates).rolling(window=10).mean()
            plt.plot(rolling_r2.index, rolling_r2.values, 'r-', linewidth=2,
                     label='10-week Rolling Avg')
            plt.legend()

        plt.tight_layout()
        plt.show()

        return pd.Series(r2_values, index=dates)

    def plot_training_r2_over_time(self, pc_numbers=None, figsize=(12, 8)):
        """Plot training R² for each PC over time"""
        if not self.training_r2_history:
            print("No training R² data available")
            return

        # Prepare data
        training_r2_df = pd.DataFrame()

        for date, r2_scores in self.training_r2_history.items():
            for pc_num, r2 in r2_scores.items():
                training_r2_df.loc[date, f'PC{pc_num}'] = r2

        if pc_numbers:
            columns_to_plot = [f'PC{pc_num}' for pc_num in pc_numbers if f'PC{pc_num}' in training_r2_df.columns]
        else:
            columns_to_plot = training_r2_df.columns

        if not columns_to_plot:
            print("No valid PC numbers specified")
            return

        plt.figure(figsize=figsize)

        for column in columns_to_plot:
            plt.plot(training_r2_df.index, training_r2_df[column], 'o-',
                     linewidth=2, markersize=3, label=column)

        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Rebalance Date')
        plt.ylabel('Training R²')
        plt.title('Training R² for Principal Components Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add average line
        avg_r2 = training_r2_df[columns_to_plot].mean(axis=1)
        plt.plot(avg_r2.index, avg_r2.values, 'k-', linewidth=3,
                 label='Average R²', alpha=0.8)
        plt.legend()

        plt.tight_layout()
        plt.show()

        return training_r2_df

    def validate_stock_predictions(self, rebalance_date, predicted_returns, actual_returns):
        """Validate stock prediction accuracy"""
        print(f"\nSTOCK PREDICTION VALIDATION for {rebalance_date}")
        print("=" * 60)

        # Calculate R²
        aligned_data = self.record_actual_vs_predicted(rebalance_date, predicted_returns, actual_returns)

        if len(aligned_data) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(
                aligned_data['predicted'], aligned_data['actual']
            )
            r2 = r_value ** 2

            print(f"Out-of-sample R²: {r2:.4f}")
            print(f"Number of stocks with predictions: {len(aligned_data)}")
            print(f"Correlation: {r_value:.4f}")

            # Top predictions
            print(f"\nTop 5 predicted winners:")
            top_winners = predicted_returns.sort_values(ascending=False).head()
            for stock, pred in top_winners.items():
                actual = actual_returns.get(stock, np.nan)
                print(f"  {stock}: Pred={pred:.4f}, Actual={actual:.4f}")

            print(f"\nTop 5 predicted losers:")
            top_losers = predicted_returns.sort_values().head()
            for stock, pred in top_losers.items():
                actual = actual_returns.get(stock, np.nan)
                print(f"  {stock}: Pred={pred:.4f}, Actual={actual:.4f}")

        else:
            print("Insufficient data for R² calculation")

        print("=" * 60)
        return r2 if len(aligned_data) >= 2 else np.nan


def main():
    """Main function to run all steps with weekly factor integration"""
    print("Financial Strategy Backtest - Steps 1-6 Implementation")
    print("=" * 60)

    # Initialize data loader
    data_loader = FinancialDataLoader(
        start_date="2018-01-01",
        end_date="2025-08-22"
    )

    # Load price data
    stock_prices, factor_prices = data_loader.load_price_data()

    # Calculate returns
    stock_daily_returns, stock_weekly_returns = data_loader.calculate_returns()

    # Get weekly factor data
    all_factors_weekly = data_loader.get_all_factor_data(weekly=True)
    print(f"Weekly factors shape: {all_factors_weekly.shape}")

    # Step 2: Rebalance schedule generation
    rebalance_manager = RebalanceManager(stock_weekly_returns, lookback_weeks=52)
    rebalance_dates = rebalance_manager.generate_rebalance_dates()

    # Use a sample date for demonstration
    sample_date = rebalance_dates[len(rebalance_dates) // 2]
    print(f"Using sample date for analysis: {sample_date}")

    # Get lookback data for PCA
    lookback_returns, lookback_dates = rebalance_manager.get_lookback_data(sample_date)

    # Step 3: PCA Analysis
    pca_analyzer = PCAAnalyzer(lookback_weeks=52)
    pca, loadings, pc_returns, standardized_returns = pca_analyzer.perform_pca_analysis(lookback_returns, sample_date)

    # Step 4: Factor Regression with Weekly Factors
    factor_regressor = FactorRegression(lookback_weeks=52)

    # Prepare weekly factor data
    factor_data = factor_regressor.prepare_factor_data(
        all_factors_weekly, lookback_dates, weekly=True
    )

    # Run regression for first 5 PCs
    for pc_num in range(1, 6):
        model, selected_factors, coefficients, r_squared, cv_score = factor_regressor.run_lasso_linear_regression(
            pc_returns, factor_data, sample_date, pc_num
        )

    print(f"Weekly factor regression completed successfully for {sample_date}!")

    # Step 5: PC Return Prediction
    pc_predictor = PCPredictor()

    # Calculate PC volatility
    pc_volatility = {}
    for pc_num in range(1, 6):
        pc_volatility[f'PC{pc_num}'] = pc_returns[f'PC{pc_num}'].std()

    # Get current weekly factor values for prediction
    next_week_date = sample_date + pd.Timedelta(weeks=1)
    current_factors_weekly = all_factors_weekly.loc[all_factors_weekly.index > sample_date]

    if not current_factors_weekly.empty:
        current_factors_weekly = current_factors_weekly.iloc[0:1]
    else:
        current_factors_weekly = pd.DataFrame(columns=all_factors_weekly.columns, index=[next_week_date])

    # Predict PC returns
    predicted_returns, prediction_details = pc_predictor.predict_pc_returns(
        factor_regressor.regression_models, current_factors_weekly, sample_date, pc_volatility
    )

    print(f"PC predictions completed for {sample_date}!")

    # Step 6: Stock Prediction & Validation
    print("\n" + "=" * 60)
    print("STEP 6: STOCK RETURN PREDICTION & VALIDATION")
    print("=" * 60)

    stock_predictor = StockPredictor()

    # Calculate correlation matrix for centrality
    correlation_matrix = lookback_returns.corr()

    # Predict stock returns
    predicted_stock_returns = stock_predictor.predict_stock_returns(
        predicted_returns, loadings, pc_volatility, correlation_matrix, sample_date
    )

    print(f"Stock predictions generated for {len(predicted_stock_returns)} stocks")

    # Get actual next week returns for validation
    next_week_date = sample_date + pd.Timedelta(weeks=1)
    actual_next_returns = stock_weekly_returns.loc[
        next_week_date] if next_week_date in stock_weekly_returns.index else None

    if actual_next_returns is not None:
        # Validate predictions
        r2_score = stock_predictor.validate_stock_predictions(
            sample_date, predicted_stock_returns, actual_next_returns
        )

        # Record training R² for this rebalance
        training_r2_scores = {}
        for pc_num in range(1, 6):
            if (sample_date in factor_regressor.regression_models and
                    pc_num in factor_regressor.regression_models[sample_date]):
                training_r2_scores[pc_num] = factor_regressor.regression_models[sample_date][pc_num].get('r_squared',
                                                                                                         np.nan)

        stock_predictor.record_training_r2(sample_date, training_r2_scores)

        # Create visualizations
        print("\nGenerating visualizations...")

        # 1. Predicted vs Actual scatter plot
        stock_predictor.plot_predicted_vs_actual_scatter(sample_date)

        # 2. Out-of-sample R² over time (will show just this point for now)
        stock_predictor.plot_out_of_sample_r2_over_time()

        # 3. Training R² over time
        stock_predictor.plot_training_r2_over_time(pc_numbers=[1, 2, 3, 4, 5])

    else:
        print(f"No actual returns available for {next_week_date}")

    print(f"\nAll steps completed successfully for {sample_date}!")

    # Save results
    predictions_df = pd.DataFrame.from_dict(predicted_returns, orient='index', columns=['predicted_return'])
    predictions_df['predicted_std_move'] = predictions_df['predicted_return'] / predictions_df.index.map(pc_volatility)
    predictions_df.to_csv(f'pc_predictions_{sample_date.strftime("%Y%m%d")}.csv')

    predicted_stock_returns.to_csv(f'stock_predictions_{sample_date.strftime("%Y%m%d")}.csv')

    return {
        'stock_predictor': stock_predictor,
        'predicted_stock_returns': predicted_stock_returns,
        'actual_next_returns': actual_next_returns,
        'rebalance_dates': rebalance_dates,
        'pca_analyzer': pca_analyzer,
        'factor_regressor': factor_regressor,
        'pc_predictor': pc_predictor,
        'sample_date': sample_date
    }


if __name__ == "__main__":
    results = main()

if __name__ == "__main__":
    results = main()