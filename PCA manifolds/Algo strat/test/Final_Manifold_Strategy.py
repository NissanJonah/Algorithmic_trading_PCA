from statistics import correlation

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


warnings.filterwarnings('ignore')

class PCA_Manifold_Strategy:
    def __init__(self, lookback_weeks=52, stocks = [], start_date="2022-01-01", end_date="2025-08-22"):
        self.lookback_weeks = lookback_weeks
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.factors = ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']

        # Factor pairs for spread calculations
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

        # get earliest data start (need extra data for lookback)
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        self.earliest_data_start = (start_dt - timedelta(weeks=self.lookback_weeks + 4)).strftime('%Y-%m-%d')
        # Combine all tickers
        self.all_tickers = self._get_all_tickers()

        # Data storage
        self.raw_data = None

        # Data storage
        self.weekly_prices = None
        self.stock_returns = None
        self.factor_returns = None
        self.rotation_returns = None
        self.momentum_returns = None
        self.macro_returns = None
        self.sector_rotation_returns = None
        self.volatility_returns = None
        self.all_factor_returns = None
        self.stock_matrix = None  # <- only stock returns for the whole period
        self.factor_matrix = None  # <- only factor returns for the whole period

    def _get_all_tickers(self):
        """Combine all tickers into a single list."""
        tickers = set()

        # Add stocks and base factors
        tickers.update(self.stocks)
        tickers.update(self.factors)

        # Add all factor pair tickers
        for pair_dict in [self.rotation_pairs, self.momentum_factors, self.macro_factors,
                          self.sector_rotation_factors, self.volatility_factors]:
            for name, (ticker1, ticker2) in pair_dict.items():
                tickers.add(ticker1)
                tickers.add(ticker2)

        return sorted(list(tickers))

    def download_data(self):
        """Download all price data from Yahoo Finance."""
        print(f"Downloading data for {len(self.all_tickers)} tickers...")
        print(f"Date range: {self.earliest_data_start} to {self.end_date}")

        try:
            # Download all data at once
            self.raw_data = yf.download(
                self.all_tickers,
                start=self.earliest_data_start,
                end=self.end_date,
                auto_adjust=True
            )

            # If single ticker, yfinance returns different structure
            if len(self.all_tickers) == 1:
                self.raw_data = pd.DataFrame({self.all_tickers[0]: self.raw_data['Close']})
            else:
                # Use Close prices only
                self.raw_data = self.raw_data['Close']

            print(f"Successfully downloaded data for {len(self.raw_data.columns)} tickers")
            print(f"Data shape: {self.raw_data.shape}")
            print(f"Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")

        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def calculate_weekly_returns(self, raw_data):
        """Convert daily prices to weekly returns and calculate factor spreads."""
        print(f"Converting daily prices to weekly returns...")

        # Resample to weekly prices (Friday close)
        self.weekly_prices = raw_data.resample('W-FRI').last()

        # Calculate weekly returns
        weekly_returns = self.weekly_prices.pct_change().dropna() * 100

        # ---- Separate stocks ----
        stock_cols = [ticker for ticker in self.stocks if ticker in weekly_returns.columns]
        self.stock_matrix = weekly_returns[stock_cols].copy()

        # ---- Separate base factors ----
        factor_cols = [ticker for ticker in self.factors if ticker in weekly_returns.columns]
        self.factor_matrix = weekly_returns[factor_cols].copy()  # Base factors only

        # ---- Calculate spread factors ----
        self._calculate_factor_spreads(weekly_returns)

        # ---- Combine base + spreads into all_factor_returns ----
        self._combine_all_factors()

        print(f"Stock matrix shape: {self.stock_matrix.shape}")
        print(f"Base factor matrix shape: {self.factor_matrix.shape}")
        print(f"All factor returns shape: {self.all_factor_returns.shape}")
        print(f"All Factors: {self.all_factor_returns.columns}")

        return self.stock_matrix, self.factor_matrix

    def _calculate_factor_spreads(self, weekly_returns):
        """Calculate spread returns for all factor pairs."""

        # Rotation factors
        rotation_pairs = {
            "Growth_vs_Value": ("VUG", "VTV"),
            "Large_vs_Small_Cap": ("SPY", "IWM"),
            "Tech_vs_Market": ("XLK", "SPY"),
            "Financials_vs_Market": ("XLF", "SPY"),
            "Banking_vs_Financials": ("KBE", "XLF"),
            "Regional_vs_Banks": ("KRE", "KBE")
        }

        self.rotation_returns = pd.DataFrame(index=weekly_returns.index)
        for name, (ticker1, ticker2) in rotation_pairs.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.rotation_returns[name] = weekly_returns[ticker1] - weekly_returns[ticker2]

        # Momentum factors
        momentum_factors = {
            "High_vs_Low_Beta": ("SPHB", "SPLV"),
            "Momentum_vs_Anti_momentum": ("MTUM", "VMOT"),
            "Quality_vs_Junk": ("QUAL", "SJNK")
        }

        self.momentum_returns = pd.DataFrame(index=weekly_returns.index)
        for name, (ticker1, ticker2) in momentum_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.momentum_returns[name] = weekly_returns[ticker1] - weekly_returns[ticker2]

        # Macro factors
        macro_factors = {
            "Dollar_Strength": ("UUP", "UDN"),
            "Inflation_Expectation": ("SCHP", "VTEB"),
            "Credit_Spread": ("LQD", "HYG"),
            "Yield_Curve": ("SHY", "TLT"),
            "Real_vs_Nominal": ("VTEB", "VGIT")
        }

        self.macro_returns = pd.DataFrame(index=weekly_returns.index)
        for name, (ticker1, ticker2) in macro_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.macro_returns[name] = weekly_returns[ticker1] - weekly_returns[ticker2]

        # Sector rotation factors
        sector_rotation_factors = {
            "Cyclical_vs_Defensive": ("XLI", "XLP"),
            "Risk_on_vs_Risk_off": ("XLY", "XLRE"),
            "Energy_vs_Utilities": ("XLE", "XLU"),
            "Healthcare_vs_Tech": ("XLV", "XLK"),
            "Materials_vs_Staples": ("XLB", "XLP")
        }

        self.sector_rotation_returns = pd.DataFrame(index=weekly_returns.index)
        for name, (ticker1, ticker2) in sector_rotation_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.sector_rotation_returns[name] = weekly_returns[ticker1] - weekly_returns[ticker2]

        # Volatility factors
        volatility_factors = {
            "Vol_Surface": ("VXX", "SVXY"),
            "Equity_vs_Bond_Vol": ("^VIX", "^MOVE")
        }

        self.volatility_returns = pd.DataFrame(index=weekly_returns.index)
        for name, (ticker1, ticker2) in volatility_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.volatility_returns[name] = weekly_returns[ticker1] - weekly_returns[ticker2]

    def _combine_all_factors(self):
        """Combine all factor returns into a single DataFrame."""
        factor_dfs = []

        # Add base factors
        if self.factor_matrix is not None and not self.factor_matrix.empty:
            factor_dfs.append(self.factor_matrix)

        # Add spread factors
        for df in [self.rotation_returns, self.momentum_returns, self.macro_returns,
                   self.sector_rotation_returns, self.volatility_returns]:
            if df is not None and not df.empty:
                factor_dfs.append(df)

        if factor_dfs:
            self.all_factor_returns = pd.concat(factor_dfs, axis=1)
            # Remove any columns with all NaN values
            self.all_factor_returns = self.all_factor_returns.dropna(axis=1, how='all')
            # Forward fill missing values for factors (common in factor data)
            self.all_factor_returns = self.all_factor_returns.fillna(method='ffill')
        else:
            self.all_factor_returns = pd.DataFrame()


# all factor and stock data in weekly returns
    def calculate_pca(self, current_date, stock_matrix):
        """
        Perform PCA on the past 52-week window of stock returns
        and compute PC return time series for each of the top 5 PCs.
        """
        lookback_start = current_date - pd.Timedelta(weeks=self.lookback_weeks)

        # Slice 52-week window, excluding the current date
        window_data = stock_matrix.loc[lookback_start:current_date].iloc[:-1]
        if len(window_data) < self.lookback_weeks:
            return None  # not enough data

        # Standardize stock returns
        standardized_data = (window_data - window_data.mean()) / window_data.std()

        # Correlation matrix
        correlation_matrix = standardized_data.corr()

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        idx = np.argsort(eigenvalues)[::-1]  # sort descending
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract top 5 eigenvectors
        top_5_vectors = eigenvectors[:, :5]

        # ---- Compute PC return series ----
        # PC_t = R_t * w for each week t
        pc_return_series = {}
        for i in range(5):
            weights = top_5_vectors[:, i]  # eigenvector
            pc_return_series[f'pc{i + 1}'] = window_data.values @ weights  # shape (52,)

        # Convert to DataFrame with dates as index
        pc_df = pd.DataFrame(pc_return_series, index=window_data.index)

        return {
            "date": current_date,
            "correlation_matrix": correlation_matrix,
            "top_5_eigenvalues": eigenvalues[:5],
            "top_5_explained_variance_ratio": eigenvalues[:5] / eigenvalues.sum(),
            "pc_df": pc_df  # <-- each PC now has 52 weekly values
        }

    def _compute_pca_from_corr_matrix(self, corr_matrix):
        """
        Private helper to compute PCA given a correlation matrix.
        Returns:
            - eigenvalues (sorted)
            - eigenvectors (sorted to match eigenvalues)
            - explained_variance_ratio
        """
        import numpy as np

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

        # Sort by largest eigenvalues
        sorted_idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Explained variance ratio
        explained_variance_ratio = eigenvalues / eigenvalues.sum()

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "explained_variance_ratio": explained_variance_ratio
        }

    def _get_rebalance_dates(self):
        """Generate a list of all weekly rebalance dates (Fridays) between start_date and end_date."""
        weekly_fridays = self.weekly_prices.index  # already resampled to W-FRI in calculate_weekly_returns
        # Filter to only dates within start and end
        rebalance_dates = [date for date in weekly_fridays if date >= pd.to_datetime(self.start_date) and date <= pd.to_datetime(self.end_date)]
        return rebalance_dates

    def get_factor_window(self, rebalance_date):
        """
        Extracts a 52-week rolling window of factor returns ending at the given rebalance_date.
        """
        if self.all_factor_returns is None:
            raise ValueError("All factor returns have not been initialized. Load factor data first.")

        # Ensure rebalance_date exists in factor data
        if rebalance_date not in self.all_factor_returns.index:
            print(f"Warning: {rebalance_date} not in factor data index. Skipping.")
            return None

        # Find index of current rebalance date
        current_idx = self.all_factor_returns.index.get_loc(rebalance_date)

        # Check that there's enough historical data
        if current_idx < self.lookback_weeks:
            print(f"Not enough factor history for {rebalance_date}. Need {self.lookback_weeks} weeks.")
            return None

        # Slice past 52 weeks (excludes current week)
        factor_window = self.all_factor_returns.iloc[current_idx - self.lookback_weeks: current_idx]

        return factor_window

    def forecast_factor_returns(self, factor_window):
        """
        Forecast the next week's factor returns using the most recent week's returns (lagged approach).

        Parameters:
        - factor_window: pd.DataFrame of factor returns over the 52-week lookback window

        Returns:
        - forecasted_factors: np.array of forecasted factor returns for the next week
        """
        # Use the last week's factor returns as the forecast for the next week
        forecasted_factors = factor_window.iloc[-1].values
        return forecasted_factors

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso

    def run_lasso_and_linear_regression(self, pc_series, factor_window, pc_number):
        """
        Predict the next week's PC returns using Lasso for feature selection and Linear Regression.

        Parameters:
        - pc_series: pd.Series of current PC returns (length = lookback_weeks)
        - factor_window: pd.DataFrame of factor returns over the same lookback window
        - pc_number: int, 1 through 5

        Returns:
        - predicted_pc: np.array, predicted standardized PC returns for the 52-week window
        - predicted_pc_next: float, predicted PC return for the next week (out-of-sample)
        - selected_factors: list of factors selected by Lasso
        - coefficients: regression coefficients
        - r_squared: R² of OLS fit
        - scaler_X: StandardScaler for factors
        - scaler_y: StandardScaler for PC returns
        """
        # Align PC returns with factor dates
        merged_data = factor_window.copy()
        merged_data['PC'] = pc_series.values

        # Standardize factors and PC returns for regression
        X = merged_data.drop(columns='PC').values
        y = merged_data['PC'].values.reshape(-1, 1)

        factor_scaler = StandardScaler()
        pc_scaler = StandardScaler()

        X_scaled = factor_scaler.fit_transform(X)
        y_scaled = pc_scaler.fit_transform(y).ravel()

        # Time-series CV for Lasso
        n_splits = min(3, len(X_scaled) // 15)
        n_splits = max(n_splits, 2)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Lasso with GridSearchCV for feature selection
        param_grid = {'alpha': np.logspace(-4, -1, 20)}
        lasso = GridSearchCV(
            estimator=Lasso(fit_intercept=True, max_iter=5000, tol=1e-5),
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        lasso.fit(X_scaled, y_scaled)

        # Get selected features
        selected_mask = lasso.best_estimator_.coef_ != 0
        selected_factors = merged_data.drop(columns='PC').columns[selected_mask].tolist()
        selected_indices = np.where(selected_mask)[0]

        # If no factors selected, take top 5 by correlation
        if len(selected_indices) == 0:
            corrs = np.abs(np.corrcoef(X_scaled.T, y_scaled)[-1, :-1])
            top_indices = np.argsort(corrs)[-5:]
            selected_indices = top_indices
            selected_factors = merged_data.drop(columns='PC').columns[selected_indices].tolist()

        # Linear regression on selected factors
        X_selected = X_scaled[:, selected_indices]
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_selected, y_scaled)
        y_pred = ols.predict(X_selected)
        r_squared = r2_score(y_scaled, y_pred)

        coefficients = ols.coef_

        # Forecast next week's factor returns
        forecasted_factors = self.forecast_factor_returns(factor_window)
        forecasted_factors_scaled = factor_scaler.transform(forecasted_factors.reshape(1, -1))[:, selected_indices]

        # Predict next week's PC return (out-of-sample)
        predicted_pc_next_scaled = ols.predict(forecasted_factors_scaled)[0]
        predicted_pc_next = pc_scaler.inverse_transform([[predicted_pc_next_scaled]])[0][0]

        print(f"Running regression for pc{pc_number}...")
        print(f"  Lasso selected {len(selected_factors)} factors: {selected_factors}")
        print(f"  OLS R²: {r_squared:.4f} | MSE: {mean_squared_error(y_scaled, y_pred):.4f}")
        print(f"  pc{pc_number} Coefficients: {np.round(coefficients, 4)}")
        print(f"  Predicted PC{pc_number} for next week: {predicted_pc_next:.4f}")

        return y_pred, predicted_pc_next, selected_factors, coefficients, r_squared, factor_scaler, pc_scaler

    def main(self):
        """
        Main execution flow:
          1. Download & prep stock + factor data
          2. Get all weekly rebalance dates
          3. Loop through each rebalance date:
             - Run PCA on past 52-week stock returns
             - Slice factor window for same 52-week period
             - Run Lasso + Linear Regression for top 5 PCs
             - Compute expected stock returns (r_hat)
          4. Compute actual stock returns and generate scatter plot
        """
        import numpy as np

        # Initialize storage for regression and r_hat results
        self.predicted_pc_returns = {}
        self.selected_factors = {}
        self.regression_coefficients = {}
        self.r_hat_results = {}
        self.r_hat = {}
        self.pca_results = {}
        self.r_squared = {}
        self.total_r_hat = {}
        self.scalers = {}  # Store scalers for each PC and rebalance date

        # ---------- STEP 1: Download and prep data ----------
        self.download_data()
        stock_matrix, factor_matrix = self.calculate_weekly_returns(self.raw_data)

        # Store factor matrix in the object for later use
        self.factor_matrix = factor_matrix

        print("\nStock Matrix Preview:")
        print(stock_matrix.head())

        # ---------- STEP 2: Get list of rebalance dates ----------
        rebalance_dates = self._get_rebalance_dates()
        print(f"\nTotal rebalance dates: {len(rebalance_dates)}")
        print(f"Rebalance dates preview: {rebalance_dates[:5]}")

        # ---------- STEP 3: Loop through each rebalance date ----------
        for rebalance_date in rebalance_dates:
            print(f"\n=== Processing Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')} ===")

            # ---- 3A. Run PCA on stock returns ----
            pca_result = self.calculate_pca(rebalance_date, stock_matrix)
            if pca_result is None:
                print(f"Skipping {rebalance_date.strftime('%Y-%m-%d')} due to insufficient stock data.")
                continue
            self.pca_results[rebalance_date] = pca_result

            pc_df = pca_result["pc_df"]  # DataFrame with columns pc1, pc2, ..., pc5

            # ---- 3B. Get factor window for this date ----
            factor_window = self.get_factor_window(rebalance_date)
            if factor_window is None:
                print(f"Skipping {rebalance_date.strftime('%Y-%m-%d')} due to insufficient factor data.")
                continue
            print(f"Factor window shape: {factor_window.shape}")

            # ---- 3C. Run Lasso + Linear Regression for each PC ----
            self.r_squared[rebalance_date] = {}
            self.scalers[rebalance_date] = {}
            for pc_name in pc_df.columns:
                pc_number = int(pc_name.replace("pc", ""))  # Extract 1–5
                pc_series = pc_df[pc_name]

                # Updated call to handle seven return values
                predicted_pc, predicted_pc_next, selected_factors, coefficients, r2, scaler_X, scaler_y = self.run_lasso_and_linear_regression(
                    pc_series, factor_window, pc_number
                )

                # Store results
                self.predicted_pc_returns[rebalance_date] = self.predicted_pc_returns.get(rebalance_date, {})
                self.selected_factors[rebalance_date] = self.selected_factors.get(rebalance_date, {})
                self.regression_coefficients[rebalance_date] = self.regression_coefficients.get(rebalance_date, {})
                self.scalers[rebalance_date][pc_name] = {'scaler_X': scaler_X, 'scaler_y': scaler_y}

                self.predicted_pc_returns[rebalance_date][pc_name] = predicted_pc
                self.selected_factors[rebalance_date][pc_name] = selected_factors
                self.regression_coefficients[rebalance_date][pc_name] = coefficients
                self.r_squared[rebalance_date][pc_name] = r2

                print(f"\n{pc_name} Regression Results:")
                print(f"  Selected Factors: {selected_factors}")
                print(f"  Coefficients: {np.round(coefficients, 4)}")
                print(f"  R²: {r2:.4f}")

if __name__ == "__main__":
    stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
    start_date = '2025-01-01'
    end_date = '2025-08-22'
    strategy = PCA_Manifold_Strategy(lookback_weeks=52, stocks = stocks)
    strategy.main()