from statistics import correlation

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd

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

    from sklearn.linear_model import LassoCV, LinearRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    def run_lasso_and_linear_regression(self, pc_series, factor_window, pc_number):
        """
        Predict the NEXT WEEK'S PC CHANGE using lagged factors and Ridge regression.

        Parameters:
        - pc_series: pd.Series of current PC returns (length = lookback_weeks)
        - factor_window: pd.DataFrame of factor returns over the same lookback window
        - pc_number: int, 1 through 5

        Returns:
        - predicted_pc_change: float, predicted change in PC for next week
        - selected_factors: list of factors selected
        - coefficients: regression coefficients
        - r_squared: R² of the model
        """

        # Create lagged factor features (1-week and 2-week lags)
        factor_features = factor_window.copy()

        # Add momentum features (3-week and 6-week moving averages)
        for col in factor_window.columns:
            factor_features[f'{col}_ma3'] = factor_window[col].rolling(3).mean()
            factor_features[f'{col}_ma6'] = factor_window[col].rolling(6).mean()
            factor_features[f'{col}_lag1'] = factor_window[col].shift(1)
            factor_features[f'{col}_lag2'] = factor_window[col].shift(2)

        # Create target: PC changes (week-over-week differences)
        pc_changes = pc_series.diff().dropna()

        # Align data (drop NaN rows from lagged features)
        factor_features = factor_features.dropna()

        # Ensure same length
        min_length = min(len(pc_changes), len(factor_features))
        if min_length < 20:  # Need minimum data for regression
            return 0.0, [], [], 0.0

        X = factor_features.iloc[-min_length:].values
        y = pc_changes.iloc[-min_length:].values

        # Use Ridge regression instead of Lasso for better generalization
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.linear_model import RidgeCV

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time-series CV for Ridge
        n_splits = min(3, len(X_scaled) // 10)
        n_splits = max(n_splits, 2)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Ridge regression with cross-validation
        ridge = RidgeCV(alphas=np.logspace(-3, 1, 20), cv=tscv)
        ridge.fit(X_scaled, y)

        y_pred = ridge.predict(X_scaled)
        r_squared = r2_score(y, y_pred)

        # Select top factors by absolute coefficient value
        feature_importance = np.abs(ridge.coef_)
        top_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
        selected_factors = factor_features.columns[top_indices].tolist()

        # Predict next week's PC change using the most recent factor values
        predicted_pc_change = ridge.predict(X_scaled[-1:].reshape(1, -1))[0]

        print(f"Running Ridge regression for pc{pc_number}...")
        print(f"  Selected top {len(selected_factors)} factors by importance.")
        print(f"  Ridge R²: {r_squared:.4f} | MSE: {mean_squared_error(y, y_pred):.4f}")
        print(f"  Predicted PC change: {predicted_pc_change:.4f}")

        return predicted_pc_change, selected_factors, ridge.coef_, r_squared

    def compute_expected_stock_returns(self, pca_result, predicted_pc_change, pc_name, pc_series):
        """
        Compute expected stock returns using predicted PC CHANGE and loadings.

        Parameters:
        - pca_result: dict, PCA results
        - predicted_pc_change: float, predicted change in PC for next week
        - pc_name: str, name of the PC
        - pc_series: pd.Series, PC returns over the 52-week window

        Returns:
        - r_hat: np.array, expected stock returns for this PC
        - predicted_pc_change: float, the predicted change
        - pc_volatility: float, PC volatility measure
        """

        # Get the PCA loadings (eigenvector) for this PC
        pc_number = int(pc_name.replace("pc", "")) - 1
        correlation_matrix = pca_result["correlation_matrix"]
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        pc_loadings = eigenvectors[:, pc_number]

        # Compute PC volatility (for risk adjustment)
        pc_volatility = pc_series.std()

        # Compute r_hat: loadings × predicted change
        r_hat = pc_loadings * predicted_pc_change

        return r_hat, predicted_pc_change, pc_volatility


    def compute_actual_stock_returns(self):
        """
        Compute actual stock returns for the week following each rebalance date.

        Returns:
        - actual_returns: dict, maps rebalance_date to a np.array of actual stock returns
                         (length = n_stocks) for the week starting at rebalance_date.
        """
        actual_returns = {}
        rebalance_dates = self._get_rebalance_dates()

        for rebalance_date in rebalance_dates:
            # Check if rebalance_date exists in stock_matrix
            if rebalance_date not in self.stock_matrix.index:
                print(f"Warning: No stock returns available for {rebalance_date.strftime('%Y-%m-%d')}. Skipping.")
                continue

            # Get actual returns for the week starting at rebalance_date
            actual_returns[rebalance_date] = self.stock_matrix.loc[rebalance_date].values

        return actual_returns

    def main(self):
        """
        Main execution with improved PC combination and prediction logic.
        """
        # Initialize storage
        self.predicted_pc_changes = {}
        self.selected_factors = {}
        self.regression_coefficients = {}
        self.r_hat_results = {}
        self.r_hat = {}

        # Download and prep data
        self.download_data()
        stock_matrix, factor_matrix = self.calculate_weekly_returns(self.raw_data)
        self.factor_matrix = factor_matrix

        print("\nStock Matrix Preview:")
        print(stock_matrix.head())

        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates()
        print(f"\nTotal rebalance dates: {len(rebalance_dates)}")

        # Process each rebalance date
        for rebalance_date in rebalance_dates:
            print(f"\n=== Processing Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')} ===")

            # Run PCA
            pca_result = self.calculate_pca(rebalance_date, stock_matrix)
            if pca_result is None:
                continue

            pc_df = pca_result["pc_df"]
            explained_variance = pca_result["top_5_explained_variance_ratio"]

            # Get factor window
            factor_window = self.get_factor_window(rebalance_date)
            if factor_window is None:
                continue

            # Store PC predictions and weights
            pc_predictions = {}
            pc_weights = {}

            # Run regression for each PC
            for i, pc_name in enumerate(pc_df.columns):
                pc_number = int(pc_name.replace("pc", ""))
                pc_series = pc_df[pc_name]

                predicted_pc_change, selected_factors, coefficients, r2 = self.run_lasso_and_linear_regression(
                    pc_series, factor_window, pc_number
                )

                # Weight by explained variance and model quality
                model_confidence = max(0.1, r2)  # Minimum 10% weight
                pc_weight = explained_variance[i] * model_confidence

                pc_predictions[pc_name] = predicted_pc_change
                pc_weights[pc_name] = pc_weight

                # Store results
                self.predicted_pc_changes[rebalance_date] = self.predicted_pc_changes.get(rebalance_date, {})
                self.selected_factors[rebalance_date] = self.selected_factors.get(rebalance_date, {})
                self.regression_coefficients[rebalance_date] = self.regression_coefficients.get(rebalance_date, {})

                self.predicted_pc_changes[rebalance_date][pc_name] = predicted_pc_change
                self.selected_factors[rebalance_date][pc_name] = selected_factors
                self.regression_coefficients[rebalance_date][pc_name] = coefficients

                # Compute expected stock returns for this PC
                r_hat, pred_change, pc_vol = self.compute_expected_stock_returns(
                    pca_result, predicted_pc_change, pc_name, pc_series
                )

                # Store individual PC contributions
                self.r_hat_results[rebalance_date] = self.r_hat_results.get(rebalance_date, {})
                self.r_hat_results[rebalance_date][pc_name] = {
                    'r_hat': r_hat,
                    'predicted_change': pred_change,
                    'pc_volatility': pc_vol,
                    'weight': pc_weight
                }

                print(f"\n{pc_name} Results:")
                print(f"  Predicted Change: {pred_change:.4f}")
                print(f"  Model R²: {r2:.4f}")
                print(f"  PC Weight: {pc_weight:.4f}")

            # Combine PCs using weighted average
            total_weight = sum(pc_weights.values())
            if total_weight > 0:
                # Normalize weights
                normalized_weights = {pc: w / total_weight for pc, w in pc_weights.items()}

                # Weighted combination of r_hat
                combined_r_hat = np.zeros(len(stock_matrix.columns))
                for pc_name in pc_df.columns:
                    if pc_name in self.r_hat_results[rebalance_date]:
                        r_hat = self.r_hat_results[rebalance_date][pc_name]['r_hat']
                        weight = normalized_weights[pc_name]
                        combined_r_hat += r_hat * weight

                # Store combined result
                self.r_hat[rebalance_date] = combined_r_hat

                print(f"\nCombined weighted prediction for {rebalance_date.strftime('%Y-%m-%d')}:")
                print(f"  Normalized weights: {normalized_weights}")
                print(f"  Combined r_hat range: [{combined_r_hat.min():.4f}, {combined_r_hat.max():.4f}]")

        print("\n=== All rebalance dates processed successfully ===")

        # Generate improved scatter plot
        import matplotlib.pyplot as plt

        actual_returns = self.compute_actual_stock_returns()
        print("\nComputed actual stock returns for each rebalance date.")

        # Collect predictions and actuals
        predicted_returns = []
        actual_returns_plot = []

        for rebalance_date in rebalance_dates:
            if rebalance_date not in self.r_hat or rebalance_date not in actual_returns:
                continue

            # Use combined weighted prediction
            combined_prediction = self.r_hat[rebalance_date]
            predicted_returns.extend(combined_prediction)
            actual_returns_plot.extend(actual_returns[rebalance_date])

        # Create improved scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(predicted_returns, actual_returns_plot, alpha=0.6, s=30)
        plt.xlabel('Predicted Returns (%)')
        plt.ylabel('Actual Returns (%)')
        plt.title('Improved Predicted vs Actual Stock Returns\n(Weighted PC Combination with Change Prediction)')
        plt.grid(True, alpha=0.3)

        # Add diagonal reference line
        min_val = min(min(predicted_returns), min(actual_returns_plot))
        max_val = max(max(predicted_returns), max(actual_returns_plot))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction (y=x)')

        # Calculate and display correlation
        correlation = np.corrcoef(predicted_returns, actual_returns_plot)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

        plt.legend()
        plt.tight_layout()
        plt.savefig('improved_predicted_vs_actual_returns.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Improved scatter plot saved. Correlation: {correlation:.3f}")
        print("Key improvements: Predicting PC changes, weighted combination, Ridge regression")
if __name__ == "__main__":
    stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
    start_date = '2025-01-01'
    end_date = '2025-08-22'
    strategy = PCA_Manifold_Strategy(lookback_weeks=52, stocks = stocks)
    strategy.main()