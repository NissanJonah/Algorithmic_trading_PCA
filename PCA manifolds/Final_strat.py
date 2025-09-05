import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score



class PCAFactorStrategy:
    def __init__(self, start_date, end_date, lookback_days=252):
        """
        Initialize the PCAFactorStrategy with date range and stock/factor definitions.

        Args:
            start_date (str): Start date for strategy (e.g., '2020-01-01')
            end_date (str): End date for strategy (e.g., '2025-08-22')
            lookback_days (int): Number of trading days for lookback (default: 252, ~1 year)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days

        # Define stocks and factors
        self.stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']
        self.factors = ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']

        # Define factor pairs for spreads
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

        # Combine all tickers for download
        self.all_tickers = list(set(
            self.stocks +
            self.factors +
            [pair[0] for pair in self.rotation_pairs.values()] +
            [pair[1] for pair in self.rotation_pairs.values()] +
            [pair[0] for pair in self.momentum_factors.values()] +
            [pair[1] for pair in self.momentum_factors.values()] +
            [pair[0] for pair in self.macro_factors.values()] +
            [pair[1] for pair in self.macro_factors.values()] +
            [pair[0] for pair in self.sector_rotation_factors.values()] +
            [pair[1] for pair in self.sector_rotation_factors.values()] +
            [pair[0] for pair in self.volatility_factors.values()] +
            [pair[1] for pair in self.volatility_factors.values()]
        ))

        # Calculate earliest data start (1 year before start_date)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        self.earliest_data_start = (start_dt - timedelta(days=365)).strftime('%Y-%m-%d')

        # Initialize data containers
        self.raw_data = None
        self.stock_returns = None
        self.factor_returns = None
        self.factor_spreads = None
        self.rebalance_dates = None
        self.all_factors_weekly_returns_matrix = None

    def download_data(self):
        """
        Download daily closing prices for all tickers using yfinance.
        """
        try:
            # Download all data at once
            self.raw_data = yf.download(
                self.all_tickers,
                start=self.earliest_data_start,
                end=self.end_date,
                auto_adjust=True
            )['Close']

            # Validate data
            if self.raw_data.empty:
                raise ValueError("No data downloaded from yfinance.")

            # Remove any tickers with all NaN values
            self.raw_data = self.raw_data.dropna(axis=1, how='all')

            # Check if all required tickers are present
            missing_tickers = [ticker for ticker in self.all_tickers if ticker not in self.raw_data.columns]
            if missing_tickers:
                print(f"Warning: Missing data for tickers: {missing_tickers}")

            # Forward-fill any missing values, then back-fill
            self.raw_data = self.raw_data.ffill().bfill()

        except Exception as e:
            print(f"Error downloading data: {e}")
            self.raw_data = None

    def compute_factor_spreads(self):
        """
        Compute weekly return spreads for factor pairs and base factor returns (Friday close).
        Returns are in percentage terms.
        """
        if self.raw_data is None:
            raise ValueError("Raw data not available. Run download_data() first.")

        # Filter to Fridays for weekly data
        weekly_data = self.raw_data[self.raw_data.index.weekday == 4]  # 4 = Friday

        # Calculate weekly returns: (P_t - P_{t-1}) / P_{t-1} * 100
        weekly_returns = weekly_data.pct_change() * 100
        weekly_returns = weekly_returns.dropna()

        # Initialize factor spreads DataFrame
        self.factor_spreads = pd.DataFrame(index=weekly_returns.index)

        # Compute spreads for each factor pair category
        for name, (ticker1, ticker2) in self.rotation_pairs.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.factor_spreads[name] = weekly_returns[ticker1] - weekly_returns[ticker2]
            else:
                print(f"Warning: Missing data for {name} ({ticker1}, {ticker2})")

        for name, (ticker1, ticker2) in self.momentum_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.factor_spreads[name] = weekly_returns[ticker1] - weekly_returns[ticker2]
            else:
                print(f"Warning: Missing data for {name} ({ticker1}, {ticker2})")

        for name, (ticker1, ticker2) in self.macro_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.factor_spreads[name] = weekly_returns[ticker1] - weekly_returns[ticker2]
            else:
                print(f"Warning: Missing data for {name} ({ticker1}, {ticker2})")

        for name, (ticker1, ticker2) in self.sector_rotation_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.factor_spreads[name] = weekly_returns[ticker1] - weekly_returns[ticker2]
            else:
                print(f"Warning: Missing data for {name} ({ticker1}, {ticker2})")

        for name, (ticker1, ticker2) in self.volatility_factors.items():
            if ticker1 in weekly_returns.columns and ticker2 in weekly_returns.columns:
                self.factor_spreads[name] = weekly_returns[ticker1] - weekly_returns[ticker2]
            else:
                print(f"Warning: Missing data for {name} ({ticker1}, {ticker2})")

        # Store stock returns (daily) and base factor returns (weekly)
        self.stock_returns = self.raw_data[self.stocks].pct_change() * 100
        self.stock_returns = self.stock_returns.dropna()
        self.factor_returns = weekly_returns[self.factors].dropna()

        # Combine base factor returns and factor spreads into all_factors_weekly_returns_matrix
        self.all_factors_weekly_returns_matrix = pd.concat(
            [self.factor_returns, self.factor_spreads],
            axis=1
        ).dropna()

        # Generate rebalance dates (Fridays within the date range)
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='W-FRI')
        self.rebalance_dates = [date for date in date_range if date in weekly_returns.index]

        # Ensure sufficient lookback for each rebalance date
        min_date = pd.to_datetime(self.earliest_data_start) + timedelta(days=self.lookback_days)
        self.rebalance_dates = [date for date in self.rebalance_dates if date >= min_date]

    def compute_pca(self, date):
        """
        Compute PCA for stock returns up to the given rebalance date, flipping PC signs if sum of loadings is negative.

        Args:
            date (pd.Timestamp): Rebalance date (Friday) for PCA computation.

        Returns:
            dict: PCA results containing loadings, scores, explained variance, and stock std.
        """
        print(f"Creating PCA for {date}")

        # Extract 252 days of stock returns up to the day before rebalance date
        try:
            end_idx = self.stock_returns.index.get_loc(date) - 1
            start_idx = end_idx - self.lookback_days + 1
            if start_idx < 0:
                raise ValueError(f"Insufficient lookback data for {date}")
            stock_returns_window = self.stock_returns.iloc[start_idx:end_idx + 1]

            # Validate window size
            if len(stock_returns_window) != self.lookback_days:
                raise ValueError(
                    f"Stock returns window for {date} has {len(stock_returns_window)} days, expected {self.lookback_days}")

            # Check for NaN or infinite values
            if stock_returns_window.isna().any().any() or np.isinf(stock_returns_window).any().any():
                print(f"Warning: NaN or Inf values in stock returns window for {date}")
                stock_returns_window = stock_returns_window.fillna(0).replace([np.inf, -np.inf], 0)

            # Standardize returns: (R - mean) / std
            stock_mean = stock_returns_window.mean()
            stock_std = stock_returns_window.std()
            stock_std = stock_std.where(stock_std > 1e-10, 1e-10)
            stock_returns_std = (stock_returns_window - stock_mean) / stock_std

            # Check for NaN in standardized returns
            if stock_returns_std.isna().any().any() or np.isinf(stock_returns_std).any().any():
                print(f"Warning: NaN or Inf in standardized returns for {date}")
                stock_returns_std = stock_returns_std.fillna(0).replace([np.inf, -np.inf], 0)

            # Compute covariance matrix
            cov_matrix = stock_returns_std.T @ stock_returns_std / (self.lookback_days - 1)
            cov_matrix += np.eye(len(self.stocks)) * 1e-6  # Regularize for stability

            # Check for NaN in covariance matrix
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                print(f"Warning: NaN or Inf in covariance matrix for {date}")
                cov_matrix = np.nan_to_num(cov_matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

            # Eigenvalue decomposition for top 5 PCs
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx][:5]
            eigenvectors = eigenvectors[:, idx][:, :5]  # Shape: (N_stocks, 5)

            # Flip signs of eigenvectors if sum of loadings is negative
            for i in range(eigenvectors.shape[1]):
                if np.sum(eigenvectors[:, i]) < 0:
                    eigenvectors[:, i] = -eigenvectors[:, i]
                    print(
                        f"Flipped signs for PC{i + 1} due to negative sum of loadings: {np.sum(eigenvectors[:, i]):.4f}")

            # Ensure eigenvectors are valid
            if np.any(np.isnan(eigenvectors)) or np.any(np.isinf(eigenvectors)):
                print(f"Warning: NaN or Inf in eigenvectors for {date}")
                eigenvectors = np.nan_to_num(eigenvectors, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

            # Avoid zero eigenvalues for explained variance ratio
            eigenvalues = np.where(eigenvalues < 1e-10, 1e-10, eigenvalues)
            explained_variance_ratio = eigenvalues / eigenvalues.sum()

            # Compute PC scores: R_std @ V
            stock_returns_std_np = stock_returns_std.values  # Shape: (N_days, N_stocks)
            print(f"Stock returns std shape: {stock_returns_std_np.shape}, Eigenvectors shape: {eigenvectors.shape}")
            print(f"Sample stock_returns_std: {stock_returns_std_np[:2, :2]}")
            print(f"Sample eigenvectors: {eigenvectors[:2, :2]}")

            pc_scores_np = stock_returns_std_np @ eigenvectors  # Shape: (N_days, 5)
            pc_scores = pd.DataFrame(
                pc_scores_np,
                index=stock_returns_std.index,
                columns=[f"PC{i + 1}" for i in range(5)]
            )

            # Check for NaN in PC scores
            if pc_scores.isna().any().any() or np.isinf(pc_scores).any().any():
                print(f"Warning: NaN or Inf in PC scores for {date}")
                print(f"Sample PC scores: {pc_scores.iloc[:2, :2]}")
                pc_scores = pc_scores.fillna(0).replace([np.inf, -np.inf], 0)

            # Store results
            results = {
                'loadings': eigenvectors,  # (N_stocks, 5)
                'pc_scores': pc_scores,  # (N_days, 5)
                'explained_variance_ratio': explained_variance_ratio,  # (5,)
                'stock_std': stock_std.values,  # (N_stocks,)
                'stock_mean': stock_mean.values  # (N_stocks,)
            }

            return results

        except Exception as e:
            print(f"Error in PCA computation for {date}: {e}")
            return None

    def compute_regression(self, date, pca_results):
        """
        Perform regression modeling for each PC to predict next week's returns using the entire lookback period as training data.

        Args:
            date (pd.Timestamp): Rebalance date (Friday).
            pca_results (dict): PCA results containing pc_scores, loadings, etc.

        Returns:
            dict: Regression results containing coefficients, train R², and predicted PC returns.
        """
        print(f"\nComputing regression for {date}")
        regression_results = {}

        # Extract 52 weeks of factor returns up to the rebalance date
        try:
            end_idx = self.all_factors_weekly_returns_matrix.index.get_loc(date)
            start_idx = max(0, end_idx - 52 + 1)
            factor_returns_window = self.all_factors_weekly_returns_matrix.iloc[start_idx:end_idx + 1]

            if len(factor_returns_window) < 52:
                print(f"Warning: Insufficient factor data for {date}, got {len(factor_returns_window)} weeks")
                return None

            # Check for NaN or infinite values in factor returns
            if factor_returns_window.isna().any().any() or np.isinf(factor_returns_window).any().any():
                print(f"Warning: NaN or Inf in factor returns for {date}")
                factor_returns_window = factor_returns_window.fillna(0).replace([np.inf, -np.inf], 0)

            # Align PC scores with factor returns dates
            weekly_dates = factor_returns_window.index
            pc_scores = pca_results['pc_scores']
            common_dates = pc_scores.index.intersection(weekly_dates)
            pc_scores_weekly = pc_scores.loc[common_dates]
            factor_returns_window = factor_returns_window.loc[common_dates]

            print(f"Aligned dates count: {len(common_dates)}")
            print(f"Factor returns window shape: {factor_returns_window.shape}")
            print(f"PC scores weekly shape: {pc_scores_weekly.shape}")
            print(f"Sample aligned dates: {common_dates[:5].tolist()}")
            print(f"Missing dates (if any): {[d for d in weekly_dates if d not in common_dates]}")

            if len(pc_scores_weekly) < 10:  # Ensure enough data
                print(f"Warning: Too few aligned weeks ({len(pc_scores_weekly)}) for {date}")
                return None

            if len(pc_scores_weekly) != len(factor_returns_window):
                print(
                    f"Warning: Mismatch in weeks, factor returns: {len(factor_returns_window)}, PC scores: {len(pc_scores_weekly)}")
                return None

            # Standardize factor returns
            factor_mean = factor_returns_window.mean()
            factor_std = factor_returns_window.std()
            factor_std = factor_std.where(factor_std > 1e-10, 1e-10)
            X = (factor_returns_window - factor_mean) / factor_std
            X = X.fillna(0).replace([np.inf, -np.inf], 0)

            # Get next week's factor returns (if available)
            next_idx = end_idx + 1
            X_next = None
            if next_idx < len(self.all_factors_weekly_returns_matrix):
                X_next = self.all_factors_weekly_returns_matrix.iloc[[next_idx]]
                X_next = (X_next - factor_mean) / factor_std
                X_next = X_next.fillna(0).replace([np.inf, -np.inf], 0)
                print(f"Next week's factor returns available for {date}")

            # Loop through each PC
            for pc in pc_scores_weekly.columns:
                print(f"\nProcessing {pc} for {date}")
                y = pc_scores_weekly[pc].values
                pc_std = y.std() if y.std() > 1e-10 else 1e-10
                print(f"{pc} standard deviation: {pc_std:.4f}")

                # Lasso regression for factor selection
                lasso = Lasso(alpha=0.1, max_iter=10000, tol=1e-4)
                lasso.fit(X, y)
                # Limit to top 5 factors by coefficient magnitude
                coef = np.abs(lasso.coef_)
                selected_indices = np.argsort(coef)[-5:] if np.sum(coef != 0) > 5 else np.where(coef != 0)[0]
                selected_factors = X.columns[selected_indices].tolist()

                print(f"Selected factors for {pc}: {selected_factors}")

                if not selected_factors:
                    print(f"Warning: No factors selected for {pc}")
                    regression_results[pc] = {
                        'selected_factors': [],
                        'coefficients': [],
                        'train_r2': 0.0,
                        'predicted_return': 0.0
                    }
                    continue

                # Linear regression on selected factors
                X_selected = X[selected_factors]
                lin_reg = LinearRegression()
                lin_reg.fit(X_selected, y)

                # Compute train R²
                train_r2 = r2_score(y, lin_reg.predict(X_selected))

                # Predict next week's PC return
                predicted_return = 0.0
                if X_next is not None and selected_factors:
                    X_next_selected = X_next[selected_factors]
                    predicted_return = lin_reg.predict(X_next_selected)[0] * pc_std

                print(f"{pc} - Train R² (between regression line and {pc}): {train_r2:.4f}")
                print(f"{pc} - Predicted return: {predicted_return:.4f}")

                regression_results[pc] = {
                    'selected_factors': selected_factors,
                    'coefficients': lin_reg.coef_.tolist(),
                    'train_r2': train_r2,
                    'predicted_return': predicted_return
                }

            return regression_results

        except Exception as e:
            print(f"Error in regression computation for {date}: {e}")
            return None

def main():
    # Initialize strategy
    strategy = PCAFactorStrategy(
        start_date='2020-01-01',
        end_date='2025-08-22',
        lookback_days=252
    )

    # Download data
    strategy.download_data()
    if strategy.raw_data is None:
        print("Failed to download raw data.")
        return

    # Compute factor spreads and weekly returns
    strategy.compute_factor_spreads()

    # Validate initial outputs
    if strategy.stock_returns is not None:
        print(f"Stock returns shape: {strategy.stock_returns.shape}")
        print("Stock returns sample (first 5 rows):")
        print(strategy.stock_returns.head())

    if strategy.factor_returns is not None:
        print(f"Factor returns shape: {strategy.factor_returns.shape}")
        print("Factor returns sample (first 5 rows):")
        print(strategy.factor_returns.head())

    if strategy.factor_spreads is not None:
        print(f"Factor spreads shape: {strategy.factor_spreads.shape}")
        print("Factor spreads sample (first 5 rows):")
        print(strategy.factor_spreads.head())

    if strategy.all_factors_weekly_returns_matrix is not None:
        print(f"All factors weekly returns matrix shape: {strategy.all_factors_weekly_returns_matrix.shape}")
        print("All factors weekly returns matrix sample (first 5 rows):")
        print(strategy.all_factors_weekly_returns_matrix.head())

    if strategy.rebalance_dates:
        print(f"Number of rebalance dates: {len(strategy.rebalance_dates)}")
        print(f"First few rebalance dates: {strategy.rebalance_dates[:5]}")

        # Initialize storage for all results
        all_results = []

        # Loop through rebalance dates
        for rebalance_date in strategy.rebalance_dates:
            print(f"\n--- Processing rebalance date: {rebalance_date} ---")

            # Compute PCA
            try:
                pca_results = strategy.compute_pca(rebalance_date)
                if pca_results is None:
                    print(f"Skipping regression for {rebalance_date} due to PCA failure")
                    continue

                # Extract PCA results
                loadings = pd.DataFrame(
                    pca_results['loadings'],
                    index=strategy.stocks,
                    columns=[f"PC{i + 1}" for i in range(5)]
                )
                pc_scores = pd.DataFrame(
                    pca_results['pc_scores'],
                    index=strategy.stock_returns.iloc[
                        strategy.stock_returns.index.get_loc(rebalance_date) - strategy.lookback_days + 1:
                        strategy.stock_returns.index.get_loc(rebalance_date)
                    ].index,
                    columns=[f"PC{i + 1}" for i in range(5)]
                )
                explained_variance_ratio = pd.Series(
                    pca_results['explained_variance_ratio'],
                    index=[f"PC{i + 1}" for i in range(5)]
                )
                stock_std = pd.Series(
                    pca_results['stock_std'],
                    index=strategy.stocks
                )
                stock_mean = pd.Series(
                    pca_results['stock_mean'],
                    index=strategy.stocks
                )

                # Print scalar variables
                print(f"Rebalance date: {rebalance_date}")
                print(f"Number of stocks: {len(strategy.stocks)}")
                print(f"Lookback days: {strategy.lookback_days}")

                # Print matrices (first and last two rows/columns)
                end_idx = strategy.stock_returns.index.get_loc(rebalance_date) - 1
                start_idx = end_idx - strategy.lookback_days + 1
                stock_returns_window = strategy.stock_returns.iloc[start_idx:end_idx + 1]

                print("\nStock returns window (first and last two rows/columns):")
                stock_returns_subset = stock_returns_window.iloc[[0, 1, -2, -1], [0, 1, -2, -1]]
                print(stock_returns_subset)

                print("\nPCA Loadings (first and last two rows/columns):")
                loadings_subset = loadings.iloc[[0, 1, -2, -1], [0, 1, -2, -1]]
                print(loadings_subset)

                print("\nPC Scores (first and last two rows/columns):")
                pc_scores_subset = pc_scores.iloc[[0, 1, -2, -1], [0, 1, -2, -1]]
                print(pc_scores_subset)

                print("\nExplained Variance Ratio (all PCs):")
                print(explained_variance_ratio)

                print("\nStock Standard Deviations (first and last two):")
                stock_std_subset = stock_std.iloc[[0, 1, -2, -1]]
                print(stock_std_subset)

                print("\nStock Means (first and last two):")
                stock_mean_subset = stock_mean.iloc[[0, 1, -2, -1]]
                print(stock_mean_subset)

                # Compute regression for each PC
                regression_results = strategy.compute_regression(rebalance_date, pca_results)
                if regression_results is None:
                    print(f"Skipping further processing for {rebalance_date} due to regression failure")
                    continue

                # Store results
                rebalance_result = {
                    'date': rebalance_date,
                    'pca_results': pca_results,
                    'regression_results': regression_results
                }
                all_results.append(rebalance_result)

                # Print regression summary
                print("\nRegression Summary:")
                for pc in regression_results:
                    print(f"{pc}:")
                    print(f"  Selected factors: {regression_results[pc]['selected_factors']}")
                    print(f"  Coefficients: {regression_results[pc]['coefficients']}")
                    print(
                        f"  Train R² (between regression line and {pc}): {regression_results[pc]['train_r2']:.4f}")
                    print(f"  Predicted return: {regression_results[pc]['predicted_return']:.4f}")

            except Exception as e:
                print(f"Error processing for {rebalance_date}: {e}")

        # Save results to JSON
        import json
        with open('rebalance_results.json', 'w') as f:
            json.dump([{
                'date': str(r['date']),
                'pca_results': {
                    k: v.to_dict('records') if isinstance(v, pd.DataFrame) else
                    v.tolist() if isinstance(v, (np.ndarray, pd.Series)) else v
                    for k, v in r['pca_results'].items()
                },
                'regression_results': r['regression_results']
            } for r in all_results], f, indent=4)
        print("\nResults saved to rebalance_results.json")
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()