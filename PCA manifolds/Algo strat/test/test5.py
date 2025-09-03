import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class DataManager:
    def __init__(self):
        # Stock universe - financial sector
        self.stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET', 'PRU']

        # Base factors
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

        # Date parameters
        self.start_date = '2020-01-01'
        self.end_date = '2021-01-01'
        self.lookback_weeks = 52

        # Calculate earliest data start (need extra data for lookback)
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        self.earliest_data_start = (start_dt - timedelta(weeks=self.lookback_weeks + 4)).strftime('%Y-%m-%d')

        # Combine all tickers
        self.all_tickers = self._get_all_tickers()

        # Data storage
        self.raw_data = None

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

    def validate_data_integrity(self):
        """Validate the downloaded data for completeness and quality."""
        print("\n=== DATA VALIDATION RESULTS ===")

        # Check if data was downloaded
        if self.raw_data is None:
            raise ValueError("No data downloaded. Call download_data() first.")

        validation_results = {
            'total_tickers_requested': len(self.all_tickers),
            'total_tickers_downloaded': len(self.raw_data.columns),
            'missing_tickers': [],
            'data_start_date': self.raw_data.index[0],
            'data_end_date': self.raw_data.index[-1],
            'total_trading_days': len(self.raw_data),
            'missing_data_summary': {},
            'extreme_values_detected': {},
            'validation_passed': True,
            'issues': []
        }

        # Check for missing tickers
        downloaded_tickers = set(self.raw_data.columns)
        requested_tickers = set(self.all_tickers)
        missing_tickers = requested_tickers - downloaded_tickers

        if missing_tickers:
            validation_results['missing_tickers'] = list(missing_tickers)
            validation_results['issues'].append(f"Missing {len(missing_tickers)} tickers")
            print(f"❌ Missing tickers: {missing_tickers}")
        else:
            print("✅ All requested tickers downloaded successfully")

        # Check date range
        expected_start = datetime.strptime(self.earliest_data_start, '%Y-%m-%d').date()
        actual_start = self.raw_data.index[0].date()

        if actual_start > expected_start + timedelta(days=7):  # Allow some flexibility
            validation_results['issues'].append("Start date significantly later than expected")
            print(f"⚠️  Start date: Expected ~{expected_start}, Got {actual_start}")
        else:
            print(f"✅ Start date acceptable: {actual_start}")

        # Check for missing data
        print(f"\n📊 Missing Data Analysis:")
        for ticker in self.raw_data.columns:
            missing_count = self.raw_data[ticker].isnull().sum()
            missing_pct = (missing_count / len(self.raw_data)) * 100

            validation_results['missing_data_summary'][ticker] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }

            if missing_pct > 10:  # More than 10% missing
                validation_results['issues'].append(f"{ticker}: {missing_pct:.1f}% missing data")
                print(f"❌ {ticker}: {missing_count} missing ({missing_pct:.1f}%)")
            elif missing_pct > 5:  # 5-10% missing
                print(f"⚠️  {ticker}: {missing_count} missing ({missing_pct:.1f}%)")
            elif missing_count > 0:
                print(f"✅ {ticker}: {missing_count} missing ({missing_pct:.1f}%)")

        # Check for extreme values (potential data errors)
        print(f"\n📈 Extreme Values Analysis:")
        for ticker in self.raw_data.columns:
            if ticker in self.raw_data.columns:
                prices = self.raw_data[ticker].dropna()
                if len(prices) > 0:
                    # Calculate daily returns
                    returns = prices.pct_change().dropna()

                    # Check for extreme daily returns (>50% or <-50%)
                    extreme_positive = (returns > 0.5).sum()
                    extreme_negative = (returns < -0.5).sum()

                    if extreme_positive > 0 or extreme_negative > 0:
                        validation_results['extreme_values_detected'][ticker] = {
                            'extreme_positive': extreme_positive,
                            'extreme_negative': extreme_negative
                        }
                        print(
                            f"⚠️  {ticker}: {extreme_positive} extreme positive, {extreme_negative} extreme negative returns")

        # Check data coverage for critical periods
        print(f"\n📅 Data Coverage Analysis:")
        strategy_start = datetime.strptime(self.start_date, '%Y-%m-%d')
        strategy_data = self.raw_data[self.raw_data.index >= strategy_start]

        if len(strategy_data) < 200:  # Less than ~8 months of trading days
            validation_results['issues'].append("Insufficient data for strategy period")
            print(f"❌ Strategy period has only {len(strategy_data)} trading days")
        else:
            print(f"✅ Strategy period has {len(strategy_data)} trading days")

        # Stock-specific validation
        print(f"\n🏦 Stock Universe Validation:")
        missing_stocks = set(self.stocks) - downloaded_tickers
        if missing_stocks:
            validation_results['issues'].append(f"Missing critical stocks: {missing_stocks}")
            print(f"❌ Missing stocks: {missing_stocks}")
        else:
            print("✅ All stocks in universe downloaded")

        # Factor-specific validation
        print(f"\n📊 Factor Universe Validation:")
        missing_factors = set(self.factors) - downloaded_tickers
        if missing_factors:
            validation_results['issues'].append(f"Missing base factors: {missing_factors}")
            print(f"❌ Missing base factors: {missing_factors}")
        else:
            print("✅ All base factors downloaded")

        # Overall validation result
        if len(validation_results['issues']) == 0:
            print(f"\n✅ DATA VALIDATION PASSED")
            print(f"Ready to proceed with {len(self.raw_data.columns)} tickers")
            print(f"Data range: {validation_results['data_start_date']} to {validation_results['data_end_date']}")
        else:
            validation_results['validation_passed'] = False
            print(f"\n❌ DATA VALIDATION ISSUES DETECTED:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        return validation_results


class ReturnsCalculator:
    def __init__(self):
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

    def calculate_weekly_returns(self, raw_data):
        """Convert daily prices to weekly returns and calculate factor spreads."""
        print(f"Converting daily prices to weekly returns...")

        # Resample to weekly prices (using Friday close, or last available price of week)
        self.weekly_prices = raw_data.resample('W-FRI').last()

        print(f"Daily data shape: {raw_data.shape}")
        print(f"Weekly data shape: {self.weekly_prices.shape}")
        print(f"Weekly data range: {self.weekly_prices.index[0]} to {self.weekly_prices.index[-1]}")

        # Calculate weekly returns as percentage
        weekly_returns = self.weekly_prices.pct_change().dropna() * 100

        # Extract stock returns
        stock_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PNC', 'TFC', 'USB', 'ALL', 'MET',
                         'PRU']
        available_stocks = [ticker for ticker in stock_tickers if ticker in weekly_returns.columns]
        self.stock_returns = weekly_returns[available_stocks].copy()

        # Extract base factor returns
        base_factors = ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']
        available_base_factors = [factor for factor in base_factors if factor in weekly_returns.columns]
        self.factor_returns = weekly_returns[available_base_factors].copy()

        # Calculate factor spreads
        self._calculate_factor_spreads(weekly_returns)

        # Combine all factor returns
        self._combine_all_factors()

        print(f"Stock returns shape: {self.stock_returns.shape}")
        print(f"Total factor returns shape: {self.all_factor_returns.shape}")

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
        if self.factor_returns is not None and not self.factor_returns.empty:
            factor_dfs.append(self.factor_returns)

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

    def validate_returns_calculation(self):
        """Validate the calculated returns for quality and correctness."""
        print("\n=== RETURNS CALCULATION VALIDATION ===")

        validation_results = {
            'stock_returns_shape': self.stock_returns.shape if self.stock_returns is not None else (0, 0),
            'factor_returns_shape': self.all_factor_returns.shape if self.all_factor_returns is not None else (0, 0),
            'date_range': {
                'start': self.stock_returns.index[0] if self.stock_returns is not None and len(
                    self.stock_returns) > 0 else None,
                'end': self.stock_returns.index[-1] if self.stock_returns is not None and len(
                    self.stock_returns) > 0 else None
            },
            'validation_passed': True,
            'issues': []
        }

        # Check if returns were calculated
        if self.stock_returns is None or self.stock_returns.empty:
            validation_results['issues'].append("Stock returns not calculated")
            validation_results['validation_passed'] = False
            print("❌ Stock returns not calculated")
            return validation_results

        if self.all_factor_returns is None or self.all_factor_returns.empty:
            validation_results['issues'].append("Factor returns not calculated")
            validation_results['validation_passed'] = False
            print("❌ Factor returns not calculated")
            return validation_results

        # Validate stock returns
        print(f"📊 Stock Returns Analysis:")
        print(f"✅ Shape: {self.stock_returns.shape}")
        print(f"✅ Date range: {self.stock_returns.index[0]} to {self.stock_returns.index[-1]}")

        # Check for extreme returns
        extreme_threshold = 50  # 50% weekly return threshold
        for stock in self.stock_returns.columns:
            extreme_returns = (np.abs(self.stock_returns[stock]) > extreme_threshold).sum()
            if extreme_returns > 5:  # More than 5 extreme weeks
                validation_results['issues'].append(f"{stock}: {extreme_returns} extreme weekly returns")
                print(f"⚠️  {stock}: {extreme_returns} weeks with >50% returns")

        # Check return distributions
        print(f"\n📈 Return Distribution Analysis:")
        stock_return_stats = self.stock_returns.describe()
        mean_weekly_return = stock_return_stats.loc['mean'].mean()
        std_weekly_return = stock_return_stats.loc['std'].mean()

        print(f"✅ Average weekly return: {mean_weekly_return:.2f}%")
        print(f"✅ Average weekly volatility: {std_weekly_return:.2f}%")

        if abs(mean_weekly_return) > 2:  # More than 2% average weekly return seems high
            validation_results['issues'].append(f"Unusually high average weekly return: {mean_weekly_return:.2f}%")
            print(f"⚠️  High average weekly return: {mean_weekly_return:.2f}%")

        # Validate factor returns
        print(f"\n📊 Factor Returns Analysis:")
        print(f"✅ Total factors: {self.all_factor_returns.shape[1]}")
        print(f"✅ Date range: {self.all_factor_returns.index[0]} to {self.all_factor_returns.index[-1]}")

        # Check factor categories
        factor_categories = {
            'Base Factors': len([col for col in self.all_factor_returns.columns if
                                 col in ['XLF', 'VFH', 'IYF', 'KRE', '^GSPC', '^VIX', '^TNX', 'FAS', 'DIA', 'GLD']]),
            'Rotation Factors': len([col for col in self.all_factor_returns.columns if 'vs' in col or '_vs_' in col]),
            'Other Factors': self.all_factor_returns.shape[1]
        }

        for category, count in factor_categories.items():
            if count > 0:
                print(f"✅ {category}: {count}")

        # Check for missing data
        missing_data_pct = (self.all_factor_returns.isnull().sum() / len(self.all_factor_returns)) * 100
        high_missing = missing_data_pct[missing_data_pct > 10]

        if len(high_missing) > 0:
            validation_results['issues'].append(f"Factors with >10% missing data: {list(high_missing.index)}")
            print(f"⚠️  Factors with >10% missing data: {list(high_missing.index)}")

        # Check data alignment
        if len(self.stock_returns) != len(self.all_factor_returns):
            validation_results['issues'].append("Stock and factor returns have different lengths")
            print(f"⚠️  Length mismatch: Stocks {len(self.stock_returns)}, Factors {len(self.all_factor_returns)}")
        else:
            print(f"✅ Data alignment: {len(self.stock_returns)} weeks")

        # Validate factor spreads calculation
        print(f"\n🔄 Factor Spread Validation:")
        spread_factors = [col for col in self.all_factor_returns.columns if '_vs_' in col]
        print(f"✅ Calculated {len(spread_factors)} spread factors")

        # Check spread factor properties
        for factor in spread_factors[:3]:  # Check first 3 as examples
            if factor in self.all_factor_returns.columns:
                factor_data = self.all_factor_returns[factor].dropna()
                if len(factor_data) > 0:
                    factor_mean = factor_data.mean()
                    factor_std = factor_data.std()
                    print(f"✅ {factor}: mean={factor_mean:.2f}%, std={factor_std:.2f}%")

        # Overall validation result
        if len(validation_results['issues']) == 0:
            print(f"\n✅ RETURNS CALCULATION VALIDATION PASSED")
            print(
                f"Ready for PCA analysis with {len(self.stock_returns)} stocks and {len(self.all_factor_returns.columns)} factors")
        else:
            validation_results['validation_passed'] = False
            print(f"\n❌ RETURNS CALCULATION VALIDATION ISSUES:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        return validation_results


class PCAAnalyzer:
    def __init__(self):
        # Data storage
        self.correlation_matrices = {}
        self.pca_loadings = {}
        self.eigenvalues = {}
        self.pc_returns = {}
        self.pc_standard_deviations = {}
        self.centrality_vectors = {}
        self.rebalance_dates = []
        self.lookback_weeks = 52

    def perform_rolling_pca(self, stock_returns):
        """Perform rolling PCA analysis on stock returns."""
        print(f"Performing rolling PCA analysis...")

        # Generate rebalance dates (weekly)
        start_date = stock_returns.index[self.lookback_weeks]  # First date with full lookback
        self.rebalance_dates = stock_returns.index[stock_returns.index >= start_date].tolist()

        print(f"PCA analysis period: {start_date} to {stock_returns.index[-1]}")
        print(f"Total rebalance dates: {len(self.rebalance_dates)}")

        for i, rebalance_date in enumerate(self.rebalance_dates):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing date {i + 1}/{len(self.rebalance_dates)}: {rebalance_date}")

            # Get lookback period data
            end_idx = stock_returns.index.get_loc(rebalance_date)
            start_idx = end_idx - self.lookback_weeks + 1

            lookback_returns = stock_returns.iloc[start_idx:end_idx + 1]

            # Skip if insufficient data
            if len(lookback_returns) < self.lookback_weeks:
                continue

            # Calculate correlation matrix and PCA
            self._calculate_pca_for_date(rebalance_date, lookback_returns)

        print(f"✅ PCA analysis complete for {len(self.rebalance_dates)} dates")

    def _calculate_pca_for_date(self, rebalance_date, lookback_returns):
        """Calculate PCA for a specific rebalance date."""

        # Remove any stocks with missing data in the lookback period
        clean_returns = lookback_returns.dropna(axis=1)

        if clean_returns.shape[1] < 3:  # Need at least 3 stocks for meaningful PCA
            return

        # Step 1: Standardize returns (subtract mean, divide by std)
        standardized_returns = (clean_returns - clean_returns.mean()) / clean_returns.std()

        # Step 2: Calculate correlation matrix
        correlation_matrix = standardized_returns.corr()
        self.correlation_matrices[rebalance_date] = correlation_matrix

        # Step 3: Calculate PCA using eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store eigenvalues
        self.eigenvalues[rebalance_date] = eigenvalues

        # Step 4: Create PCA loadings matrix (rescaled by stock return std devs)
        stock_std_devs = clean_returns.std().values
        loadings = eigenvectors.copy()

        # CORRECTED: Rescale loadings by dividing each stock's loadings by that stock's std dev
        for i in range(loadings.shape[0]):
            loadings[i, :] = loadings[i, :] / stock_std_devs[i]

        # Flip signs if sum of loadings is negative (for consistency)
        for j in range(loadings.shape[1]):
            if np.sum(loadings[:, j]) < 0:
                loadings[:, j] = -loadings[:, j]

        # Store loadings as DataFrame
        self.pca_loadings[rebalance_date] = pd.DataFrame(
            loadings,
            index=clean_returns.columns,
            columns=[f'PC{i + 1}' for i in range(len(eigenvalues))]
        )

        # Step 5: Calculate PC returns by projecting stock returns onto loadings
        # Use ORIGINAL returns (not standardized) for projection to maintain scale
        pc_returns_matrix = clean_returns.values @ loadings
        pc_returns_df = pd.DataFrame(
            pc_returns_matrix,
            index=clean_returns.index,
            columns=[f'PC{i + 1}' for i in range(len(eigenvalues))]
        )
        self.pc_returns[rebalance_date] = pc_returns_df

        # Step 6: Calculate PC standard deviations
        pc_std_devs = pc_returns_df.std().values
        self.pc_standard_deviations[rebalance_date] = pc_std_devs

        # Step 7: Calculate centrality vector (dominant eigenvector of correlation matrix)
        # CORRECTED: Don't take absolute values, preserve directional information
        centrality = eigenvectors[:, 0]  # First eigenvector

        # Normalize to have mean = 1.0 and std = 0.13 as specified
        centrality = centrality / np.mean(centrality)  # Mean = 1.0
        current_std = np.std(centrality)
        if current_std > 0:
            target_std = 0.13
            centrality = 1.0 + (centrality - 1.0) * (target_std / current_std)

        self.centrality_vectors[rebalance_date] = pd.Series(
            centrality,
            index=clean_returns.columns
        )

    def validate_pca_results(self):
        """Validate the PCA analysis results."""
        print("\n=== PCA ANALYSIS VALIDATION ===")

        validation_results = {
            'total_rebalance_dates': len(self.rebalance_dates),
            'successful_pca_calculations': len(self.pca_loadings),
            'average_stocks_per_analysis': 0,
            'average_eigenvalue_explained_variance': {},
            'centrality_vector_stats': {},
            'validation_passed': True,
            'issues': []
        }

        if len(self.pca_loadings) == 0:
            validation_results['issues'].append("No PCA calculations completed")
            validation_results['validation_passed'] = False
            print("❌ No PCA calculations completed")
            return validation_results

        # Calculate average number of stocks
        total_stocks = sum(len(loadings) for loadings in self.pca_loadings.values())
        validation_results['average_stocks_per_analysis'] = total_stocks / len(self.pca_loadings)

        print(f"📊 PCA Analysis Summary:")
        print(f"✅ Rebalance dates: {len(self.rebalance_dates)}")
        print(f"✅ Successful PCA calculations: {len(self.pca_loadings)}")
        print(f"✅ Average stocks per analysis: {validation_results['average_stocks_per_analysis']:.1f}")

        # Validate eigenvalues and explained variance
        print(f"\n📈 Eigenvalue Analysis:")
        sample_dates = list(self.eigenvalues.keys())[:5]  # Check first 5 dates

        for date in sample_dates:
            eigenvals = self.eigenvalues[date]
            total_variance = np.sum(eigenvals)

            # Check that eigenvalues are in descending order
            if not np.all(eigenvals[:-1] >= eigenvals[1:]):
                validation_results['issues'].append(f"Eigenvalues not in descending order for {date}")
                print(f"⚠️  {date}: Eigenvalues not properly ordered")

            # Calculate explained variance for first 5 PCs
            explained_var = eigenvals[:5] / total_variance * 100
            validation_results['average_eigenvalue_explained_variance'][str(date)] = explained_var.tolist()

            print(f"✅ {date}: PC1={explained_var[0]:.1f}%, PC2={explained_var[1]:.1f}%, PC3={explained_var[2]:.1f}%")

        # Validate PCA loadings
        print(f"\n🔄 PCA Loadings Validation:")
        sample_date = list(self.pca_loadings.keys())[0]
        sample_loadings = self.pca_loadings[sample_date]

        # Check loadings matrix properties
        num_stocks, num_pcs = sample_loadings.shape
        print(f"✅ Loadings matrix shape: {num_stocks} stocks × {num_pcs} PCs")

        # Check for reasonable loading values
        loading_stats = sample_loadings.describe()
        extreme_loadings = np.abs(sample_loadings).max().max()

        if extreme_loadings > 10:  # Loadings should generally be reasonable
            validation_results['issues'].append(f"Extreme loading values detected: {extreme_loadings:.2f}")
            print(f"⚠️  Extreme loading value: {extreme_loadings:.2f}")
        else:
            print(f"✅ Loading values reasonable (max: {extreme_loadings:.2f})")

        # Validate PC returns
        print(f"\n📊 PC Returns Validation:")
        sample_pc_returns = self.pc_returns[sample_date]

        # Check PC orthogonality (should be uncorrelated)
        pc_corr = sample_pc_returns.corr()
        off_diagonal = pc_corr.values[np.triu_indices_from(pc_corr.values, k=1)]
        max_correlation = np.abs(off_diagonal).max()

        if max_correlation > 0.1:  # PCs should be nearly orthogonal
            validation_results['issues'].append(f"PCs not orthogonal (max correlation: {max_correlation:.3f})")
            print(f"⚠️  PC orthogonality issue: max correlation = {max_correlation:.3f}")
        else:
            print(f"✅ PC orthogonality good (max correlation: {max_correlation:.3f})")

        # Validate centrality vectors
        print(f"\n🎯 Centrality Vector Validation:")
        centrality_means = []
        centrality_stds = []

        for date in sample_dates:
            if date in self.centrality_vectors:
                centrality = self.centrality_vectors[date]
                centrality_means.append(centrality.mean())
                centrality_stds.append(centrality.std())

        avg_mean = np.mean(centrality_means)
        avg_std = np.mean(centrality_stds)

        validation_results['centrality_vector_stats'] = {
            'average_mean': avg_mean,
            'average_std': avg_std
        }

        print(f"✅ Centrality vector mean: {avg_mean:.3f} (target: 1.000)")
        print(f"✅ Centrality vector std: {avg_std:.3f} (target: 0.130)")

        if abs(avg_mean - 1.0) > 0.05:
            validation_results['issues'].append(f"Centrality mean off target: {avg_mean:.3f}")
            print(f"⚠️  Centrality mean deviation: {avg_mean:.3f}")

        if abs(avg_std - 0.13) > 0.02:
            validation_results['issues'].append(f"Centrality std off target: {avg_std:.3f}")
            print(f"⚠️  Centrality std deviation: {avg_std:.3f}")

        # Check data consistency across dates
        print(f"\n🔄 Data Consistency Check:")
        stock_counts = [len(loadings) for loadings in self.pca_loadings.values()]
        min_stocks = min(stock_counts)
        max_stocks = max(stock_counts)

        if max_stocks - min_stocks > 2:  # Some variation is expected due to data availability
            validation_results['issues'].append(f"Inconsistent stock counts: {min_stocks}-{max_stocks}")
            print(f"⚠️  Stock count variation: {min_stocks} to {max_stocks}")
        else:
            print(f"✅ Consistent stock counts: {min_stocks} to {max_stocks}")

        # Overall validation result
        if len(validation_results['issues']) == 0:
            print(f"\n✅ PCA ANALYSIS VALIDATION PASSED")
            print(f"Ready for factor regression with {len(self.pca_loadings)} complete PCA analyses")
        else:
            validation_results['validation_passed'] = False
            print(f"\n❌ PCA ANALYSIS VALIDATION ISSUES:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        return validation_results


import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


class FactorRegressor:
    def __init__(self):
        # Data storage for regression models
        self.lasso_models = {}  # Dict[date][pc] -> fitted LassoCV model
        self.selected_factors = {}  # Dict[date][pc] -> list of selected factors
        self.regression_coefficients = {}  # Dict[date][pc] -> coefficients dict
        self.training_r2 = {}  # Dict[date][pc] -> R² on training data
        self.alpha_values = {}  # Dict[date][pc] -> optimal regularization parameter
        self.predicted_pc_returns = {}  # Dict[date] -> predicted PC returns vector

        # Factor data storage
        self.factor_returns = None
        self.pc_returns = None
        self.pc_standard_deviations = None

        # Configuration
        self.lookback_weeks = 52
        self.max_pcs_to_model = 5  # Model first 5 PCs
        self.cv_folds = 5  # Cross-validation folds

        # Standardization scalers for each date and PC
        self.factor_scalers = {}  # Dict[date][pc] -> StandardScaler for factors
        self.pc_scalers = {}  # Dict[date][pc] -> StandardScaler for PC returns

    def build_regression_models(self, all_factor_returns, pca_analyzer):
        """Build Lasso regression models for each PC using factor returns."""
        print(f"Building regression models for PC prediction...")

        # Store references to data
        self.factor_returns = all_factor_returns
        self.pc_returns = pca_analyzer.pc_returns
        self.pc_standard_deviations = pca_analyzer.pc_standard_deviations

        # Get rebalance dates from PCA analyzer
        rebalance_dates = pca_analyzer.rebalance_dates

        print(f"Building models for {len(rebalance_dates)} rebalance dates")
        print(f"Factor data shape: {self.factor_returns.shape}")

        for i, rebalance_date in enumerate(rebalance_dates):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing regression models for date {i + 1}/{len(rebalance_dates)}: {rebalance_date}")

            self._build_models_for_date(rebalance_date)

        print(f"✅ Regression model building complete for {len(rebalance_dates)} dates")

    def _build_models_for_date(self, rebalance_date):
        """Build regression models for all PCs on a specific date."""

        # Skip if no PCA data available for this date
        if rebalance_date not in self.pc_returns:
            return

        # Get PC returns for this date
        pc_data = self.pc_returns[rebalance_date]

        # Determine training period (52 weeks before rebalance date)
        end_idx = self.factor_returns.index.get_loc(rebalance_date)
        start_idx = max(0, end_idx - self.lookback_weeks + 1)

        # Get training data
        factor_training = self.factor_returns.iloc[start_idx:end_idx + 1].copy()

        # Align PC returns with factor returns (same date range)
        pc_training = pc_data.reindex(factor_training.index, method='nearest').copy()

        # Skip if insufficient training data
        if len(factor_training) < 20 or len(pc_training) < 20:
            return

        # Remove any rows with NaN values
        combined_data = pd.concat([factor_training, pc_training], axis=1).dropna()

        if len(combined_data) < 20:  # Need minimum data for robust regression
            return

        # Split back into factors and PCs
        factor_cols = factor_training.columns
        pc_cols = pc_training.columns

        X_train = combined_data[factor_cols]
        y_train_all = combined_data[pc_cols]

        # Initialize storage for this date
        self.lasso_models[rebalance_date] = {}
        self.selected_factors[rebalance_date] = {}
        self.regression_coefficients[rebalance_date] = {}
        self.training_r2[rebalance_date] = {}
        self.alpha_values[rebalance_date] = {}
        self.factor_scalers[rebalance_date] = {}
        self.pc_scalers[rebalance_date] = {}

        # Build model for each PC (up to max_pcs_to_model)
        num_pcs = min(self.max_pcs_to_model, len(pc_cols))

        for pc_idx in range(num_pcs):
            pc_name = pc_cols[pc_idx]

            try:
                self._build_single_pc_model(rebalance_date, pc_name, X_train, y_train_all[pc_name])
            except Exception as e:
                # Skip this PC if model building fails
                print(f"Warning: Failed to build model for {pc_name} on {rebalance_date}: {e}")
                continue

    def _build_single_pc_model(self, rebalance_date, pc_name, X_train, y_train):
        """Build Lasso regression model for a single PC."""

        # Remove any remaining NaN values
        valid_idx = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_clean = X_train[valid_idx]
        y_clean = y_train[valid_idx]

        if len(X_clean) < 10:  # Need minimum samples
            return

        # Standardize features and target
        factor_scaler = StandardScaler()
        pc_scaler = StandardScaler()

        X_scaled = factor_scaler.fit_transform(X_clean)
        y_scaled = pc_scaler.fit_transform(y_clean.values.reshape(-1, 1)).flatten()

        # Store scalers
        self.factor_scalers[rebalance_date][pc_name] = factor_scaler
        self.pc_scalers[rebalance_date][pc_name] = pc_scaler

        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(self.cv_folds, len(X_scaled) // 10))

        # Define alpha range for Lasso regularization
        alphas = np.logspace(-4, 1, 50)  # From 0.0001 to 10

        # Fit Lasso with cross-validation
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=tscv,
            random_state=42,
            max_iter=2000,
            selection='random'
        )

        lasso_cv.fit(X_scaled, y_scaled)

        # Store the fitted model
        self.lasso_models[rebalance_date][pc_name] = lasso_cv

        # Store optimal alpha
        self.alpha_values[rebalance_date][pc_name] = lasso_cv.alpha_

        # Calculate training R²
        y_pred_scaled = lasso_cv.predict(X_scaled)
        training_r2 = r2_score(y_scaled, y_pred_scaled)
        self.training_r2[rebalance_date][pc_name] = training_r2

        # Store coefficients and selected factors
        coefficients = lasso_cv.coef_
        factor_names = X_clean.columns.tolist()

        # Identify selected factors (non-zero coefficients)
        selected_indices = np.abs(coefficients) > 1e-6  # Small threshold for numerical precision
        selected_factor_names = [factor_names[i] for i in range(len(factor_names)) if selected_indices[i]]
        selected_coefficients = {factor_names[i]: coefficients[i] for i in range(len(coefficients))}

        self.selected_factors[rebalance_date][pc_name] = selected_factor_names
        self.regression_coefficients[rebalance_date][pc_name] = selected_coefficients

    def predict_pc_movements(self):
        """Generate PC predictions for the next week using the fitted models."""
        print(f"Generating PC predictions for next week...")

        prediction_count = 0

        for rebalance_date in self.lasso_models.keys():

            # Get the next week's factor data for prediction
            try:
                next_week_date = self._get_next_week_date(rebalance_date)
                if next_week_date is None:
                    continue

                # Get factor returns for the prediction date
                if next_week_date not in self.factor_returns.index:
                    continue

                factor_data_next = self.factor_returns.loc[next_week_date]

                # Initialize prediction storage for this date
                pc_predictions = {}

                # Generate prediction for each modeled PC
                for pc_name in self.lasso_models[rebalance_date].keys():
                    try:
                        prediction = self._predict_single_pc(rebalance_date, pc_name, factor_data_next)
                        if prediction is not None:
                            pc_predictions[pc_name] = prediction
                    except Exception as e:
                        # Skip this PC prediction if it fails
                        continue

                if pc_predictions:
                    self.predicted_pc_returns[rebalance_date] = pc_predictions
                    prediction_count += 1

            except Exception as e:
                # Skip this date if prediction fails
                continue

        print(f"✅ Generated PC predictions for {prediction_count} dates")

    def _predict_single_pc(self, rebalance_date, pc_name, factor_data_next):
        """Predict a single PC's movement for next week."""

        # Get the trained model and scalers
        model = self.lasso_models[rebalance_date][pc_name]
        factor_scaler = self.factor_scalers[rebalance_date][pc_name]
        pc_scaler = self.pc_scalers[rebalance_date][pc_name]

        # Get the factors that were used in training
        training_factors = factor_scaler.feature_names_in_

        # Extract relevant factor data
        factor_values = []
        for factor in training_factors:
            if factor in factor_data_next.index and not pd.isna(factor_data_next[factor]):
                factor_values.append(factor_data_next[factor])
            else:
                # If factor data is missing, use 0 (neutral)
                factor_values.append(0.0)

        factor_values = np.array(factor_values).reshape(1, -1)

        # Standardize using training scaler
        factor_values_scaled = factor_scaler.transform(factor_values)

        # Make prediction (in scaled space)
        pc_pred_scaled = model.predict(factor_values_scaled)[0]

        # Transform back to original scale
        pc_pred = pc_scaler.inverse_transform([[pc_pred_scaled]])[0][0]

        return pc_pred

    def _get_next_week_date(self, current_date):
        """Get the next week's date for prediction."""
        try:
            current_idx = self.factor_returns.index.get_loc(current_date)
            if current_idx + 1 < len(self.factor_returns.index):
                return self.factor_returns.index[current_idx + 1]
            else:
                return None
        except (KeyError, IndexError):
            return None

    def validate_regression_models(self):
        """Validate the regression model results."""
        print("\n=== FACTOR REGRESSION VALIDATION ===")

        validation_results = {
            'total_models_built': 0,
            'successful_predictions': len(self.predicted_pc_returns),
            'average_training_r2': {},
            'factor_selection_stats': {},
            'alpha_distribution': {},
            'model_stability': {},
            'validation_passed': True,
            'issues': []
        }

        if len(self.lasso_models) == 0:
            validation_results['issues'].append("No regression models built")
            validation_results['validation_passed'] = False
            print("❌ No regression models built")
            return validation_results

        # Count total models
        total_models = sum(len(models) for models in self.lasso_models.values())
        validation_results['total_models_built'] = total_models

        print(f"📊 Regression Model Summary:")
        print(f"✅ Total models built: {total_models}")
        print(f"✅ Rebalance dates with models: {len(self.lasso_models)}")
        print(f"✅ Successful predictions: {len(self.predicted_pc_returns)}")

        # Validate training R² values
        print(f"\n📈 Training R² Analysis:")
        all_r2_values = []
        pc_r2_summary = {}

        for date, models in self.training_r2.items():
            for pc_name, r2_value in models.items():
                all_r2_values.append(r2_value)

                if pc_name not in pc_r2_summary:
                    pc_r2_summary[pc_name] = []
                pc_r2_summary[pc_name].append(r2_value)

        if all_r2_values:
            avg_r2 = np.mean(all_r2_values)
            median_r2 = np.median(all_r2_values)
            min_r2 = np.min(all_r2_values)
            max_r2 = np.max(all_r2_values)

            validation_results['average_training_r2']['overall'] = {
                'mean': avg_r2,
                'median': median_r2,
                'min': min_r2,
                'max': max_r2
            }

            print(f"✅ Overall R² - Mean: {avg_r2:.3f}, Median: {median_r2:.3f}")
            print(f"✅ R² Range: {min_r2:.3f} to {max_r2:.3f}")

            # Check for concerning R² values
            negative_r2_count = sum(1 for r2 in all_r2_values if r2 < 0)
            high_r2_count = sum(1 for r2 in all_r2_values if r2 > 0.8)

            if negative_r2_count > total_models * 0.1:  # More than 10% negative
                validation_results['issues'].append(f"High number of negative R² values: {negative_r2_count}")
                print(f"⚠️  {negative_r2_count} models with negative R² (possible overfitting)")

            if high_r2_count > total_models * 0.05:  # More than 5% very high
                validation_results['issues'].append(f"Suspiciously high R² values: {high_r2_count}")
                print(f"⚠️  {high_r2_count} models with R² > 0.8 (possible overfitting)")

        # PC-specific R² analysis
        for pc_name, r2_values in pc_r2_summary.items():
            avg_pc_r2 = np.mean(r2_values)
            validation_results['average_training_r2'][pc_name] = avg_pc_r2
            print(f"✅ {pc_name}: Average R² = {avg_pc_r2:.3f} ({len(r2_values)} models)")

        # Validate factor selection
        print(f"\n🎯 Factor Selection Analysis:")
        all_selected_factors = []
        factors_frequency = {}

        for date, models in self.selected_factors.items():
            for pc_name, selected_factors in models.items():
                all_selected_factors.extend(selected_factors)
                for factor in selected_factors:
                    factors_frequency[factor] = factors_frequency.get(factor, 0) + 1

        if factors_frequency:
            # Most frequently selected factors
            top_factors = sorted(factors_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

            validation_results['factor_selection_stats'] = {
                'total_factor_selections': len(all_selected_factors),
                'unique_factors_selected': len(factors_frequency),
                'top_factors': top_factors
            }

            avg_factors_per_model = len(all_selected_factors) / total_models if total_models > 0 else 0

            print(f"✅ Average factors per model: {avg_factors_per_model:.1f}")
            print(f"✅ Total unique factors selected: {len(factors_frequency)}")
            print(f"✅ Top 5 factors: {[f[0] for f in top_factors[:5]]}")

            if avg_factors_per_model > 20:  # Too many factors selected
                validation_results['issues'].append(f"Too many factors per model: {avg_factors_per_model:.1f}")
                print(f"⚠️  High average factors per model: {avg_factors_per_model:.1f}")

        # Validate alpha values
        print(f"\n🎛️  Regularization Parameter Analysis:")
        all_alphas = []

        for date, models in self.alpha_values.items():
            for pc_name, alpha in models.items():
                all_alphas.append(alpha)

        if all_alphas:
            avg_alpha = np.mean(all_alphas)
            median_alpha = np.median(all_alphas)

            validation_results['alpha_distribution'] = {
                'mean': avg_alpha,
                'median': median_alpha,
                'min': np.min(all_alphas),
                'max': np.max(all_alphas)
            }

            print(f"✅ Alpha values - Mean: {avg_alpha:.4f}, Median: {median_alpha:.4f}")

            # Check for extreme alpha values
            very_high_alpha = sum(1 for a in all_alphas if a > 1)
            very_low_alpha = sum(1 for a in all_alphas if a < 0.001)

            if very_high_alpha > len(all_alphas) * 0.1:
                validation_results['issues'].append(f"Many models with very high regularization: {very_high_alpha}")
                print(f"⚠️  {very_high_alpha} models with alpha > 1.0 (heavy regularization)")

            if very_low_alpha > len(all_alphas) * 0.1:
                validation_results['issues'].append(f"Many models with very low regularization: {very_low_alpha}")
                print(f"⚠️  {very_low_alpha} models with alpha < 0.001 (minimal regularization)")

        # Check prediction coverage
        print(f"\n🎯 Prediction Coverage Analysis:")
        prediction_rate = len(self.predicted_pc_returns) / len(self.lasso_models) if len(self.lasso_models) > 0 else 0
        print(f"✅ Prediction success rate: {prediction_rate:.1%}")

        if prediction_rate < 0.8:  # Less than 80% prediction success
            validation_results['issues'].append(f"Low prediction success rate: {prediction_rate:.1%}")
            print(f"⚠️  Low prediction coverage: {prediction_rate:.1%}")

        # Overall validation result
        if len(validation_results['issues']) == 0:
            print(f"\n✅ FACTOR REGRESSION VALIDATION PASSED")
            print(f"Models ready for stock return prediction")
        else:
            validation_results['validation_passed'] = False
            print(f"\n❌ FACTOR REGRESSION VALIDATION ISSUES:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        return validation_results


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self):
        # Data storage for stock predictions
        self.predicted_stock_returns = {}  # Dict[date] -> dict[stock] -> predicted return %
        self.actual_stock_returns = {}  # Dict[date] -> dict[stock] -> actual return %
        self.prediction_r2_training = {}  # Dict[date] -> dict[stock] -> R² on training data
        self.prediction_r2_testing = {}  # Dict[date] -> dict[stock] -> R² on testing data

        # Data for plotting
        self.scatter_plot_data = []  # List of dicts with 'date', 'stock', 'predicted', 'actual'
        self.stock_r2_timeseries = {}  # Dict[stock] -> {'training': [], 'testing': [], 'dates': []}

        # Data references
        self.stock_returns = None
        self.pca_analyzer = None
        self.factor_regressor = None

    def generate_stock_predictions(self, stock_returns, pca_analyzer, factor_regressor):
        """Generate stock return predictions using PC predictions and PCA loadings."""
        print(f"Generating stock return predictions...")

        # Store references
        self.stock_returns = stock_returns
        self.pca_analyzer = pca_analyzer
        self.factor_regressor = factor_regressor

        prediction_count = 0

        for rebalance_date in factor_regressor.predicted_pc_returns.keys():
            try:
                # Get predicted PC movements for this date
                pc_predictions = factor_regressor.predicted_pc_returns[rebalance_date]

                # Get PCA loadings for this date
                if rebalance_date not in pca_analyzer.pca_loadings:
                    continue

                loadings = pca_analyzer.pca_loadings[rebalance_date]
                centrality = pca_analyzer.centrality_vectors[rebalance_date]
                pc_std_devs = pca_analyzer.pc_standard_deviations[rebalance_date]

                # Generate predictions for this date
                stock_predictions = self._predict_stocks_for_date(
                    rebalance_date, pc_predictions, loadings, centrality, pc_std_devs
                )

                if stock_predictions:
                    self.predicted_stock_returns[rebalance_date] = stock_predictions

                    # Get actual returns for next week
                    actual_returns = self._get_actual_returns_next_week(rebalance_date)
                    if actual_returns:
                        self.actual_stock_returns[rebalance_date] = actual_returns

                        # Calculate R² values
                        self._calculate_r2_values(rebalance_date, stock_predictions, actual_returns, loadings)

                        # Store scatter plot data
                        self._store_scatter_data(rebalance_date, stock_predictions, actual_returns)

                        prediction_count += 1

            except Exception as e:
                # Skip this date if prediction fails
                continue

        # Build time series of R² values
        self._build_r2_timeseries()

        print(f"✅ Generated stock predictions for {prediction_count} dates")
        print(f"✅ Scatter plot data points: {len(self.scatter_plot_data)}")

    def _predict_stocks_for_date(self, rebalance_date, pc_predictions, loadings, centrality, pc_std_devs):
        """Predict individual stock returns for a specific date."""

        # Convert PC predictions to array
        pc_predictions_array = np.zeros(len(pc_std_devs))
        for pc_name, predicted_return in pc_predictions.items():
            pc_idx = int(pc_name.replace('PC', '')) - 1  # Convert PC1 to index 0
            if pc_idx < len(pc_std_devs):
                pc_predictions_array[pc_idx] = predicted_return

        # CORRECTED: Apply PC standard deviations before loadings transformation
        # This implements: V · (ΔPC_pred ⊙ σ_PC)
        scaled_pc_predictions = pc_predictions_array * pc_std_devs[:len(pc_predictions_array)]

        # Calculate raw predicted returns using PCA loadings
        raw_predictions = loadings.values @ scaled_pc_predictions

        # Apply centrality weighting
        # Final prediction = raw_prediction * (centrality / mean_centrality)
        centrality_weights = centrality / centrality.mean()
        adjusted_predictions = raw_predictions * centrality_weights.values

        # Create dictionary of stock predictions
        stock_predictions = {}
        for i, stock in enumerate(loadings.index):
            stock_predictions[stock] = adjusted_predictions[i]

        return stock_predictions

    def _get_actual_returns_next_week(self, rebalance_date):
        """Get actual stock returns for the week following the rebalance date."""
        try:
            # Find the next week's date
            current_idx = self.stock_returns.index.get_loc(rebalance_date)
            if current_idx + 1 < len(self.stock_returns.index):
                next_week_date = self.stock_returns.index[current_idx + 1]
                actual_returns = self.stock_returns.loc[next_week_date].to_dict()

                # Filter out any NaN values
                actual_returns = {k: v for k, v in actual_returns.items() if not pd.isna(v)}
                return actual_returns
            else:
                return None
        except (KeyError, IndexError):
            return None

    def _calculate_r2_values(self, rebalance_date, predicted_returns, actual_returns, loadings):
        """Calculate R² values for training and testing periods."""

        # Get common stocks between predictions and actuals
        common_stocks = set(predicted_returns.keys()) & set(actual_returns.keys())

        if len(common_stocks) < 3:  # Need minimum stocks for meaningful R²
            return

        # Calculate R² for testing (predicted vs actual)
        pred_values = [predicted_returns[stock] for stock in common_stocks]
        actual_values = [actual_returns[stock] for stock in common_stocks]

        if len(pred_values) > 1:
            testing_r2 = r2_score(actual_values, pred_values)
        else:
            testing_r2 = 0.0

        # Calculate R² for training (how well the regression explains the training period)
        # This requires reconstructing what the model predicted for the training period
        training_r2_dict = self._calculate_training_r2(rebalance_date, loadings, common_stocks)

        # Store results
        self.prediction_r2_testing[rebalance_date] = {stock: testing_r2 for stock in common_stocks}
        self.prediction_r2_training[rebalance_date] = training_r2_dict

    def _calculate_training_r2(self, rebalance_date, loadings, stocks):
        """Calculate R² between regression line and lookback period for each stock."""

        training_r2_dict = {}

        try:
            # Get the training period data (52 weeks before rebalance date)
            end_idx = self.stock_returns.index.get_loc(rebalance_date)
            start_idx = max(0, end_idx - 51)  # 52 weeks including current

            training_data = self.stock_returns.iloc[start_idx:end_idx + 1]

            # Get PC returns for the same period
            if rebalance_date in self.pca_analyzer.pc_returns:
                pc_returns = self.pca_analyzer.pc_returns[rebalance_date]
                pc_std_devs = self.pca_analyzer.pc_standard_deviations[rebalance_date]

                # Align PC returns with training period
                aligned_pc_returns = pc_returns.reindex(training_data.index, method='nearest')

                # For each stock, calculate how well PC projections explain actual returns
                for stock in stocks:
                    if stock in training_data.columns and stock in loadings.index:

                        actual_stock_returns = training_data[stock].dropna()
                        stock_loadings = loadings.loc[stock].values[:len(aligned_pc_returns.columns)]

                        # CORRECTED: Reconstruct stock returns using proper scaling
                        reconstructed_returns = []
                        for date in actual_stock_returns.index:
                            if date in aligned_pc_returns.index:
                                pc_values = aligned_pc_returns.loc[date].values[:len(stock_loadings)]
                                # Apply the same scaling as in prediction: loadings @ (PC_values * PC_std_devs)
                                scaled_pc_values = pc_values * pc_std_devs[:len(pc_values)]
                                reconstructed_return = np.dot(stock_loadings, scaled_pc_values)
                                reconstructed_returns.append(reconstructed_return)

                        # Calculate R² between actual and reconstructed returns
                        if len(reconstructed_returns) == len(actual_stock_returns) and len(reconstructed_returns) > 1:
                            training_r2 = r2_score(actual_stock_returns.values, reconstructed_returns)
                            training_r2_dict[stock] = training_r2
                        else:
                            training_r2_dict[stock] = 0.0

        except Exception as e:
            # If calculation fails, assign 0 R²
            for stock in stocks:
                training_r2_dict[stock] = 0.0

        return training_r2_dict

    def _store_scatter_data(self, rebalance_date, predicted_returns, actual_returns):
        """Store data for scatter plot generation."""

        common_stocks = set(predicted_returns.keys()) & set(actual_returns.keys())

        for stock in common_stocks:
            self.scatter_plot_data.append({
                'date': rebalance_date,
                'stock': stock,
                'predicted': predicted_returns[stock],
                'actual': actual_returns[stock]
            })

    def _build_r2_timeseries(self):
        """Build time series of R² values for each stock."""

        # Initialize storage for each stock
        all_stocks = set()
        for date_dict in self.prediction_r2_training.values():
            all_stocks.update(date_dict.keys())
        for date_dict in self.prediction_r2_testing.values():
            all_stocks.update(date_dict.keys())

        for stock in all_stocks:
            self.stock_r2_timeseries[stock] = {
                'training': [],
                'testing': [],
                'dates': []
            }

        # Collect R² values over time
        all_dates = sorted(set(self.prediction_r2_training.keys()) | set(self.prediction_r2_testing.keys()))

        for date in all_dates:
            for stock in all_stocks:
                training_r2 = self.prediction_r2_training.get(date, {}).get(stock, np.nan)
                testing_r2 = self.prediction_r2_testing.get(date, {}).get(stock, np.nan)

                self.stock_r2_timeseries[stock]['dates'].append(date)
                self.stock_r2_timeseries[stock]['training'].append(training_r2)
                self.stock_r2_timeseries[stock]['testing'].append(testing_r2)

    def create_scatter_plot(self):
        """Create scatter plot of predicted vs actual returns."""

        if len(self.scatter_plot_data) == 0:
            print("No data available for scatter plot")
            return None

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.scatter_plot_data)

        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['predicted'], df['actual'], alpha=0.6, s=20)

        # Add diagonal line (perfect prediction line)
        min_val = min(df['predicted'].min(), df['actual'].min())
        max_val = max(df['predicted'].max(), df['actual'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')

        # Calculate and display overall R²
        overall_r2 = r2_score(df['actual'], df['predicted'])

        plt.xlabel('Predicted Returns (%)')
        plt.ylabel('Actual Returns (%)')
        plt.title(f'Predicted vs Actual Stock Returns\n(R² = {overall_r2:.4f}, n = {len(df)} predictions)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add some statistics as text
        plt.text(0.05, 0.95,
                 f'Total predictions: {len(df)}\nUnique stocks: {df["stock"].nunique()}\nDate range: {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return plt.gcf()

    def create_r2_timeseries_plots(self):
        """Create R² time series plots for each stock."""

        if len(self.stock_r2_timeseries) == 0:
            print("No R² time series data available")
            return []

        figures = []

        for stock in sorted(self.stock_r2_timeseries.keys()):
            data = self.stock_r2_timeseries[stock]

            if len(data['dates']) == 0:
                continue

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Convert dates to pandas datetime for better plotting
            dates = pd.to_datetime(data['dates'])
            training_r2 = np.array(data['training'])
            testing_r2 = np.array(data['testing'])

            # Remove NaN values for plotting
            valid_training = ~np.isnan(training_r2)
            valid_testing = ~np.isnan(testing_r2)

            # Plot 1: Training R² (regression line vs lookback period)
            if np.any(valid_training):
                ax1.plot(dates[valid_training], training_r2[valid_training],
                         'b-', marker='o', markersize=3, linewidth=1, alpha=0.7)
                ax1.set_title(f'{stock} - Training R² (Regression vs 52-Week Lookback)')
                ax1.set_ylabel('R² Value')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1)

                # Add average line
                avg_training_r2 = np.mean(training_r2[valid_training])
                ax1.axhline(y=avg_training_r2, color='red', linestyle='--', alpha=0.7,
                            label=f'Average: {avg_training_r2:.3f}')
                ax1.legend()

            # Plot 2: Testing R² (predicted vs actual next week)
            if np.any(valid_testing):
                ax2.plot(dates[valid_testing], testing_r2[valid_testing],
                         'g-', marker='s', markersize=3, linewidth=1, alpha=0.7)
                ax2.set_title(f'{stock} - Testing R² (Predicted vs Actual Next Week)')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('R² Value')
                ax2.grid(True, alpha=0.3)

                # Add average line
                avg_testing_r2 = np.mean(testing_r2[valid_testing])
                ax2.axhline(y=avg_testing_r2, color='red', linestyle='--', alpha=0.7,
                            label=f'Average: {avg_testing_r2:.3f}')
                ax2.legend()

            plt.tight_layout()
            figures.append(fig)

        return figures

    def validate_predictions(self):
        """Validate the stock prediction results."""
        print("\n=== STOCK PREDICTION VALIDATION ===")

        validation_results = {
            'total_prediction_dates': len(self.predicted_stock_returns),
            'total_scatter_points': len(self.scatter_plot_data),
            'stocks_with_predictions': 0,
            'overall_testing_r2': 0.0,
            'average_training_r2': 0.0,
            'prediction_statistics': {},
            'r2_statistics': {},
            'validation_passed': True,
            'issues': []
        }

        if len(self.predicted_stock_returns) == 0:
            validation_results['issues'].append("No stock predictions generated")
            validation_results['validation_passed'] = False
            print("❌ No stock predictions generated")
            return validation_results

        # Basic statistics
        print(f"📊 Prediction Summary:")
        print(f"✅ Prediction dates: {len(self.predicted_stock_returns)}")
        print(f"✅ Scatter plot points: {len(self.scatter_plot_data)}")

        # Calculate overall statistics
        if self.scatter_plot_data:
            df = pd.DataFrame(self.scatter_plot_data)
            validation_results['stocks_with_predictions'] = df['stock'].nunique()
            validation_results['overall_testing_r2'] = r2_score(df['actual'], df['predicted'])

            print(f"✅ Unique stocks: {df['stock'].nunique()}")
            print(f"✅ Overall testing R²: {validation_results['overall_testing_r2']:.4f}")

            # Prediction range statistics
            pred_stats = df['predicted'].describe()
            actual_stats = df['actual'].describe()

            validation_results['prediction_statistics'] = {
                'predicted_mean': pred_stats['mean'],
                'predicted_std': pred_stats['std'],
                'actual_mean': actual_stats['mean'],
                'actual_std': actual_stats['std']
            }

            print(f"✅ Predicted returns: mean={pred_stats['mean']:.2f}%, std={pred_stats['std']:.2f}%")
            print(f"✅ Actual returns: mean={actual_stats['mean']:.2f}%, std={actual_stats['std']:.2f}%")

        # R² analysis
        print(f"\n📈 R² Analysis:")

        # Training R² statistics
        all_training_r2 = []
        for date_dict in self.prediction_r2_training.values():
            all_training_r2.extend([r2 for r2 in date_dict.values() if not np.isnan(r2)])

        if all_training_r2:
            validation_results['average_training_r2'] = np.mean(all_training_r2)
            print(f"✅ Average training R²: {validation_results['average_training_r2']:.4f}")
            print(f"✅ Training R² range: {np.min(all_training_r2):.3f} to {np.max(all_training_r2):.3f}")

        # Testing R² by stock
        if self.stock_r2_timeseries:
            print(f"\n🎯 Stock-specific R² Analysis:")
            stock_avg_testing_r2 = {}

            for stock, data in self.stock_r2_timeseries.items():
                testing_values = [r2 for r2 in data['testing'] if not np.isnan(r2)]
                if testing_values:
                    avg_testing_r2 = np.mean(testing_values)
                    stock_avg_testing_r2[stock] = avg_testing_r2

            if stock_avg_testing_r2:
                # Show top 5 and bottom 5 performers
                sorted_stocks = sorted(stock_avg_testing_r2.items(), key=lambda x: x[1], reverse=True)

                print("Top 5 stocks by testing R²:")
                for stock, r2 in sorted_stocks[:5]:
                    print(f"  ✅ {stock}: {r2:.4f}")

                print("Bottom 5 stocks by testing R²:")
                for stock, r2 in sorted_stocks[-5:]:
                    print(f"  ⚠️  {stock}: {r2:.4f}")

                validation_results['r2_statistics'] = {
                    'best_stock': sorted_stocks[0],
                    'worst_stock': sorted_stocks[-1],
                    'average_across_stocks': np.mean(list(stock_avg_testing_r2.values()))
                }

        # Check for concerning patterns
        if validation_results['overall_testing_r2'] < -0.1:
            validation_results['issues'].append(f"Very poor overall R²: {validation_results['overall_testing_r2']:.4f}")
            print(f"⚠️  Very poor overall predictive performance")

        if validation_results['average_training_r2'] > 0.8:
            validation_results['issues'].append("Very high training R² suggests overfitting")
            print(f"⚠️  High training R² may indicate overfitting")

        # Data quality checks
        if len(self.scatter_plot_data) < 1000:
            validation_results['issues'].append("Limited prediction data for analysis")
            print(f"⚠️  Limited data points for robust analysis")

        # Overall validation result
        if len(validation_results['issues']) == 0:
            print(f"\n✅ STOCK PREDICTION VALIDATION PASSED")
            print(f"Ready for visualization and analysis")
        else:
            validation_results['validation_passed'] = False
            print(f"\n❌ STOCK PREDICTION VALIDATION ISSUES:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")

        return validation_results


# Updated Main Class
class StrategyValidator:
    def __init__(self):
        print("Initializing Strategy Validator...")
        self.data_manager = DataManager()
        self.returns_calculator = ReturnsCalculator()
        self.pca_analyzer = PCAAnalyzer()
        self.factor_regressor = FactorRegressor()
        self.stock_predictor = StockPredictor()

    def run_validation_step1(self):
        """Run Step 1: Data Download and Validation"""
        print("=" * 50)
        print("STEP 1: DATA DOWNLOAD AND VALIDATION")
        print("=" * 50)

        self.data_manager.download_data()
        validation_results = self.data_manager.validate_data_integrity()
        return validation_results

    def run_validation_step2(self):
        """Run Step 2: Returns Calculation and Validation"""
        print("=" * 50)
        print("STEP 2: RETURNS CALCULATION AND VALIDATION")
        print("=" * 50)

        self.returns_calculator.calculate_weekly_returns(self.data_manager.raw_data)
        validation_results = self.returns_calculator.validate_returns_calculation()
        return validation_results

    def run_validation_step3(self):
        """Run Step 3: PCA Analysis and Validation"""
        print("=" * 50)
        print("STEP 3: PCA ANALYSIS AND VALIDATION")
        print("=" * 50)

        self.pca_analyzer.perform_rolling_pca(self.returns_calculator.stock_returns)
        validation_results = self.pca_analyzer.validate_pca_results()
        return validation_results

    def run_validation_step4(self):
        """Run Step 4: Factor Regression and Validation"""
        print("=" * 50)
        print("STEP 4: FACTOR REGRESSION AND VALIDATION")
        print("=" * 50)

        self.factor_regressor.build_regression_models(
            self.returns_calculator.all_factor_returns,
            self.pca_analyzer
        )
        self.factor_regressor.predict_pc_movements()
        validation_results = self.factor_regressor.validate_regression_models()
        return validation_results

    def run_validation_step5(self):
        """Run Step 5: Stock Prediction and Validation"""
        print("=" * 50)
        print("STEP 5: STOCK PREDICTION AND VALIDATION")
        print("=" * 50)

        self.stock_predictor.generate_stock_predictions(
            self.returns_calculator.stock_returns,
            self.pca_analyzer,
            self.factor_regressor
        )
        validation_results = self.stock_predictor.validate_predictions()
        return validation_results

    def run_all_validation_steps(self):
        """Run all validation steps"""
        results = []

        # Step 1
        step1_results = self.run_validation_step1()
        results.append(step1_results)
        if not step1_results['validation_passed']:
            print("❌ Step 1 validation failed. Stopping.")
            return results

        # Step 2
        step2_results = self.run_validation_step2()
        results.append(step2_results)
        if not step2_results['validation_passed']:
            print("❌ Step 2 validation failed. Stopping.")
            return results

        # Step 3
        step3_results = self.run_validation_step3()
        results.append(step3_results)
        if not step3_results['validation_passed']:
            print("❌ Step 3 validation failed. Stopping.")
            return results

        # Step 4
        step4_results = self.run_validation_step4()
        results.append(step4_results)
        if not step4_results['validation_passed']:
            print("⚠️  Step 4 has validation warnings but continuing...")

        # Step 5
        step5_results = self.run_validation_step5()
        results.append(step5_results)

        return results

    def create_all_plots(self):
        """Create and display all plots"""
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATION PLOTS")
        print("=" * 50)

        # Create scatter plot
        print("Creating scatter plot...")
        scatter_fig = self.stock_predictor.create_scatter_plot()
        if scatter_fig:
            scatter_fig.show()

        # Create R² time series plots
        print("Creating R² time series plots...")
        r2_figures = self.stock_predictor.create_r2_timeseries_plots()

        print(f"Created {len(r2_figures)} R² time series plots")

        # Show first few plots (to avoid overwhelming display)
        for i, fig in enumerate(r2_figures[:5]):  # Show first 5 stocks
            fig.show()

        if len(r2_figures) > 5:
            print(f"Note: Only showing first 5 stock plots. {len(r2_figures) - 5} additional plots created.")

        return scatter_fig, r2_figures

    def debug_scaling_pipeline(self):
        """Debug function to trace scaling through the entire pipeline."""
        print("\n" + "=" * 60)
        print("SCALING DEBUG ANALYSIS")
        print("=" * 60)

        if len(self.stock_predictor.scatter_plot_data) == 0:
            print("No scatter plot data available for debugging")
            return

        # Get a sample date for detailed analysis
        sample_date = list(self.factor_regressor.predicted_pc_returns.keys())[0]
        print(f"Analyzing sample date: {sample_date}")

        # 1. Check raw weekly returns scale
        sample_stock_returns = self.returns_calculator.stock_returns.loc[sample_date]
        print(f"\n1. WEEKLY STOCK RETURNS SCALE:")
        print(f"   Sample values: {sample_stock_returns.head(3).values}")
        print(f"   Range: {sample_stock_returns.min():.4f} to {sample_stock_returns.max():.4f}")
        print(f"   Mean: {sample_stock_returns.mean():.4f}, Std: {sample_stock_returns.std():.4f}")

        # 2. Check PC returns scale
        if sample_date in self.pca_analyzer.pc_returns:
            sample_pc_returns = self.pca_analyzer.pc_returns[sample_date]
            sample_pc_data = sample_pc_returns.loc[sample_date] if sample_date in sample_pc_returns.index else \
            sample_pc_returns.iloc[-1]
            print(f"\n2. PC RETURNS SCALE:")
            print(f"   Sample PC values: {sample_pc_data.head(3).values}")
            print(f"   Range: {sample_pc_data.min():.4f} to {sample_pc_data.max():.4f}")
            print(f"   Mean: {sample_pc_data.mean():.4f}, Std: {sample_pc_data.std():.4f}")

        # 3. Check PC standard deviations
        if sample_date in self.pca_analyzer.pc_standard_deviations:
            pc_std_devs = self.pca_analyzer.pc_standard_deviations[sample_date]
            print(f"\n3. PC STANDARD DEVIATIONS:")
            print(f"   PC std devs: {pc_std_devs[:5]}")
            print(f"   Range: {pc_std_devs.min():.4f} to {pc_std_devs.max():.4f}")

        # 4. Check factor returns scale
        sample_factor_returns = self.returns_calculator.all_factor_returns.loc[sample_date]
        print(f"\n4. FACTOR RETURNS SCALE:")
        print(f"   Sample factor values: {sample_factor_returns.head(3).values}")
        print(f"   Range: {sample_factor_returns.min():.4f} to {sample_factor_returns.max():.4f}")
        print(f"   Mean: {sample_factor_returns.mean():.4f}, Std: {sample_factor_returns.std():.4f}")

        # 5. Check predicted PC movements
        if sample_date in self.factor_regressor.predicted_pc_returns:
            predicted_pcs = self.factor_regressor.predicted_pc_returns[sample_date]
            print(f"\n5. PREDICTED PC MOVEMENTS:")
            print(f"   Predicted PC values: {list(predicted_pcs.values())}")
            print(f"   Range: {min(predicted_pcs.values()):.4f} to {max(predicted_pcs.values()):.4f}")

        # 6. Check PCA loadings scale
        if sample_date in self.pca_analyzer.pca_loadings:
            loadings = self.pca_analyzer.pca_loadings[sample_date]
            print(f"\n6. PCA LOADINGS SCALE:")
            print(f"   Sample loadings (first stock, first 3 PCs): {loadings.iloc[0, :3].values}")
            print(f"   Loadings range: {loadings.min().min():.4f} to {loadings.max().max():.4f}")
            print(f"   Loadings mean: {loadings.mean().mean():.4f}, Std: {loadings.std().mean():.4f}")

        # 7. Check predicted stock returns scale
        if sample_date in self.stock_predictor.predicted_stock_returns:
            predicted_stocks = self.stock_predictor.predicted_stock_returns[sample_date]
            pred_values = list(predicted_stocks.values())
            print(f"\n7. PREDICTED STOCK RETURNS:")
            print(f"   Sample predicted values: {pred_values[:3]}")
            print(f"   Range: {min(pred_values):.4f} to {max(pred_values):.4f}")
            print(f"   Mean: {np.mean(pred_values):.4f}, Std: {np.std(pred_values):.4f}")

        # 8. Check actual stock returns scale
        if sample_date in self.stock_predictor.actual_stock_returns:
            actual_stocks = self.stock_predictor.actual_stock_returns[sample_date]
            actual_values = list(actual_stocks.values())
            print(f"\n8. ACTUAL STOCK RETURNS (NEXT WEEK):")
            print(f"   Sample actual values: {actual_values[:3]}")
            print(f"   Range: {min(actual_values):.4f} to {max(actual_values):.4f}")
            print(f"   Mean: {np.mean(actual_values):.4f}, Std: {np.std(actual_values):.4f}")

        # 9. Overall scatter plot analysis
        df = pd.DataFrame(self.stock_predictor.scatter_plot_data)
        print(f"\n9. SCATTER PLOT DATA ANALYSIS:")
        print(f"   Predicted range: {df['predicted'].min():.4f} to {df['predicted'].max():.4f}")
        print(f"   Actual range: {df['actual'].min():.4f} to {df['actual'].max():.4f}")
        print(f"   Predicted std: {df['predicted'].std():.4f}")
        print(f"   Actual std: {df['actual'].std():.4f}")
        print(f"   Std ratio (actual/predicted): {df['actual'].std() / df['predicted'].std():.2f}")

        # 10. Check for any unit inconsistencies
        print(f"\n10. UNIT CONSISTENCY CHECK:")

        # Check if factor regression is doing any scaling
        for date in list(self.factor_regressor.factor_scalers.keys())[:1]:
            for pc in list(self.factor_regressor.factor_scalers[date].keys())[:1]:
                factor_scaler = self.factor_regressor.factor_scalers[date][pc]
                pc_scaler = self.factor_regressor.pc_scalers[date][pc]
                print(f"   Factor scaler mean: {factor_scaler.mean_[:3]}")
                print(f"   Factor scaler scale: {factor_scaler.scale_[:3]}")
                print(f"   PC scaler mean: {pc_scaler.mean_[0]:.4f}")
                print(f"   PC scaler scale: {pc_scaler.scale_[0]:.4f}")

        print(f"\n" + "=" * 60)
        print("END SCALING DEBUG")
        print("=" * 60)

# Add this to the end of your main execution after all validation steps
if __name__ == "__main__":
    validator = StrategyValidator()

    # Run all validation steps
    all_results = validator.run_all_validation_steps()

    # Run the debug analysis
    validator.debug_scaling_pipeline()

    # Create plots
    scatter_plot, r2_plots = validator.create_all_plots()