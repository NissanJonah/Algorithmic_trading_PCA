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

class PCAFactorStrategy:
    def __init__(self, stocks, start_date, end_date, lookback=252, initial_capital=10000):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.data = None
        self.rebalance_dates = None
        self.factors = ['XLI', '^TNX', '^GSPC', 'UUP', 'IYT', 'DIA', 'COPX', 'XME', '^VIX', 'GLD']
        self.factor_data = None
        self.pca_matrix_count = 0
        self.selected_factors = {}
        self.r2_history = {f'PC_{i+1}': [] for i in range(5)}
        self.r2_training_history = {f'PC_{i+1}': [] for i in range(5)}

    def download_data(self):
        nominal_start = pd.to_datetime(self.start_date)
        earliest_data_start = nominal_start - pd.offsets.BDay(self.lookback + 15 + 20)
        raw_data = yf.download(self.stocks, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            self.data = raw_data['Close']
        else:
            self.data = raw_data
        self.data = self.data.dropna(axis=0, how='any')
        raw_factor_data = yf.download(self.factors, start=earliest_data_start, end=self.end_date, auto_adjust=True)
        if isinstance(raw_factor_data.columns, pd.MultiIndex):
            self.factor_data = raw_factor_data['Close']
        else:
            self.factor_data = raw_factor_data
        self.factor_data = self.factor_data.dropna(axis=1, how='all').dropna(axis=0, how='any')
        if self.factor_data.empty or len(self.factor_data.columns) == 0:
            raise ValueError("No valid factor data available")
        self.factors = list(self.factor_data.columns)
        all_dates = self.data.index
        self.rebalance_dates = all_dates[all_dates.weekday == 4]
        first_possible_rebalance = all_dates[all_dates >= (nominal_start + pd.offsets.BDay(self.lookback))][0]
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates >= first_possible_rebalance]
        rebalance_diffs = self.rebalance_dates[1:] - self.rebalance_dates[:-1]
        rebalance_days = [diff.days for diff in rebalance_diffs]
        print(f"Stock data shape: {self.data.shape}, Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Factor data shape: {self.factor_data.shape}, Available factors: {self.factors}")
        print(f"Rebalance dates: {len(self.rebalance_dates)}, Mean days between: {np.mean(rebalance_days):.2f}")

    def compute_log_returns(self, prices):
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def standardize_returns(self, returns):
        scaler = StandardScaler()
        std_returns = pd.DataFrame(scaler.fit_transform(returns), index=returns.index, columns=returns.columns)
        return std_returns

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
        u_centrality = eigenvectors[:, idx]
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

    def compute_factor_changes(self, factor_prices, rebalance_date):
        if factor_prices.empty or len(factor_prices) < self.lookback:
            return pd.DataFrame()
        try:
            idx = factor_prices.index.get_loc(rebalance_date)
            lookback_start = factor_prices.index[max(0, idx - self.lookback + 1)]
            prices = factor_prices.loc[lookback_start:rebalance_date]
            returns = self.compute_log_returns(prices)
            return returns
        except KeyError:
            return pd.DataFrame()

    def factor_regression(self, pc_series, factor_changes, rebalance_date):
        if factor_changes.empty or rebalance_date not in pc_series.index or rebalance_date not in factor_changes.index:
            return {}
        common_dates = pc_series.index.intersection(factor_changes.index)
        pc_series_full = pc_series.loc[common_dates]
        factor_changes_full = factor_changes.loc[common_dates]
        if len(common_dates) < 50:
            return {}
        try:
            rebalance_idx = pc_series.index.get_loc(rebalance_date)
            start_idx = max(0, rebalance_idx - 4)
            period_dates = pc_series.index[start_idx:rebalance_idx + 1]
            pc_series_period = pc_series.loc[period_dates]
            factor_changes_period = factor_changes.loc[factor_changes.index.intersection(period_dates)]
            common_period_dates = pc_series_period.index.intersection(factor_changes_period.index)
            pc_series_period = pc_series_period.loc[common_period_dates]
            factor_changes_period = factor_changes_period.loc[common_period_dates]
            if len(common_period_dates) < 3:
                return {}
        except (KeyError, IndexError):
            return {}
        results = {}
        self.selected_factors = {}
        for pc in pc_series_full.columns:
            y_full = pc_series_full[pc].to_numpy()
            X_full = factor_changes_full.to_numpy()
            if np.any(np.isnan(y_full)) or np.any(np.isnan(X_full)):
                self.r2_history[pc].append((rebalance_date, 0.0))
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                continue
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_full)
            y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()
            correlations = pd.Series(np.abs(np.corrcoef(X_scaled.T, y_scaled)[:-1, -1]), index=self.factors)
            best_subset = []
            sorted_factors = correlations[correlations > 0.05].sort_values(ascending=False)
            if len(sorted_factors) == 0:
                self.r2_history[pc].append((rebalance_date, 0.0))
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                continue
            best_subset.append(sorted_factors.index[0])
            if len(sorted_factors) > 1:
                for f in sorted_factors.index[1:]:
                    if all(abs(factor_changes_full[f].corr(factor_changes_full[bf])) < 0.75 for bf in best_subset):
                        best_subset.append(f)
                        break
            X_subset_full = factor_changes_full[best_subset].to_numpy()
            if np.any(np.isnan(X_subset_full)):
                self.r2_history[pc].append((rebalance_date, 0.0))
                self.r2_training_history[pc].append((rebalance_date, 0.0))
                continue
            model = LinearRegression()
            model.fit(X_subset_full, y_full)
            r2_training = model.score(X_subset_full, y_full)
            adjusted_r2_training = 1 - (1 - r2_training) * (len(y_full) - 1) / (len(y_full) - X_subset_full.shape[1] - 1)
            y_period = pc_series_period[pc].to_numpy()
            X_period = factor_changes_period[best_subset].to_numpy()
            if len(X_period) == len(y_period) and len(y_period) >= 3:
                r2_rebalancing = model.score(X_period, y_period)
                adjusted_r2_rebalancing = 1 - (1 - r2_rebalancing) * (len(y_period) - 1) / (len(y_period) - X_period.shape[1] - 1)
                results[pc] = {
                    'alpha': model.intercept_,
                    'beta': dict(zip(best_subset, model.coef_)),
                    'r2_training': adjusted_r2_training,
                    'r2_rebalancing': adjusted_r2_rebalancing,
                    'factors': best_subset
                }
                self.selected_factors[pc] = best_subset
                self.r2_history[pc].append((rebalance_date, adjusted_r2_rebalancing))
                self.r2_training_history[pc].append((rebalance_date, adjusted_r2_training))
        return results

    def validate_step2(self):
        validation_results = []
        for rebalance_date in self.rebalance_dates:
            idx = self.data.index.get_loc(rebalance_date)
            lookback_start = self.data.index[max(0, idx - self.lookback + 1)]
            prices = self.data.loc[lookback_start:rebalance_date]
            returns = self.compute_log_returns(prices)
            std_returns = self.standardize_returns(returns)
            V, eigenvalues, explained_var = self.compute_pca(std_returns)
            C, u_centrality = self.compute_centrality_matrix(std_returns)
            pc_series = self.compute_pc_series(std_returns, V)
            factor_changes = self.compute_factor_changes(self.factor_data, rebalance_date)
            regression_results = self.factor_regression(pc_series, factor_changes, rebalance_date)
            validation_results.append({
                'date': rebalance_date,
                'returns_shape': returns.shape,
                'std_returns_mean': std_returns.mean().mean(),
                'std_returns_std': std_returns.std().mean(),
                'pca_explained_var': explained_var.tolist(),
                'pca_orthogonal': np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-6),
                'centrality_mean': np.mean(u_centrality),
                'centrality_std': np.std(u_centrality),
                'pc_series_shape': pc_series.shape,
                'regression_results': regression_results
            })

        print("\nValidation Summary:")
        print(f"Total PCA matrices: {self.pca_matrix_count}")
        print(f"Total rebalance periods: {len(validation_results)}")
        print("\nLast 5 Rebalance Periods:")
        for result in validation_results[-5:]:
            print(f"\nDate: {result['date'].date()}")
            print(f"Returns Shape: {result['returns_shape']}")
            print(f"Std Returns Mean: {result['std_returns_mean']:.6f}, Std: {result['std_returns_std']:.6f}")
            print(f"PC Sector Movement: " + ", ".join(f"PC_{i+1}: {var*100:.2f}%" for i, var in enumerate(result['pca_explained_var'])))
            print(f"PCA Orthogonal: {result['pca_orthogonal']}")
            print(f"Centrality Mean: {result['centrality_mean']:.6f}, Std: {result['centrality_std']:.6f}")
            print(f"PC Series Shape: {result['pc_series_shape']}")
            for pc, res in result['regression_results'].items():
                print(f"{pc}:")
                print(f"  Eq: PC = {res['alpha']:.6f} + " + " + ".join(f"{coef:.6f}*{f}" for f, coef in res['beta'].items()))
                print(f"  Factors: {res['factors']}")
                print(f"  R² (252-day): {res['r2_training']:.6f}, R² (5-day): {res['r2_rebalancing']:.6f}")

        if any(self.r2_history.values()):
            print("\nR² Summary (Top 3 PCs):")
            for pc in [f'PC_{i+1}' for i in range(min(3, len(self.r2_history)))]:
                if self.r2_history[pc]:
                    training_r2s = [r2 for _, r2 in self.r2_training_history[pc]]
                    rebalancing_r2s = [r2 for _, r2 in self.r2_history[pc]]
                    print(f"{pc}:")
                    print(f"  Training R² Mean: {np.mean(training_r2s):.4f}, Std: {np.std(training_r2s):.4f}")
                    print(f"  Rebalancing R² Mean: {np.mean(rebalancing_r2s):.4f}, Std: {np.std(rebalancing_r2s):.4f}")
                    print(f"  R² Correlation: {np.corrcoef(training_r2s, rebalancing_r2s)[0,1]:.4f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        for pc in [f'PC_{i+1}' for i in range(min(3, len(self.r2_training_history)))]:
            if self.r2_training_history[pc]:
                dates, r2s = zip(*self.r2_training_history[pc])
                ax1.plot(dates, r2s, marker='o', label=f'Training R² {pc}')
        ax1.set_title('Training R² (252-day)')
        ax1.set_ylabel('Adjusted R²')
        ax1.legend()
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        for pc in [f'PC_{i+1}' for i in range(min(3, len(self.r2_history)))]:
            if self.r2_history[pc]:
                dates, r2s = zip(*self.r2_history[pc])
                ax2.plot(dates, r2s, marker='s', label=f'Rebalancing R² {pc}')
        ax2.set_title('Rebalancing R² (5-day)')
        ax2.set_xlabel('Rebalance Date')
        ax2.set_ylabel('Adjusted R²')
        ax2.legend()
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    stocks = ['GE', 'CAT', 'RTX', 'UNP', 'HON', 'BA', 'DE', 'UPS', 'LMT', 'ETN', 'PH', 'ITW', 'GD', 'NOC', 'MMM']
    start_date = '2021-01-01'
    end_date = '2025-08-22'
    strategy = PCAFactorStrategy(stocks, start_date, end_date)
    strategy.download_data()
    strategy.validate_step2()