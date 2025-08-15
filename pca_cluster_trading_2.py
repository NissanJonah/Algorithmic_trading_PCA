import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import joblib
import os
from datetime import datetime

# -----------------------------
# 1. Define sector ETFs & stocks
# -----------------------------
sector_lists = {
    "Technology": ["XLK"],
    "Finance": ["XLF"],
    "Healthcare": ["XLV"],
    "Consumer Discretionary": ["XLY"],
    "Consumer Staples": ["XLP"],
    "Energy": ["XLE"],
    "Materials": ["XLB"],
    "Industrials": ["XLI"],
    "Real Estate": ["XLRE"],
    "Utilities": ["XLU"],
    "Communication Services": ["XLC"]
}

# -----------------------------
# 2. Download and preprocess data
# -----------------------------
all_etfs = [etf for etfs in sector_lists.values() for etf in etfs]
data = yf.download(all_etfs, period="20y", auto_adjust=True)
data = data['Close']
print(f"Using {len(data.columns)} ETFs after filtering.")

# Compute daily returns for ETFs
returns = data.pct_change().dropna()

# -----------------------------
# 3. Train/Test Split (No data leakage)
# -----------------------------
split_date = returns.index[int(len(returns) * 0.8)]
print(f"Split date: {split_date}")

train_returns = returns.loc[:split_date]
test_returns = returns.loc[split_date:]

print(f"Train samples: {len(train_returns)}")
print(f"Test samples: {len(test_returns)}")

# Standardize ETF returns using ONLY training statistics
train_mean = train_returns.mean()
train_std = train_returns.std()
train_returns_std = (train_returns - train_mean) / train_std
test_returns_std = (test_returns - train_mean) / train_std

# -----------------------------
# 4. Fit PCA on training data only
# -----------------------------
pca = PCA()
Y_train = pca.fit_transform(train_returns_std)
V = pca.components_.T
Y_test = pca.transform(test_returns_std)


# -----------------------------
# 5. Deterministic sign alignment
# -----------------------------
def align_pca_signs(V, Y_train, Y_test=None):
    V_aligned = V.copy()
    Y_train_aligned = Y_train.copy()
    Y_test_aligned = Y_test.copy() if Y_test is not None else None

    for i in range(V.shape[1]):
        max_loading_idx = np.argmax(np.abs(V[:, i]))
        if V[max_loading_idx, i] < 0:
            V_aligned[:, i] *= -1
            Y_train_aligned[:, i] *= -1
            if Y_test_aligned is not None:
                Y_test_aligned[:, i] *= -1

    return V_aligned, Y_train_aligned, Y_test_aligned


V_aligned, Y_train_aligned, Y_test_aligned = align_pca_signs(V, Y_train, Y_test)

train_pc_df = pd.DataFrame(Y_train_aligned, index=train_returns_std.index,
                           columns=[f"PC{i + 1}" for i in range(Y_train_aligned.shape[1])])
test_pc_df = pd.DataFrame(Y_test_aligned, index=test_returns_std.index,
                          columns=[f"PC{i + 1}" for i in range(Y_test_aligned.shape[1])])

# -----------------------------
# 6. Standardize PC scores using training statistics
# -----------------------------
train_pc_mean = train_pc_df.mean()
train_pc_std = train_pc_df.std()
train_pc_scaled = (train_pc_df - train_pc_mean) / train_pc_std
test_pc_scaled = (test_pc_df - train_pc_mean) / train_pc_std

print("Train PC scaled stats (should be ~0 mean, ~1 std):")
print(f"Mean: {train_pc_scaled.mean().round(3)}")
print(f"Std: {train_pc_scaled.std().round(3)}")


# -----------------------------
# 7. Save preprocessing artifacts
# -----------------------------
def save_preprocessing_artifacts(train_mean, train_std, pca, V_aligned, train_pc_mean, train_pc_std,
                                 save_dir="pca_artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    artifacts = {
        'train_etf_mean': train_mean,
        'train_etf_std': train_std,
        'pca_object': pca,
        'pca_loadings_aligned': V_aligned,
        'train_pc_mean': train_pc_mean,
        'train_pc_std': train_pc_std,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'feature_names': train_mean.index.tolist(),
        'n_components': pca.n_components_,
        'split_date': split_date,
        'created_at': datetime.now()
    }
    artifact_path = os.path.join(save_dir, 'pca_preprocessing.joblib')
    joblib.dump(artifacts, artifact_path)
    print(f"Preprocessing artifacts saved to: {artifact_path}")
    return artifacts


artifacts = save_preprocessing_artifacts(train_mean, train_std, pca, V_aligned, train_pc_mean, train_pc_std)


# -----------------------------
# 8. Function to load and apply preprocessing
# -----------------------------
def load_and_transform_new_data(new_returns, artifact_path="pca_artifacts/pca_preprocessing.joblib"):
    artifacts = joblib.load(artifact_path)
    train_etf_mean = artifacts['train_etf_mean']
    train_etf_std = artifacts['train_etf_std']
    pca_object = artifacts['pca_object']
    V_aligned = artifacts['pca_loadings_aligned']
    train_pc_mean = artifacts['train_pc_mean']
    train_pc_std = artifacts['train_pc_std']

    new_returns_std = (new_returns - train_etf_mean) / train_etf_std
    Y_new = pca_object.transform(new_returns_std)

    for i in range(V_aligned.shape[1]):
        original_loading = pca_object.components_.T[:, i]
        aligned_loading = V_aligned[:, i]
        if not np.allclose(original_loading, aligned_loading):
            Y_new[:, i] *= -1

    pc_df = pd.DataFrame(Y_new, index=new_returns_std.index,
                         columns=[f"PC{i + 1}" for i in range(Y_new.shape[1])])
    pc_scaled = (pc_df - train_pc_mean) / train_pc_std
    return pc_scaled, artifacts


test_pc_inference, _ = load_and_transform_new_data(test_returns)
print(f"\nInference test - Max difference: {np.max(np.abs(test_pc_scaled.values - test_pc_inference.values)):.6f}")

# -----------------------------
# 9. Analysis and diagnostics
# -----------------------------
explained_var = pca.explained_variance_ratio_
print(f"\nExplained variance by PC1: {explained_var[0]:.2%}")

loadings_signs = np.sign(V_aligned[:, 0])
if (loadings_signs == loadings_signs[0]).all():
    print("PC1 likely represents overall market movement (all loadings have the same sign).")
else:
    print("PC1 may not represent overall market movement (mixed loading signs).")

print("\nTrain PC correlations (should be ~diagonal):")
train_corr = train_pc_df.corr()
print(f"Off-diagonal max: {np.max(np.abs(train_corr.values - np.eye(len(train_corr)))):.3f}")

print("\nTest PC correlations (should be near-diagonal):")
test_corr = test_pc_df.corr()
print(f"Off-diagonal max: {np.max(np.abs(test_corr.values - np.eye(len(test_corr)))):.3f}")


# -----------------------------
# 10. Rolling PCA analysis
# -----------------------------
def analyze_pca_stability_over_time(returns, sector_lists, window_size=63, etfs_to_plot=None, num_graphs=5):
    print("\n" + "=" * 60)
    print("RUNNING ROLLING PCA ANALYSIS (FOR DIAGNOSTICS ONLY)")
    print("=" * 60)

    ticker_to_sector = {ticker: sector for sector, tickers in sector_lists.items() for ticker in tickers}
    total_windows = len(returns) // window_size
    etfs_to_plot = returns.columns[:num_graphs] if etfs_to_plot is None else etfs_to_plot[:num_graphs]

    r2_over_time = {etf: {f'PC{i + 1}': [] for i in range(len(returns.columns))} for etf in etfs_to_plot}
    window_dates = []

    for w in range(total_windows):
        start = w * window_size
        end = start + window_size
        window_returns = returns.iloc[start:end]
        window_returns_std = (window_returns - window_returns.mean()) / window_returns.std()
        window_pca = PCA()
        Y_window = window_pca.fit_transform(window_returns_std)
        pc_df_window = pd.DataFrame(Y_window, index=window_returns_std.index,
                                    columns=[f'PC{i + 1}' for i in range(Y_window.shape[1])])

        mid_date = window_returns_std.index[window_size // 2]
        window_dates.append(mid_date)

        for etf in etfs_to_plot:
            y = window_returns[etf].loc[pc_df_window.index]
            for pc in pc_df_window.columns:
                X = sm.add_constant(pc_df_window[pc])
                model = sm.OLS(y, X).fit()
                r2_over_time[etf][pc].append(model.rsquared)

    for etf in etfs_to_plot:
        sector_name = ticker_to_sector.get(etf, "Unknown Sector")
        plt.figure(figsize=(12, 5))
        for pc in r2_over_time[etf]:
            plt.plot(window_dates, r2_over_time[etf][pc], marker='o', label=pc, markersize=4)
        plt.title(f"{sector_name} ({etf}) — Rolling R² Analysis (Diagnostic Only)")
        plt.xlabel("Date")
        plt.ylabel("R²")
        plt.ylim(0, 1)
        plt.legend(title="PC", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


analyze_pca_stability_over_time(returns, sector_lists)


# -----------------------------
# 11. Forecasting setup
# -----------------------------
def setup_forecasting_model(train_pc_scaled, lookback=5):
    print(f"\nSetting up forecasting model with lookback={lookback} days")
    X_list, y_list = [], []

    for i in range(lookback, len(train_pc_scaled)):
        X_list.append(train_pc_scaled.iloc[i - lookback:i].values.flatten())
        y_list.append(train_pc_scaled.iloc[i].values)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Forecasting dataset shape: X={X.shape}, y={y.shape}")
    return X, y


X_forecast, y_forecast = setup_forecasting_model(train_pc_scaled)

# -----------------------------
# 12. PCA Visualization Plots
# -----------------------------
full_pc_df = pd.concat([train_pc_df, test_pc_df])


def plot_all_pcs_over_time(pc_df):
    plt.figure(figsize=(14, 7))
    for pc in pc_df.columns:
        plt.plot(pc_df.index, pc_df[pc], label=pc, linewidth=1)
    plt.title("PCA Scores Over Time for All Principal Components")
    plt.xlabel("Date")
    plt.ylabel("PC Score")
    plt.legend(loc="upper right", fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_rolling_r2_for_etf(returns, pc_df, etf, window=63):
    r2_dict = {}
    returns_aligned = returns.loc[pc_df.index]

    for pc in pc_df.columns:
        r2_series = []
        for end_ix in range(window, len(pc_df) + 1):
            start_ix = end_ix - window
            y = returns_aligned[etf].iloc[start_ix:end_ix]
            X = pc_df[pc].iloc[start_ix:end_ix]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r2_series.append(model.rsquared)
        r2_series = [np.nan] * (window - 1) + r2_series
        r2_dict[pc] = r2_series
    return pd.DataFrame(r2_dict, index=pc_df.index)


def plot_etf_r2_over_time(returns, pc_df, sector_lists, window=63):
    ticker_to_sector = {ticker: sector for sector, tickers in sector_lists.items() for ticker in tickers}

    for etf in returns.columns:
        sector_name = ticker_to_sector.get(etf, "Unknown Sector")
        r2_df = compute_rolling_r2_for_etf(returns, pc_df, etf, window)

        plt.figure(figsize=(14, 7))
        for pc in r2_df.columns:
            plt.plot(r2_df.index, r2_df[pc], label=pc, linewidth=1.5)
        plt.title(f"{sector_name} ({etf}) — Rolling R² with All PCs Over Time")
        plt.xlabel("Date")
        plt.ylabel("R² (Rolling Window)")
        plt.ylim(0, 1)
        plt.legend(title="Principal Components", loc="upper right", fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_eigenvalues_bar(pca):
    eigenvalues = pca.explained_variance_
    pc_names = [f'PC{i + 1}' for i in range(len(eigenvalues))]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(pc_names, eigenvalues, alpha=0.7, color='steelblue')
    plt.title("Eigenvalues by Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, eigenvalues):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()


print("\n" + "=" * 60)
print("GENERATING THE 3 REQUESTED PCA PLOTS")
print("=" * 60)

print("1. Plotting all PCs over time...")
plot_all_pcs_over_time(full_pc_df)
# print("2. Plotting rolling R² for each ETF...")
# plot_etf_r2_over_time(returns, full_pc_df, sector_lists, window=63)
print("3. Plotting eigenvalues bar chart...")
plot_eigenvalues_bar(pca)
print("\n" + "=" * 60)
print("ALL 3 REQUESTED PLOTS GENERATED SUCCESSFULLY!")
print("=" * 60)


# -----------------------------
# Enhanced Sector Momentum Trading System
# -----------------------------
def calculate_sector_correlations_and_lags(returns, sector_lists, window=63, max_lag=5):
    print("Calculating inter-sector correlations and lead-lag relationships...")
    ticker_to_sector = {ticker: sector for sector, tickers in sector_lists.items() for ticker in tickers}
    sector_to_ticker = {sector: tickers[0] for sector, tickers in sector_lists.items()}
    sector_returns = pd.DataFrame({sector: returns[ticker] for sector, ticker in sector_to_ticker.items()})

    rolling_corrs, correlation_dates = [], []
    for i in range(window, len(sector_returns)):
        window_data = sector_returns.iloc[i - window:i]
        corr_matrix = window_data.corr()
        rolling_corrs.append(corr_matrix)
        correlation_dates.append(sector_returns.index[i])

    lead_lag_results = {}
    for sector1 in sector_returns.columns:
        lead_lag_results[sector1] = {}
        for sector2 in sector_returns.columns:
            if sector1 != sector2:
                correlations = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        corr = sector_returns[sector1].corr(sector_returns[sector2])
                    elif lag > 0:
                        s1 = sector_returns[sector1].iloc[:-lag]
                        s2 = sector_returns[sector2].iloc[lag:]
                        corr = s1.corr(s2)
                    else:
                        s1 = sector_returns[sector1].iloc[-lag:]
                        s2 = sector_returns[sector2].iloc[:lag]
                        corr = s1.corr(s2)
                    correlations.append(corr)
                max_corr_idx = np.nanargmax(np.abs(correlations))
                optimal_lag = max_corr_idx - max_lag
                optimal_corr = correlations[max_corr_idx]
                lead_lag_results[sector1][sector2] = {
                    'optimal_lag': optimal_lag,
                    'optimal_correlation': optimal_corr,
                    'all_correlations': correlations
                }

    return {
        'rolling_correlations': rolling_corrs,
        'correlation_dates': correlation_dates,
        'lead_lag_relationships': lead_lag_results,
        'sector_returns': sector_returns,
        'sector_to_ticker': sector_to_ticker
    }

# Modified: calculate_sector_momentum_rankings to support vol-adjusted momentum
def calculate_sector_momentum_rankings(returns, sector_lists, lookbacks, enable_vol_adjusted_momentum=False, vol_window=20):
    print(f"Calculating sector momentum rankings for lookbacks: {lookbacks}")
    if enable_vol_adjusted_momentum:
        print("  Using volatility-adjusted momentum scores")
    sector_to_ticker = {sector: tickers[0] for sector, tickers in sector_lists.items()}
    momentum_data = {}

    for lookback in lookbacks:
        print(f"  Processing {lookback}-day momentum...")
        momentum_data[f'{lookback}d'] = {}
        for i in range(lookback, len(returns)):
            date = returns.index[i]
            sector_momentum = {}
            for sector, ticker in sector_to_ticker.items():
                if ticker in returns.columns:
                    period_returns = returns[ticker].iloc[i - lookback:i + 1]
                    momentum = (1 + period_returns).prod() - 1
                    if enable_vol_adjusted_momentum:
                        vol = returns[ticker].iloc[i - vol_window:i].std()
                        momentum = momentum / vol if vol > 0 else momentum  # Sharpe-like adjustment
                    sector_momentum[sector] = momentum
            sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
            rankings = {sector: rank + 1 for rank, (sector, _) in enumerate(sorted_sectors)}
            momentum_scores = dict(sorted_sectors)
            momentum_data[f'{lookback}d'][date] = {
                'rankings': rankings,
                'scores': momentum_scores
            }

    ranking_dfs = {}
    score_dfs = {}
    for timeframe in momentum_data.keys():
        ranking_data, score_data, dates = [], [], []
        for date, data in momentum_data[timeframe].items():
            ranking_data.append(data['rankings'])
            score_data.append(data['scores'])
            dates.append(date)
        ranking_dfs[timeframe] = pd.DataFrame(ranking_data, index=dates)
        score_dfs[timeframe] = pd.DataFrame(score_data, index=dates)

    return {
        'rankings': ranking_dfs,
        'scores': score_dfs,
        'raw_data': momentum_data
    }


def identify_early_strength_sectors(momentum_rankings, corr_data, early_strength_threshold=3, min_lead_correlation=0.3):
    print("Identifying sectors with early strength...")
    short_rankings = momentum_rankings['rankings'][min(momentum_rankings['rankings'].keys())]
    medium_rankings = momentum_rankings['rankings'][sorted(momentum_rankings['rankings'].keys())[1]]

    lead_lag = corr_data['lead_lag_relationships']
    leading_sectors = set()
    for sector1, relationships in lead_lag.items():
        leading_count = 0
        total_significant = 0
        for sector2, data in relationships.items():
            if abs(data['optimal_correlation']) > min_lead_correlation:
                total_significant += 1
                if data['optimal_lag'] < 0:
                    leading_count += 1
        if total_significant > 0 and leading_count / total_significant > 0.5:
            leading_sectors.add(sector1)

    print(f"Identified leading sectors: {leading_sectors}")
    early_strength_signals = pd.DataFrame(0, index=short_rankings.index, columns=short_rankings.columns)

    for date in short_rankings.index:
        if date in medium_rankings.index:
            for sector in short_rankings.columns:
                short_rank = short_rankings.loc[date, sector]
                medium_rank = medium_rankings.loc[date, sector]
                is_early_strength = (
                        short_rank <= early_strength_threshold and
                        short_rank < medium_rank and
                        sector in leading_sectors
                )
                if is_early_strength:
                    early_strength_signals.loc[date, sector] = 1

    return early_strength_signals


def generate_multi_timeframe_consensus_signals(momentum_rankings, consensus_threshold=2):
    print("Generating multi-timeframe consensus signals...")
    rankings = momentum_rankings['rankings']
    common_dates = rankings[min(rankings.keys())].index
    for timeframe in rankings.keys():
        common_dates = common_dates.intersection(rankings[timeframe].index)
    common_sectors = rankings[min(rankings.keys())].columns
    for timeframe in rankings.keys():
        common_sectors = common_sectors.intersection(rankings[timeframe].columns)

    consensus_signals = pd.DataFrame(0, index=common_dates, columns=common_sectors)
    signal_strength = pd.DataFrame(0, index=common_dates, columns=common_sectors)
    top_threshold = 1

    for date in common_dates:
        for sector in common_sectors:
            votes = 0
            for timeframe in rankings.keys():
                if rankings[timeframe].loc[date, sector] <= top_threshold:
                    votes += 1
            signal_strength.loc[date, sector] = votes
            if votes >= consensus_threshold:
                consensus_signals.loc[date, sector] = 1

    return consensus_signals, signal_strength


def weighted_multi_timeframe_signals(momentum_rankings, weights, signal_threshold=0.75):
    print("Generating weighted multi-timeframe signals...")
    rankings = momentum_rankings['rankings']
    common_dates = rankings[min(rankings.keys())].index
    for timeframe in rankings.keys():
        common_dates = common_dates.intersection(rankings[timeframe].index)
    common_sectors = rankings[min(rankings.keys())].columns
    for timeframe in rankings.keys():
        common_sectors = common_sectors.intersection(rankings[timeframe].columns)

    weighted_scores = pd.DataFrame(0.0, index=common_dates, columns=common_sectors)
    weighted_signals = pd.DataFrame(0, index=common_dates, columns=common_sectors)
    max_rank = len(common_sectors)

    for date in common_dates:
        for sector in common_sectors:
            weighted_score = 0
            for timeframe in rankings.keys():
                score = (max_rank - rankings[timeframe].loc[date, sector] + 1) / max_rank
                weighted_score += weights[timeframe] * score
            weighted_scores.loc[date, sector] = weighted_score
            if weighted_score >= signal_threshold:
                weighted_signals.loc[date, sector] = 1

    return weighted_signals, weighted_scores


def avoid_late_movers(momentum_rankings, early_strength_signals, late_mover_penalty_days=2):
    print("Identifying late movers to avoid...")
    rankings_short = momentum_rankings['rankings'][min(momentum_rankings['rankings'].keys())]
    late_mover_filter = pd.DataFrame(1, index=rankings_short.index, columns=rankings_short.columns)

    for i, date in enumerate(rankings_short.index):
        if i < late_mover_penalty_days:
            continue
        for sector in rankings_short.columns:
            recent_rankings = rankings_short.iloc[i - late_mover_penalty_days:i][sector]
            current_ranking = rankings_short.loc[date, sector]
            was_weak = (recent_rankings > 6).mean() > 0.6
            now_strong = current_ranking <= 3
            early_strength_recently = False
            if i >= late_mover_penalty_days:
                recent_early_strength = early_strength_signals.iloc[i - late_mover_penalty_days:i]
                early_strength_recently = recent_early_strength.sum(axis=1).max() > 0
            if was_weak and now_strong and early_strength_recently:
                late_mover_filter.loc[date, sector] = 0

    return late_mover_filter


def calculate_market_volatility_filter(pc_scaled, window=20, volatility_threshold=3.0):
    print("Calculating market volatility filter...")
    pc1_volatility = pc_scaled['PC1'].rolling(window=window).std()
    vol_threshold = pc1_volatility.median() * volatility_threshold
    volatility_filter = pd.DataFrame(1, index=pc_scaled.index, columns=['filter'])
    volatility_filter.loc[pc1_volatility > vol_threshold, 'filter'] = 0
    return volatility_filter.squeeze()


def require_signal_confirmation(signals_df, confirmation_days=0):
    print(f"Requiring {confirmation_days} days of signal confirmation...")
    confirmed_signals = pd.DataFrame(0, index=signals_df.index, columns=signals_df.columns)
    for col in signals_df.columns:
        for i in range(confirmation_days, len(signals_df)):
            recent_signals = signals_df[col].iloc[i - confirmation_days:i]
            if recent_signals.sum() >= confirmation_days:
                confirmed_signals.iloc[i][col] = 1
    return confirmed_signals


# -----------------------------
# NEW: PC-based Sell Signal Generation
# -----------------------------
def calculate_rolling_r2_matrix(returns, pc_df, window=20):
    """
    Calculate rolling R² between each ETF and each PC
    """
    print(f"Calculating rolling R² matrix with {window}-day window...")

    # Align data
    common_dates = returns.index.intersection(pc_df.index)
    returns_aligned = returns.loc[common_dates]
    pc_aligned = pc_df.loc[common_dates]

    # Initialize results
    r2_results = {}

    for etf in returns_aligned.columns:
        r2_results[etf] = {}
        for pc in pc_aligned.columns:
            r2_series = []

            # Calculate rolling R² for this ETF-PC pair
            for i in range(window, len(common_dates) + 1):
                start_idx = i - window
                end_idx = i

                y = returns_aligned[etf].iloc[start_idx:end_idx]
                x = pc_aligned[pc].iloc[start_idx:end_idx]

                # Add constant term and fit OLS
                X = sm.add_constant(x)
                try:
                    model = sm.OLS(y, X).fit()
                    r2_series.append(model.rsquared)
                except:
                    r2_series.append(np.nan)

            # Pad with NaN for initial window
            r2_series = [np.nan] * (window - 1) + r2_series
            r2_results[etf][pc] = r2_series

    # Convert to nested DataFrames for easy access
    r2_dfs = {}
    for etf in r2_results.keys():
        r2_dfs[etf] = pd.DataFrame(r2_results[etf], index=common_dates)

    return r2_dfs

# Modified: generate_pc_based_sell_signals to include momentum decay, trailing stop, and time-based exit
def generate_pc_based_sell_signals(buy_signals, r2_dfs, weighted_scores=None,
                                   high_r2_threshold=0.1,
                                   pc1_min_r2_threshold=0.25,
                                   lookback_days=5,
                                   enable_momentum_decay=False,
                                   momentum_decay_threshold=0.8,
                                   enable_trailing_stop=False,
                                   trailing_stop_pct=0.07,
                                   enable_time_exit=False,
                                   max_hold_days=90,
                                   prices_df=None,
                                   ticker_to_sector=None):
    """
    Generate sell signals based on PC correlations, with optional momentum decay, trailing stop, and time-based exits
    """
    print(f"Generating PC-based sell signals...")
    print(f"  High R² threshold: {high_r2_threshold}")
    print(f"  PC1 minimum R² threshold: {pc1_min_r2_threshold}")
    print(f"  Lookback days: {lookback_days}")
    if enable_momentum_decay:
        print(f"  Enabling momentum decay exit with threshold: {momentum_decay_threshold}")
    if enable_trailing_stop:
        print(f"  Enabling trailing stop at {trailing_stop_pct * 100}%")
    if enable_time_exit:
        print(f"  Enabling time-based exit after {max_hold_days} days")

    # Initialize sell signals DataFrame
    sell_signals = pd.DataFrame(0, index=buy_signals.index, columns=buy_signals.columns)

    # Track positions (when we bought each ETF) and additional tracking for trailing
    positions = {etf: [] for etf in buy_signals.columns}  # List of buy dates for each ETF
    peak_prices = {etf: {} for etf in buy_signals.columns}  # Dict of buy_date: peak_price

    # Process each date
    for i, date in enumerate(buy_signals.index):
        # Process buy signals first - add to positions
        for etf in buy_signals.columns:
            if buy_signals.loc[date, etf] == 1:
                buy_price = prices_df.loc[date, etf]
                positions[etf].append(date)
                peak_prices[etf][date] = buy_price  # Initial peak is buy price

        # Check sell conditions for existing positions
        for etf in buy_signals.columns:
            if len(positions[etf]) > 0 and etf in r2_dfs:  # We have positions in this ETF
                r2_df = r2_dfs[etf]

                if date not in r2_df.index:
                    continue

                # Get current R² values
                current_r2_values = r2_df.loc[date]

                if current_r2_values.isna().all():
                    continue

                # Condition 1: Check if PC1 R² is below minimum threshold
                pc1_r2 = current_r2_values.get('PC1', 0)
                pc1_sell_signal = pc1_r2 < pc1_min_r2_threshold

                # Condition 2: Check if highest correlated PC has dropped below threshold
                high_r2_sell_signal = False

                # Look back to find the PC that had highest correlation during holding period
                for pos_date in positions[etf]:
                    # Get lookback window from position date
                    pos_idx = r2_df.index.get_loc(pos_date) if pos_date in r2_df.index else None
                    if pos_idx is None:
                        continue

                    # Define lookback window for determining highest correlated PC
                    lookback_start = max(0, pos_idx - lookback_days)
                    lookback_end = min(len(r2_df), pos_idx + lookback_days + 1)

                    # Get average R² during lookback period around position entry
                    if lookback_end > lookback_start:
                        lookback_r2 = r2_df.iloc[lookback_start:lookback_end].mean()
                        if not lookback_r2.isna().all():
                            highest_r2_pc = lookback_r2.idxmax()
                            highest_r2_value = lookback_r2.max()

                            # Check if this PC's current R² has dropped below threshold
                            current_highest_pc_r2 = current_r2_values.get(highest_r2_pc, 0)

                            # Sell if the highest correlated PC has dropped below threshold
                            if (highest_r2_value > high_r2_threshold and
                                    current_highest_pc_r2 < high_r2_threshold):
                                high_r2_sell_signal = True
                                break

                # NEW: Momentum decay condition
                momentum_decay_sell = False
                if enable_momentum_decay and weighted_scores is not None:
                    sector = ticker_to_sector.get(etf, None) if ticker_to_sector else None
                    if sector and date in weighted_scores.index and sector in weighted_scores.columns:
                        current_score = weighted_scores.loc[date, sector]
                        for pos_date in positions[etf]:
                            if pos_date in weighted_scores.index:
                                entry_score = weighted_scores.loc[pos_date, sector]
                                if current_score < momentum_decay_threshold * entry_score:
                                    momentum_decay_sell = True
                                    break

                # NEW: Trailing stop condition
                trailing_sell = False
                if enable_trailing_stop and prices_df is not None:
                    current_price = prices_df.loc[date, etf]
                    for pos_date in list(positions[etf]):  # Copy to avoid modification issues
                        peak = peak_prices[etf].get(pos_date, current_price)
                        peak = max(peak, current_price)  # Update peak
                        peak_prices[etf][pos_date] = peak
                        if (current_price / peak) - 1 < -trailing_stop_pct:
                            trailing_sell = True
                            break

                # NEW: Time-based exit
                time_sell = False
                if enable_time_exit:
                    for pos_date in positions[etf]:
                        days_held = (date - pos_date).days
                        if days_held > max_hold_days:
                            time_sell = True
                            break

                # Generate sell signal if any condition is met
                if pc1_sell_signal or high_r2_sell_signal or momentum_decay_sell or trailing_sell or time_sell:
                    sell_signals.loc[date, etf] = 1
                    # Remove all positions for this ETF (sell all)
                    positions[etf] = []
                    peak_prices[etf] = {}

                    # Optional: Log reason (commented for now)
                    # reasons = []
                    # if pc1_sell_signal: reasons.append("PC1 R² low")
                    # if high_r2_sell_signal: reasons.append("High R² drop")
                    # if momentum_decay_sell: reasons.append("Momentum decay")
                    # if trailing_sell: reasons.append("Trailing stop")
                    # if time_sell: reasons.append("Time exit")

    # Summary statistics
    total_sell_signals = sell_signals.sum().sum()
    etf_sell_counts = sell_signals.sum()
    print(f"  Total PC-based sell signals generated: {total_sell_signals}")
    print(f"  Sell signals by ETF:")
    for etf in etf_sell_counts[etf_sell_counts > 0].index:
        print(f"    {etf}: {etf_sell_counts[etf]} signals")

    return sell_signals
# Modified: comprehensive_sector_strategy to include all new parameters and logic
def comprehensive_sector_strategy(returns, sector_lists, test_start_date, lookbacks=[5, 14, 60],
                                  # PC-based sell signal parameters (existing)
                                  enable_pc_sell_signals=True,
                                  high_r2_threshold=0.1,
                                  pc1_min_r2_threshold=0.25,
                                  r2_window=20,
                                  pc_lookback_days=5,
                                  # NEW: Momentum decay
                                  enable_momentum_decay=True,
                                  momentum_decay_threshold=0.8,
                                  # NEW: Trailing stop
                                  enable_trailing_stop=True,
                                  trailing_stop_pct=0.07,
                                  # NEW: Time-based exit
                                  enable_time_exit=True,
                                  max_hold_days=90,
                                  # NEW: Regime filter
                                  enable_regime_filter=True,
                                  bull_threshold=0.5,
                                  bear_threshold=-0.5,
                                  # NEW: Vol-adjusted momentum
                                  enable_vol_adjusted_momentum=True,
                                  vol_adjust_window=20,
                                  # NEW: Shorts in bear (requires inverse ETFs; set to False by default)
                                  enable_shorts_in_bear=False,
                                  # NEW: Risk parity (applied in performance calc)
                                  enable_risk_parity=True,
                                  target_risk=0.01,
                                  # NEW: KMeans regimes
                                  enable_kmeans_regimes=False,
                                  n_clusters=3,
                                  # NEW: VIX filter
                                  enable_vix_filter=True,
                                  vix_threshold=30):
    print("\n" + "=" * 80)
    print("IMPLEMENTING COMPREHENSIVE SECTOR MOMENTUM STRATEGY")
    print("=" * 80)

    train_returns = returns.loc[:test_start_date]
    test_returns = returns.loc[test_start_date:]

    print("1/8: Calculating sector correlations and lead-lag relationships...")
    corr_data = calculate_sector_correlations_and_lags(train_returns, sector_lists)

    print("2/8: Calculating momentum rankings for multiple timeframes...")
    momentum_rankings_train = calculate_sector_momentum_rankings(train_returns, sector_lists, lookbacks,
                                                                 enable_vol_adjusted_momentum=enable_vol_adjusted_momentum,
                                                                 vol_window=vol_adjust_window)
    momentum_rankings_test = calculate_sector_momentum_rankings(test_returns, sector_lists, lookbacks,
                                                                enable_vol_adjusted_momentum=enable_vol_adjusted_momentum,
                                                                vol_window=vol_adjust_window)

    print("3/8: Identifying early strength sectors...")
    early_strength = identify_early_strength_sectors(momentum_rankings_test, corr_data,
                                                     early_strength_threshold=3, min_lead_correlation=0.25)

    print("4/8: Generating multi-timeframe consensus signals...")
    consensus_signals, signal_strength = generate_multi_timeframe_consensus_signals(momentum_rankings_test)

    print("5/8: Generating weighted timeframe signals...")
    weights = {f'{lb}d': w for lb, w in zip(lookbacks, [0.7, 0.2, 0.1][:len(lookbacks)])}
    weighted_signals, weighted_scores = weighted_multi_timeframe_signals(momentum_rankings_test, weights)

    print("6/8: Filtering out late movers...")
    late_mover_filter = avoid_late_movers(momentum_rankings_test, early_strength)

    print("7/8: Adding volatility filtering...")
    volatility_filter = calculate_market_volatility_filter(test_pc_scaled)

    # NEW: Regime detection
    regimes = pd.Series(0, index=test_pc_scaled.index)  # 0=neutral, 1=bull, -1=bear
    if enable_regime_filter:
        print("Applying market regime filter...")
        regimes[test_pc_scaled['PC1'] > bull_threshold] = 1
        regimes[test_pc_scaled['PC1'] < bear_threshold] = -1

    # NEW: KMeans regimes (overrides simple if enabled)
    if enable_kmeans_regimes:
        print(f"Applying KMeans regime clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(train_pc_scaled)
        test_clusters = kmeans.predict(test_pc_scaled)
        # Map clusters to regimes based on mean PC1
        cluster_means = [test_pc_scaled.iloc[test_clusters == c]['PC1'].mean() for c in range(n_clusters)]
        cluster_regime_map = {}
        for c in range(n_clusters):
            if cluster_means[c] > bull_threshold:
                cluster_regime_map[c] = 1
            elif cluster_means[c] < bear_threshold:
                cluster_regime_map[c] = -1
            else:
                cluster_regime_map[c] = 0
        regimes = pd.Series([cluster_regime_map[cl] for cl in test_clusters], index=test_pc_scaled.index)

    # NEW: VIX filter
    vix_filter = pd.Series(1, index=test_returns.index)  # 1=ok, 0=high vol skip
    if enable_vix_filter:
        print(f"Downloading VIX data and applying threshold: {vix_threshold}")
        vix_data = yf.download('^VIX', start=test_returns.index[0], end=test_returns.index[-1], auto_adjust=False)[
            ('Close', '^VIX')]
        vix_filter = pd.Series(1, index=vix_data.index)
        vix_filter[vix_data > vix_threshold] = 0
        vix_filter = vix_filter.reindex(test_returns.index, method='ffill')

        # Forward fill

    common_dates = (early_strength.index
                    .intersection(consensus_signals.index)
                    .intersection(weighted_signals.index)
                    .intersection(late_mover_filter.index)
                    .intersection(volatility_filter.index)
                    .intersection(regimes.index)
                    .intersection(vix_filter.index))
    common_sectors = (early_strength.columns
                      .intersection(consensus_signals.columns)
                      .intersection(weighted_signals.columns)
                      .intersection(late_mover_filter.columns))

    print("8/8: Applying confirmation requirements...")
    early_strength_confirmed = require_signal_confirmation(early_strength, confirmation_days=1)
    consensus_confirmed = require_signal_confirmation(consensus_signals, confirmation_days=1)
    weighted_confirmed = require_signal_confirmation(weighted_signals, confirmation_days=1)

    common_dates = (early_strength_confirmed.index
                    .intersection(consensus_confirmed.index)
                    .intersection(weighted_confirmed.index)
                    .intersection(late_mover_filter.index)
                    .intersection(volatility_filter.index)
                    .intersection(regimes.index)
                    .intersection(vix_filter.index))

    final_signals = pd.DataFrame(0, index=common_dates, columns=common_sectors)
    pc1_threshold = 0.8

    for date in final_signals.index:
        if date in test_pc_scaled.index and test_pc_scaled.loc[date, 'PC1'] <= pc1_threshold:
            continue
        if date not in volatility_filter.index or volatility_filter.loc[date] == 0:
            continue
        # NEW: Regime and VIX checks
        if date in regimes.index and regimes.loc[date] != 1:  # Only buy in bull
            if enable_shorts_in_bear and regimes.loc[date] == -1:
                # NEW: Short bottom sectors in bear (simple: signal=-1 for bottom 3)
                short_rankings = sorted(momentum_rankings_test['rankings'][min(lookbacks) + 'd'].loc[date], reverse=True)[:3]  # Bottom 3
                for sector, rank in short_rankings.items():
                    final_signals.loc[date, sector] = -1
            continue
        if date in vix_filter.index and vix_filter.loc[date] == 0:
            continue
        for sector in common_sectors:
            if sector not in final_signals.columns:
                continue
            early_confirmed = early_strength_confirmed.loc[
                date, sector] if date in early_strength_confirmed.index else 0
            consensus_confirmed_sig = consensus_confirmed.loc[date, sector] if date in consensus_confirmed.index else 0
            weighted_confirmed_sig = weighted_confirmed.loc[date, sector] if date in weighted_confirmed.index else 0
            not_late_mover = late_mover_filter.loc[date, sector] if date in late_mover_filter.index else 1
            has_consensus = consensus_confirmed_sig == 1
            has_strength = (early_confirmed == 1) or (weighted_confirmed_sig == 1)
            passes_filters = not_late_mover == 1
            if has_consensus and has_strength and passes_filters:
                final_signals.loc[date, sector] = 1

    sector_to_ticker = {sector: tickers[0] for sector, tickers in sector_lists.items()}
    etf_signals = pd.DataFrame(0, index=final_signals.index,
                               columns=[sector_to_ticker[s] for s in final_signals.columns])
    for date in final_signals.index:
        for sector in final_signals.columns:
            signal = final_signals.loc[date, sector]
            if signal != 0:
                etf_signals.loc[date, sector_to_ticker[sector]] = signal  # Support -1 for shorts

    # NEW: Generate PC-based sell signals with new features
    pc_sell_signals = None

    if enable_pc_sell_signals:
        print("\n" + "=" * 60)
        print("GENERATING PC-BASED SELL SIGNALS")
        print("=" * 60)

        # Calculate rolling R² matrix for test period
        r2_dfs = calculate_rolling_r2_matrix(test_returns, test_pc_df, window=r2_window)
        ticker_to_sector = {tickers[0]: sector for sector, tickers in sector_lists.items()}
        # Generate PC-based sell signals with new params
        pc_sell_signals = generate_pc_based_sell_signals(
            etf_signals, r2_dfs,
            weighted_scores=weighted_scores,
            high_r2_threshold=high_r2_threshold,
            pc1_min_r2_threshold=pc1_min_r2_threshold,
            lookback_days=pc_lookback_days,
            enable_momentum_decay=enable_momentum_decay,
            momentum_decay_threshold=momentum_decay_threshold,
            enable_trailing_stop=enable_trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            enable_time_exit=enable_time_exit,
            max_hold_days=max_hold_days,
            prices_df=data,
            ticker_to_sector=ticker_to_sector  # NEW: Pass it
        )

    strategy_details = {
        'correlation_data': corr_data,
        'momentum_rankings': momentum_rankings_test,
        'early_strength_signals': early_strength,
        'consensus_signals': consensus_signals,
        'signal_strength': signal_strength,
        'weighted_signals': weighted_signals,
        'weighted_scores': weighted_scores,
        'late_mover_filter': late_mover_filter,
        'sector_signals': final_signals,
        'etf_signals': etf_signals,
        'pc_sell_signals': pc_sell_signals,
        'r2_dfs': r2_dfs if enable_pc_sell_signals else None,
        'regimes': regimes  # NEW: For analysis
    }

    return etf_signals, pc_sell_signals, strategy_details


def plot_comprehensive_strategy_analysis(strategy_details, sector_lists):
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    signal_strength = strategy_details['signal_strength']
    im1 = axes[0].imshow(signal_strength.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('Multi-Timeframe Signal Strength')
    axes[0].set_xlabel('Date Index')
    axes[0].set_ylabel('Sectors')
    axes[0].set_yticks(range(len(signal_strength.columns)))
    axes[0].set_yticklabels(signal_strength.columns, rotation=0, fontsize=8)
    plt.colorbar(im1, ax=axes[0])

    weighted_scores = strategy_details['weighted_scores']
    im2 = axes[1].imshow(weighted_scores.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('Weighted Momentum Scores')
    axes[1].set_xlabel('Date Index')
    axes[1].set_ylabel('Sectors')
    axes[1].set_yticks(range(len(weighted_scores.columns)))
    axes[1].set_yticklabels(weighted_scores.columns, rotation=0, fontsize=8)
    plt.colorbar(im2, ax=axes[1])

    early_strength = strategy_details['early_strength_signals']
    signal_counts = early_strength.sum(axis=1)
    axes[2].plot(range(len(signal_counts)), signal_counts, 'g-', linewidth=2)
    axes[2].set_title('Early Strength Signals Over Time')
    axes[2].set_xlabel('Date Index')
    axes[2].set_ylabel('Number of Sectors')
    axes[2].grid(True, alpha=0.3)

    etf_signals = strategy_details['etf_signals']
    signal_totals = etf_signals.sum(axis=0).sort_values(ascending=True)
    signal_totals.plot(kind='barh', ax=axes[3], color='blue', alpha=0.7)
    axes[3].set_title('Total Buy Signals by ETF')
    axes[3].set_xlabel('Number of Signals')
    axes[3].grid(True, alpha=0.3)

    late_mover_filter = strategy_details['late_mover_filter']
    sectors_avoided = (late_mover_filter == 0).sum(axis=1)
    axes[4].plot(range(len(sectors_avoided)), sectors_avoided, 'r-', linewidth=2)
    axes[4].set_title('Sectors Avoided (Late Mover Filter)')
    axes[4].set_xlabel('Date Index')
    axes[4].set_ylabel('Number of Sectors Avoided')
    axes[4].grid(True, alpha=0.3)

    # NEW: Plot PC-based sell signals
    if strategy_details['pc_sell_signals'] is not None:
        pc_sell_signals = strategy_details['pc_sell_signals']
        sell_signal_totals = pc_sell_signals.sum(axis=0).sort_values(ascending=True)
        sell_signal_totals.plot(kind='barh', ax=axes[5], color='red', alpha=0.7)
        axes[5].set_title('Total PC-based Sell Signals by ETF')
        axes[5].set_xlabel('Number of Sell Signals')
        axes[5].grid(True, alpha=0.3)
    else:
        final_signals = strategy_details['sector_signals']
        daily_signals = final_signals.sum(axis=1)
        axes[5].hist(daily_signals, bins=range(len(final_signals.columns) + 2),
                     alpha=0.7, color='green', edgecolor='black')
        axes[5].set_title('Distribution of Daily Signal Count')
        axes[5].set_xlabel('Number of Sectors Signaled Per Day')
        axes[5].set_ylabel('Frequency')
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Modified: calculate_trading_performance to support risk parity and shorts
def calculate_trading_performance(buy_signals, sell_signals, test_returns, prices_df, train_std=None, enable_risk_parity=False, target_risk=0.01):
    common_dates = (buy_signals.index
                    .intersection(sell_signals.index)
                    .intersection(test_returns.index)
                    .intersection(prices_df.index))

    if len(common_dates) == 0:
        print("No common dates found between all DataFrames")
        return pd.DataFrame(), {}

    buy_signals_aligned = buy_signals.loc[common_dates]
    sell_signals_aligned = sell_signals.loc[common_dates]
    test_returns_aligned = test_returns.loc[common_dates]
    test_prices_aligned = prices_df.loc[common_dates]

    common_etfs = (buy_signals_aligned.columns
                   .intersection(sell_signals_aligned.columns)
                   .intersection(test_returns_aligned.columns)
                   .intersection(test_prices_aligned.columns))

    if len(common_etfs) == 0:
        print("No common ETFs found between all DataFrames")
        return pd.DataFrame(), {}

    buy_signals_aligned = buy_signals_aligned[common_etfs]
    sell_signals_aligned = sell_signals_aligned[common_etfs]
    test_returns_aligned = test_returns_aligned[common_etfs]
    test_prices_aligned = test_prices_aligned[common_etfs]

    trades = []
    positions = {etf: None for etf in common_etfs}
    total_capital_deployed = 0

    for date in common_dates:
        for etf in common_etfs:
            signal = buy_signals_aligned.loc[date, etf]
            if signal != 0 and positions[etf] is None:
                buy_price = test_prices_aligned.loc[date, etf]
                # NEW: Risk parity sizing
                shares = 1
                if enable_risk_parity and train_std is not None and etf in train_std.index:
                    annualized_vol = train_std[etf] * np.sqrt(252)  # Assuming daily std
                    shares = target_risk / annualized_vol if annualized_vol > 0 else 1
                direction = 1 if signal > 0 else -1  # For shorts
                positions[etf] = {
                    'buy_date': date,
                    'buy_price': buy_price,
                    'shares': shares * direction,  # Negative for shorts
                    'direction': direction
                }
                total_capital_deployed += abs(buy_price * shares)
            elif sell_signals_aligned.loc[date, etf] == 1 and positions[etf] is not None:
                sell_price = test_prices_aligned.loc[date, etf]
                position = positions[etf]
                # For shorts, invert return calculation
                price_change = (sell_price - position['buy_price']) * position['direction']
                trade_return = price_change / position['buy_price']
                trade_profit = price_change * abs(position['shares'])
                days_held = (date - position['buy_date']).days
                trades.append({
                    'etf': etf,
                    'buy_date': position['buy_date'],
                    'sell_date': date,
                    'buy_price': position['buy_price'],
                    'sell_price': sell_price,
                    'return_pct': trade_return,
                    'profit_dollar': trade_profit,
                    'days_held': days_held,
                    'annualized_return': (trade_return + 1) ** (365 / days_held) - 1 if days_held > 0 else 0,
                    'direction': 'Long' if position['direction'] > 0 else 'Short'
                })
                positions[etf] = None

    final_date = common_dates[-1]
    for etf, position in positions.items():
        if position is not None:
            final_price = test_prices_aligned.loc[final_date, etf]
            price_change = (final_price - position['buy_price']) * position['direction']
            trade_return = price_change / position['buy_price']
            trade_profit = price_change * abs(position['shares'])
            days_held = (final_date - position['buy_date']).days
            trades.append({
                'etf': etf,
                'buy_date': position['buy_date'],
                'sell_date': final_date,
                'buy_price': position['buy_price'],
                'sell_price': final_price,
                'return_pct': trade_return,
                'profit_dollar': trade_profit,
                'days_held': days_held,
                'annualized_return': (trade_return + 1) ** (365 / days_held) - 1 if days_held > 0 else 0,
                'status': 'OPEN',
                'direction': 'Long' if position['direction'] > 0 else 'Short'
            })

    if not trades:
        print("No completed trades to analyze")
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    profitable_trades = (trades_df['return_pct'] > 0).sum()
    total_profit = trades_df['profit_dollar'].sum()
    total_return_pct = trades_df['return_pct'].mean()
    win_rate = profitable_trades / total_trades
    avg_profit_per_trade = total_profit / total_trades
    avg_return_per_trade = total_return_pct
    median_return = trades_df['return_pct'].median()
    best_trade = trades_df['return_pct'].max()
    worst_trade = trades_df['return_pct'].min()
    avg_days_held = trades_df['days_held'].mean()
    return_std = trades_df['return_pct'].std()
    sharpe_ratio = avg_return_per_trade / return_std if return_std > 0 else 0
    gross_profit = trades_df[trades_df['profit_dollar'] > 0]['profit_dollar'].sum()
    gross_loss = abs(trades_df[trades_df['profit_dollar'] < 0]['profit_dollar'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    cumulative_returns = (1 + trades_df['return_pct']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    performance_metrics = {
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'total_profit_dollar': total_profit,
        'total_return_pct': total_return_pct,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_return_per_trade': avg_return_per_trade,
        'median_return': median_return,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_days_held': avg_days_held,
        'return_std': return_std,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'total_capital_deployed': total_capital_deployed,
        'annualized_return': trades_df['annualized_return'].mean()
    }

    return trades_df, performance_metrics


def print_performance_report(trades_df, performance_metrics):
    print("\n" + "=" * 80)
    print("                        TRADING PERFORMANCE REPORT")
    print("=" * 80)
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {performance_metrics['total_trades']}")
    print(f"   Profitable Trades: {performance_metrics['profitable_trades']}")
    print(f"   Win Rate: {performance_metrics['win_rate']:.2%}")
    print(f"   Total Profit: ${performance_metrics['total_profit_dollar']:,.2f}")
    print(f"   Average Return per Trade: {performance_metrics['avg_return_per_trade']:.2%}")
    print(f"   Median Return per Trade: {performance_metrics['median_return']:.2%}")
    print(f"\n📈 RISK METRICS:")
    print(f"   Return Standard Deviation: {performance_metrics['return_std']:.2%}")
    print(f"   Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
    print(f"   Maximum Drawdown: {performance_metrics['max_drawdown']:.2%}")
    print(f"   Profit Factor: {performance_metrics['profit_factor']:.2f}")
    print(f"\n⏱️  TRADE DETAILS:")
    print(f"   Average Days Held: {performance_metrics['avg_days_held']:.1f}")
    print(f"   Best Trade: {performance_metrics['best_trade']:.2%}")
    print(f"   Worst Trade: {performance_metrics['worst_trade']:.2%}")
    print(f"   Annualized Return: {performance_metrics['annualized_return']:.2%}")
    print(f"\n💰 CAPITAL METRICS:")
    print(f"   Total Capital Deployed: ${performance_metrics['total_capital_deployed']:,.2f}")
    print(f"   Average Profit per Trade: ${performance_metrics['avg_profit_per_trade']:,.2f}")
    if len(trades_df) > 0:
        print(f"\n📋 PERFORMANCE BY ETF:")
        etf_performance = trades_df.groupby('etf').agg({
            'return_pct': ['count', 'mean', 'sum'],
            'profit_dollar': 'sum',
            'days_held': 'mean'
        }).round(4)
        etf_performance.columns = ['Trades', 'Avg_Return', 'Total_Return', 'Total_Profit', 'Avg_Days']
        etf_performance['Win_Rate'] = trades_df.groupby('etf')['return_pct'].apply(lambda x: (x > 0).mean())
        for etf in etf_performance.index:
            trades_count = int(etf_performance.loc[etf, 'Trades'])
            avg_ret = etf_performance.loc[etf, 'Avg_Return']
            total_profit = etf_performance.loc[etf, 'Total_Profit']
            win_rate = etf_performance.loc[etf, 'Win_Rate']
            avg_days = etf_performance.loc[etf, 'Avg_Days']
            print(
                f"   {etf}: {trades_count} trades, {avg_ret:.2%} avg return, ${total_profit:,.2f} profit, {win_rate:.2%} win rate, {avg_days:.1f} days avg")
    print("=" * 80)


def plot_trading_results(buy_signals, sell_signals, trades_df, test_returns, test_prices):
    common_dates = (buy_signals.index
                    .intersection(sell_signals.index)
                    .intersection(test_returns.index)
                    .intersection(test_prices.index))

    if len(common_dates) == 0:
        print("No common dates for plotting")
        return

    buy_signals_plot = buy_signals.loc[common_dates]
    sell_signals_plot = sell_signals.loc[common_dates]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    for i, etf in enumerate(buy_signals_plot.columns):
        buy_dates = buy_signals_plot.index[buy_signals_plot[etf] == 1]
        sell_dates = sell_signals_plot.index[sell_signals_plot[etf] == 1]
        if len(buy_dates) > 0:
            ax1.scatter(buy_dates, [i] * len(buy_dates),
                        marker='^', s=60, color='green', alpha=0.7)
        if len(sell_dates) > 0:
            ax1.scatter(sell_dates, [i] * len(sell_dates),
                        marker='v', s=60, color='red', alpha=0.7)
    ax1.set_title("Buy (↑) and Sell (↓) Signals Timeline")
    ax1.set_ylabel("ETF")
    ax1.set_yticks(range(len(buy_signals_plot.columns)))
    ax1.set_yticklabels(buy_signals_plot.columns)


def plot_etf_signals_over_time(buy_signals, sell_signals, prices_df, sector_lists, etfs_per_page=4):
    print("Plotting buy and sell signals over time for each ETF...")
    common_dates = (buy_signals.index
                    .intersection(sell_signals.index)
                    .intersection(prices_df.index))

    if len(common_dates) == 0:
        print("No common dates for plotting")
        return

    buy_signals_plot = buy_signals.loc[common_dates]
    sell_signals_plot = sell_signals.loc[common_dates]
    prices_plot = prices_df.loc[common_dates]
    ticker_to_sector = {ticker: sector for sector, tickers in sector_lists.items() for ticker in tickers}
    etfs = buy_signals_plot.columns
    num_etfs = len(etfs)
    num_pages = (num_etfs + etfs_per_page - 1) // etfs_per_page

    for page in range(num_pages):
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        start_idx = page * etfs_per_page
        end_idx = min(start_idx + etfs_per_page, num_etfs)
        page_etfs = etfs[start_idx:end_idx]

        for idx, etf in enumerate(page_etfs):
            ax = axes[idx]
            sector_name = ticker_to_sector.get(etf, "Unknown Sector")
            ax.plot(prices_plot.index, prices_plot[etf], 'b-', label=f"{etf} Price", linewidth=1.5)
            buy_dates = buy_signals_plot.index[buy_signals_plot[etf] == 1]
            buy_prices = prices_plot.loc[buy_dates, etf]
            ax.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy Signal')
            sell_dates = sell_signals_plot.index[sell_signals_plot[etf] == 1]
            sell_prices = prices_plot.loc[sell_dates, etf]
            ax.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell Signal (PC-based)')
            ax.set_title(f"{sector_name} ({etf}) - Price and Trading Signals")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.get_xticklabels(), rotation=45)
        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.show()
        print(f"Generated plot page {page + 1}/{num_pages}")


def plot_pc_r2_analysis(r2_dfs, buy_signals, sell_signals, sector_lists, etfs_to_plot=None):
    """
    Plot R² evolution for ETFs with buy/sell signals overlaid
    """
    if r2_dfs is None:
        print("No R² data available for plotting")
        return

    ticker_to_sector = {ticker: sector for sector, tickers in sector_lists.items() for ticker in tickers}

    # Select ETFs to plot
    if etfs_to_plot is None:
        # Plot ETFs with most signals
        signal_counts = (buy_signals.sum() + sell_signals.sum()).sort_values(ascending=False)
        etfs_to_plot = signal_counts.head(6).index.tolist()

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i, etf in enumerate(etfs_to_plot):
        if i >= len(axes) or etf not in r2_dfs:
            continue

        ax = axes[i]
        r2_df = r2_dfs[etf]
        sector_name = ticker_to_sector.get(etf, "Unknown Sector")

        # Plot R² for each PC
        for pc in r2_df.columns:
            ax.plot(r2_df.index, r2_df[pc], label=pc, linewidth=1.5, alpha=0.7)

        # Overlay buy signals
        if etf in buy_signals.columns:
            buy_dates = buy_signals.index[buy_signals[etf] == 1]
            for buy_date in buy_dates:
                if buy_date in r2_df.index:
                    ax.axvline(x=buy_date, color='green', alpha=0.6, linestyle='--', linewidth=2)

        # Overlay sell signals
        if etf in sell_signals.columns:
            sell_dates = sell_signals.index[sell_signals[etf] == 1]
            for sell_date in sell_dates:
                if sell_date in r2_df.index:
                    ax.axvline(x=sell_date, color='red', alpha=0.6, linestyle='--', linewidth=2)

        ax.set_title(f"{sector_name} ({etf}) - R² with PCs Over Time")
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45)

    # Add legend for buy/sell lines
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', linestyle='--', linewidth=2),
                    Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
    fig.legend(custom_lines, ['Buy Signal', 'Sell Signal'],
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.show()


# -----------------------------
# Run the strategy and generate plots
# -----------------------------
test_start_date = split_date
lookbacks = [5, 10, 60]  # Adjustable lookback periods

# PC-based sell signal parameters (easily adjustable)
HIGH_R2_THRESHOLD = 0.1  # Sell when highest correlated PC R² drops below this
PC1_MIN_R2_THRESHOLD = 0.25  # Sell when PC1 R² drops below this
R2_WINDOW = 20  # Rolling window for R² calculation
PC_LOOKBACK_DAYS = 5  # Days to look back for determining highest correlated PC

print(f"\n{'=' * 80}")
print("PC-BASED SELL SIGNAL PARAMETERS:")
print(f"{'=' * 80}")
print(f"High R² Threshold: {HIGH_R2_THRESHOLD}")
print(f"PC1 Minimum R² Threshold: {PC1_MIN_R2_THRESHOLD}")
print(f"R² Rolling Window: {R2_WINDOW} days")
print(f"PC Lookback Days: {PC_LOOKBACK_DAYS}")
print(f"{'=' * 80}")

print("\nRunning comprehensive sector momentum strategy with PC-based sell signals...")
buy_signals, pc_sell_signals, strategy_details = comprehensive_sector_strategy(
    returns, sector_lists, test_start_date, lookbacks,
    # PC-based sell signal parameters
    enable_pc_sell_signals=True,
    high_r2_threshold=HIGH_R2_THRESHOLD,
    pc1_min_r2_threshold=PC1_MIN_R2_THRESHOLD,
    r2_window=R2_WINDOW,
    pc_lookback_days=PC_LOOKBACK_DAYS
)

# Use PC-based sell signals if available, otherwise fall back to original logic
if pc_sell_signals is not None:
    final_sell_signals = pc_sell_signals
    print(f"\nUsing PC-based sell signals: {pc_sell_signals.sum().sum()} total sell signals")
else:
    # Original sell signal logic as fallback
    final_sell_signals = pd.DataFrame(0, index=buy_signals.index, columns=buy_signals.columns)
    for date_idx in range(1, len(buy_signals)):
        prev_signals = buy_signals.iloc[date_idx - 1]
        curr_signals = buy_signals.iloc[date_idx]
        for etf in buy_signals.columns:
            if prev_signals[etf] == 1 and curr_signals[etf] == 0:
                final_sell_signals.iloc[date_idx][etf] = 1
    print(f"\nUsing traditional sell signals: {final_sell_signals.sum().sum()} total sell signals")

# Calculate and display performance
trades_df, performance_metrics = calculate_trading_performance(
    buy_signals, final_sell_signals, test_returns, data, train_std=train_std, enable_risk_parity=True, target_risk=0.01
)

print_performance_report(trades_df, performance_metrics)

# Generate all plots
print("\n" + "=" * 80)
print("GENERATING COMPREHENSIVE ANALYSIS PLOTS")
print("=" * 80)

plot_comprehensive_strategy_analysis(strategy_details, sector_lists)
plot_trading_results(buy_signals, final_sell_signals, trades_df, test_returns, data)
plot_etf_signals_over_time(buy_signals, final_sell_signals, data, sector_lists, etfs_per_page=4)

# NEW: Plot PC R² analysis
if strategy_details['r2_dfs'] is not None:
    print("\nGenerating PC R² analysis plots...")
    plot_pc_r2_analysis(strategy_details['r2_dfs'], buy_signals, final_sell_signals, sector_lists)

print("\n" + "=" * 80)
print("STRATEGY EXECUTION COMPLETE!")
print("=" * 80)
print(f"✅ Buy signals: {buy_signals.sum().sum()}")
print(f"✅ Sell signals: {final_sell_signals.sum().sum()}")
print(f"✅ Total trades: {len(trades_df)}")
print(f"✅ Win rate: {performance_metrics.get('win_rate', 0):.2%}")
print(f"✅ Average return per trade: {performance_metrics.get('avg_return_per_trade', 0):.2%}")
print("=" * 80)

print("Entry Timing Quality:")
trades_df['max_drawdown_during_hold'] = 0.0
for idx, trade in trades_df.iterrows():
    hold_prices = data.loc[trade['buy_date']:trade['sell_date'], trade['etf']]
    cum_returns = (hold_prices / trade['buy_price']) - 1
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / (running_max + 1)  # +1 to avoid div by zero
    trades_df.at[idx, 'max_drawdown_during_hold'] = drawdown.min()

print("Avg max drawdown during holds:", trades_df['max_drawdown_during_hold'].mean())