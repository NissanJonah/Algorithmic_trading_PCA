import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.dates as mdates
from sklearn.cluster import KMeans

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
data = yf.download(all_etfs, period="5y", auto_adjust=True)
data = data['Close']
print(f"Using {len(data.columns)} ETFs after filtering.")

# Compute daily returns for ETFs
returns = data.pct_change().dropna()

# Standardize ETF returns for PCA input
returns_std = (returns - returns.mean()) / returns.std()

# -----------------------------
# 3. PCA
# -----------------------------
pca = PCA()
Y = pca.fit_transform(returns_std)  # Scores matrix: rows = days, cols = PCs
V = pca.components_.T               # Loadings matrix: rows = ETFs, cols = PCs

# Create DataFrame for PC scores for convenience
pc_df = pd.DataFrame(Y, index=returns_std.index,
                     columns=[f"PC{i+1}" for i in range(Y.shape[1])])

# -----------------------------
# 4. Explained variance & loadings check
# -----------------------------
explained_var = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained_var[0]:.2%}")

loadings_signs = np.sign(V[:, 0])
if (loadings_signs == loadings_signs[0]).all():
    print("PC1 likely represents overall market movement (all loadings have the same sign).")
else:
    print("PC1 may not represent overall market movement (mixed loading signs).")

# -----------------------------
# 5. Correlation (R²) of ETFs with PCs - print top 3 PCs per ETF
# -----------------------------
print("\nCorrelation (R^2) between PCs and sector ETFs:")
print("\nTop 3 PCs by R² for each ETF:")

for etf in returns.columns:
    y = returns[etf].loc[pc_df.index]
    r2_list = []
    for pc in pc_df.columns:
        X = sm.add_constant(pc_df[pc])
        model = sm.OLS(y, X).fit()
        r2_list.append((pc, model.rsquared))
    r2_list.sort(key=lambda x: x[1], reverse=True)
    print(f"\nETF {etf}:")
    for pc, r2 in r2_list[:3]:
        print(f"  {pc} with R² = {r2:.2f}")

print("\nSummary:")
print(f"PC1 explains {explained_var[0]:.2%} of variance and " +
      ("likely" if (loadings_signs == loadings_signs[0]).all() else "may not") +
      " represent overall market movement.")

# -----------------------------
# 6. Function: Plot PCA R² over time for selected ETFs
# -----------------------------
def plot_pca_r2_over_time(returns, sector_lists, window_size=63, etfs_to_plot=None, num_graphs=11):
    """
    Rolling-window R² over time for ETFs vs PCs.
    Each ETF gets its own graph showing all PCs as lines.
    Titles show sector name + ETF ticker.
    """
    ticker_to_sector = {}
    for sector, tickers in sector_lists.items():
        for ticker in tickers:
            ticker_to_sector[ticker] = sector

    total_windows = len(returns) // window_size
    if etfs_to_plot is None:
        etfs_to_plot = returns.columns[:num_graphs]
    else:
        etfs_to_plot = etfs_to_plot[:num_graphs]

    r2_over_time = {etf: {f'PC{i + 1}': [] for i in range(len(returns.columns))} for etf in etfs_to_plot}
    window_dates = []

    for w in range(total_windows):
        start = w * window_size
        end = start + window_size
        window_returns = returns.iloc[start:end]
        window_returns_std = (window_returns - window_returns.mean()) / window_returns.std()

        pca = PCA()
        Y = pca.fit_transform(window_returns_std)
        pc_df = pd.DataFrame(Y, index=window_returns_std.index,
                             columns=[f'PC{i + 1}' for i in range(Y.shape[1])])

        mid_date = window_returns_std.index[window_size // 2]
        window_dates.append(mid_date)

        for etf in etfs_to_plot:
            y = window_returns[etf].loc[pc_df.index]
            for pc in pc_df.columns:
                X = sm.add_constant(pc_df[pc])
                model = sm.OLS(y, X).fit()
                r2_over_time[etf][pc].append(model.rsquared)

    for etf in etfs_to_plot:
        sector_name = ticker_to_sector.get(etf, "Unknown Sector")
        plt.figure(figsize=(12, 5))
        for pc in r2_over_time[etf]:
            plt.plot(window_dates, r2_over_time[etf][pc], marker='o', label=pc)
        plt.title(f"{sector_name} ({etf}) — R² over time vs PCs")
        plt.xlabel("Date")
        plt.ylabel("R²")
        plt.ylim(0, 1)
        plt.legend(title="PC")
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# -----------------------------
# 7. KMeans clustering of PCA scores
# -----------------------------
def cluster_pca_scores(pc_df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pc_df)
    return cluster_labels, kmeans

# -----------------------------
# 8. Identify dominant PCs in clusters
# -----------------------------
def identify_dominant_pcs(pc_df, cluster_labels, top_n=1):
    df = pc_df.copy()
    df['Cluster'] = cluster_labels
    dominant_pcs = {}
    for cluster in np.unique(cluster_labels):
        mean_scores = df[df['Cluster'] == cluster].mean().drop('Cluster')
        top_pcs = mean_scores.abs().sort_values(ascending=False).head(top_n).index.tolist()
        dominant_pcs[cluster] = top_pcs
    return dominant_pcs

# -----------------------------
# 9. Compute rolling R² for each sector and PC
# -----------------------------
def compute_rolling_r2(returns, pc_df, sector_lists, window=63):
    r2_dict = {}
    pc_cols = pc_df.columns
    returns_aligned = returns.loc[pc_df.index]

    for sector, etfs in sector_lists.items():
        sector_r2 = pd.DataFrame(index=pc_df.index, columns=pc_cols)
        for pc in pc_cols:
            r2_series = []
            for end_ix in range(window, len(pc_df) + 1):
                start_ix = end_ix - window
                y = returns_aligned[etfs].iloc[start_ix:end_ix].mean(axis=1)  # average sector return
                X = pc_df[pc].iloc[start_ix:end_ix]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                r2_series.append(model.rsquared)
            r2_series = [np.nan] * (window - 1) + r2_series
            sector_r2[pc] = r2_series
        r2_dict[sector] = sector_r2
    return r2_dict

# -----------------------------
# 10. Plot PCA clusters in 2D & 3D
# -----------------------------
def plot_pca_clusters(pc_df, cluster_labels, kmeans_model=None, pc_x='PC1', pc_y='PC2'):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pc_df[pc_x], pc_df[pc_y], c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    if kmeans_model is not None:
        centers = kmeans_model.cluster_centers_
        pc_cols = pc_df.columns.tolist()
        x_idx = pc_cols.index(pc_x)
        y_idx = pc_cols.index(pc_y)
        plt.scatter(centers[:, x_idx], centers[:, y_idx], marker='X', s=200, c='black', label='Cluster Centers')
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.title(f"PCA Scores Colored by K-Means Clusters")
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pca_clusters_3d(pc_df, cluster_labels, kmeans_model=None, pcs=['PC1', 'PC2', 'PC3']):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pc_df[pcs[0]], pc_df[pcs[1]], pc_df[pcs[2]],
                    c=cluster_labels, cmap='tab10', alpha=0.6, s=30)
    if kmeans_model is not None:
        centers = kmeans_model.cluster_centers_
        pc_cols = pc_df.columns.tolist()
        x_idx = pc_cols.index(pcs[0])
        y_idx = pc_cols.index(pcs[1])
        z_idx = pc_cols.index(pcs[2])
        ax.scatter(centers[:, x_idx], centers[:, y_idx], centers[:, z_idx],
                   marker='X', s=200, c='black', label='Cluster Centers')
    ax.set_xlabel(pcs[0])
    ax.set_ylabel(pcs[1])
    ax.set_zlabel(pcs[2])
    ax.set_title("3D PCA Scores Colored by K-Means Clusters")
    cbar = plt.colorbar(sc, pad=0.1, shrink=0.6)
    cbar.set_label('Cluster')
    plt.legend()
    plt.show()

# -----------------------------
# 11. Interpret clusters: mean PC scores per cluster
# -----------------------------
def interpret_clusters(pc_df, cluster_labels):
    df = pc_df.copy()
    df['Cluster'] = cluster_labels
    summary = df.groupby('Cluster').mean()
    print("Mean PC scores per cluster:")
    print(summary)
    print("\nCluster interpretations:")
    for cluster, row in summary.iterrows():
        dominant_pc = row.abs().idxmax()
        sign = 'positive' if row[dominant_pc] > 0 else 'negative'
        print(f"Cluster {cluster} is dominated by {sign} {dominant_pc} (mean = {row[dominant_pc]:.3f})")
    return summary

# -----------------------------
# 12. Generate buy signals based on cluster & R² profiles
# -----------------------------
def generate_cluster_based_signals(
        current_date, pc_df, cluster_labels, r2_dict, cluster_means,
        positive_threshold=0.1, negative_threshold=0.05, r2_threshold=0.3
):
    if current_date not in pc_df.index:
        raise ValueError(f"Date {current_date} not in PC scores index")
    day_idx = pc_df.index.get_loc(current_date)
    current_cluster = cluster_labels[day_idx]
    pc_means = cluster_means.loc[current_cluster]

    good_pcs = pc_means[pc_means > positive_threshold].index.tolist()
    bad_pcs = pc_means[pc_means.abs() < negative_threshold].index.tolist()

    sector_scores = {}
    buy_signals = {}

    for sector, r2_df in r2_dict.items():
        if current_date not in r2_df.index:
            sector_scores[sector] = np.nan
            buy_signals[sector] = 0
            continue
        today_r2 = r2_df.loc[current_date]
        good_corr = today_r2[good_pcs].fillna(0).clip(lower=0).sum()
        bad_corr = today_r2[bad_pcs].fillna(0).clip(lower=0).sum()
        score = good_corr - bad_corr
        sector_scores[sector] = score
        buy = (score > 0) and any(today_r2[good_pcs].fillna(0) > r2_threshold)
        buy_signals[sector] = int(buy)

    scores_array = np.array([v if not np.isnan(v) else -np.inf for v in sector_scores.values()])
    min_score = scores_array.min()
    if min_score < 0:
        scores_array = scores_array - min_score
    max_score = scores_array.max()
    if max_score > 0:
        scores_normalized = scores_array / max_score
    else:
        scores_normalized = np.zeros_like(scores_array)

    # Map back to dict with sector keys, normalized from 0 to 1
    sector_scores_pct = {sec: score for sec, score in zip(sector_scores.keys(), scores_normalized)}

    # Now print results with normalized scores (0 to 1)
    sorted_sectors = sorted(sector_scores_pct.items(), key=lambda x: x[1], reverse=True)

    print(f"Buy signals for {current_date.date()} (Cluster {current_cluster}):")
    print(f"Good PCs: {good_pcs}, Bad PCs: {bad_pcs}")
    print("Sector Scores (normalized 0 to 1):")
    for sec, norm_score in sorted_sectors:
        buy_str = "BUY" if buy_signals[sec] else "----"
        print(f"{sec:25} : {norm_score:.3f}   {buy_str}")

    return buy_signals, sector_scores_pct

# -----------------------------
# 13. New: Plot cluster regimes ordered by PC1 vs normalized returns
# -----------------------------
import statsmodels.api as sm

def plot_clusters_vs_returns(pc_df, cluster_labels, cluster_means, returns, market_etf, market_returns=None, zoom_dates=None):
    if market_returns is None:
        if market_etf in returns.columns:
            market_returns = returns[market_etf].loc[pc_df.index]
        else:
            market_returns = returns.loc[pc_df.index].mean(axis=1)
    else:
        market_returns = market_returns.loc[pc_df.index]

    ordered_clusters = cluster_means['PC1'].sort_values().index.tolist()
    cluster_rank_map = {c: rank for rank, c in enumerate(ordered_clusters)}
    cluster_ranks = np.array([cluster_rank_map[c] for c in cluster_labels])
    dates = pc_df.index

    min_ret = market_returns.min()
    max_ret = market_returns.max()
    ret_norm = (market_returns - min_ret) / (max_ret - min_ret)
    ret_scaled = ret_norm * (len(ordered_clusters) - 1)

    # Full plot
    plt.figure(figsize=(15,6))
    plt.step(dates, cluster_ranks, where='post', label='Cluster (ordered by PC1)', linewidth=2, color='tab:blue')
    plt.plot(dates, ret_scaled, label='Normalized Market Returns', color='tab:orange', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Cluster Rank (blue) / Returns (orange)')
    plt.title('Cluster Regimes Ordered by Mean PC1 vs Normalized Market Returns')
    yticks = list(range(len(ordered_clusters)))
    yticklabels = [f"Cluster {c}\nMean PC1={cluster_means.loc[c, 'PC1']:.2f}" for c in ordered_clusters]
    plt.yticks(yticks, yticklabels)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Zoomed in plot (optional)
    if zoom_dates is not None:
        start_date, end_date = zoom_dates
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        zoom_dates_range = dates[mask]
        zoom_cluster_ranks = cluster_ranks[mask]
        zoom_ret_scaled = ret_scaled[mask]

        plt.figure(figsize=(15,6))
        plt.step(zoom_dates_range, zoom_cluster_ranks, where='post', label='Cluster (ordered by PC1)', linewidth=2, color='tab:blue')
        plt.plot(zoom_dates_range, zoom_ret_scaled, label='Normalized Market Returns', color='tab:orange', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Cluster Rank (blue) / Returns (orange)')
        plt.title(f'Zoomed: Cluster Regimes Ordered by Mean PC1 vs Normalized Market Returns\n({start_date} to {end_date})')
        plt.yticks(yticks, yticklabels)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def compute_and_print_r2_pc1_market(pc_df, market_returns):
    pc1 = pc_df['PC1'].loc[market_returns.index]
    X = sm.add_constant(market_returns)
    model = sm.OLS(pc1, X).fit()
    r2 = model.rsquared
    print(f"R² between PC1 and market returns over full period: {r2:.4f}")
    return r2

def plot_rolling_r2_pc1_market(pc_df, market_returns, window=63):
    pc1 = pc_df['PC1'].loc[market_returns.index]
    r2_values = []
    dates = pc1.index[window-1:]

    for i in range(window, len(pc1)+1):
        y = pc1.iloc[i-window:i]
        X = sm.add_constant(market_returns.iloc[i-window:i])
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)

    plt.figure(figsize=(14,6))
    plt.plot(dates, r2_values, color='tab:green')
    plt.title(f'Rolling R² of PC1 vs Market Returns (window={window} days)')
    plt.xlabel('Date')
    plt.ylabel('R²')
    plt.ylim(0,1)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(pca):
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_*100)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Explained Variance by Component")
    plt.xticks(range(1, len(pca.explained_variance_ratio_)+1))
    plt.grid(axis='y')
    plt.show()

def plot_all_pcs_over_time(pc_df):
    plt.figure(figsize=(14, 7))
    for pc in pc_df.columns:
        plt.plot(pc_df.index, pc_df[pc], label=pc)
    plt.title("PCA Scores Over Time for All Principal Components")
    plt.xlabel("Date")
    plt.ylabel("PC Score")
    plt.legend(loc="upper right", fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_dates_by_cluster(pc_df, cluster_labels):
    """
    Returns a dict: cluster_label -> list of dates in that cluster
    """
    df = pc_df.copy()
    df['Cluster'] = cluster_labels
    cluster_dates = {}
    for cluster in np.unique(cluster_labels):
        cluster_dates[cluster] = df.index[df['Cluster'] == cluster].tolist()
    return cluster_dates


plot_explained_variance(pca)
plot_all_pcs_over_time(pc_df)


# -----------------------------
# Main workflow example:
# -----------------------------

# 1. Cluster PCA scores
cluster_labels, kmeans_model = cluster_pca_scores(pc_df, n_clusters=3)
cluster_dates = get_dates_by_cluster(pc_df, cluster_labels)
day = cluster_dates[2][3]
# 2. Plot clusters in 2D and 3D
plot_pca_clusters(pc_df, cluster_labels, kmeans_model)
# plot_pca_clusters_3d(pc_df, cluster_labels, kmeans_model)

# 3. Interpret clusters
cluster_means = interpret_clusters(pc_df, cluster_labels)

# 4. Compute rolling R² correlations
r2_dict = compute_rolling_r2(returns, pc_df, sector_lists, window=63)

# 5. Generate buy signals example for a date
print()
print(f"The day is: {day}")
print()
try:
    current_date = pd.Timestamp('2025-01-02')
    day = cluster_dates[2][-1]

    buy_signals, sector_scores = generate_cluster_based_signals(
        current_date=day,
        pc_df=pc_df,
        cluster_labels=cluster_labels,
        r2_dict=r2_dict,
        cluster_means=cluster_means,
        positive_threshold=0.1,
        negative_threshold=0.05,
        r2_threshold=0.3
    )
except ValueError as e:
    print(e)

# 6. Plot cluster regimes vs normalized returns
plot_pca_clusters_3d(pc_df, cluster_labels, kmeans_model)


# -----------------------------
# Optional: Rolling R² plot for selected ETFs & PCs
all_tickers = [etf for etfs in sector_lists.values() for etf in etfs]

plot_pca_r2_over_time(returns, sector_lists, window_size=63, etfs_to_plot=all_tickers)


spy_data = yf.download('SPY', period='5y', auto_adjust=True)['Close']
spy_returns = spy_data.pct_change().reindex(returns.index).dropna()
plot_clusters_vs_returns(
    pc_df=pc_df,
    cluster_labels=cluster_labels,
    cluster_means=cluster_means,
    returns=returns,
    market_etf='SPY',
    market_returns=spy_returns,
    zoom_dates=('2023-01-01', '2023-06-01')  # example zoom window
)

# 2. Compute and print R² over full period
r2_full = compute_and_print_r2_pc1_market(pc_df, spy_returns)

# 3. Plot rolling R²
plot_rolling_r2_pc1_market(pc_df, spy_returns, window=63)



