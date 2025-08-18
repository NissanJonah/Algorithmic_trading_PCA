import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


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
def plot_pca_r2_over_time(returns, sector_lists, window_size=63, etfs_to_plot=None, num_graphs=0):
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


def plot_clusters_vs_returns(pc_df, cluster_labels, market_returns, zoom_dates=None):
    """
    Plot cluster groups alongside market returns without normalization

    Parameters:
        pc_df: DataFrame of principal component scores
        cluster_labels: array-like of cluster assignments
        market_returns: Series of market returns
        zoom_dates: optional tuple of (start_date, end_date) for zoomed plot
    """
    # Align data
    market_returns = market_returns.reindex(pc_df.index).ffill().bfill()

    # Convert cluster_labels to numpy array if it's not already
    cluster_labels = np.array(cluster_labels)

    # Validation
    if len(cluster_labels) != len(market_returns):
        raise ValueError(f"Length mismatch: clusters ({len(cluster_labels)}) != returns ({len(market_returns)})")

    # Ensure both arrays are 1D
    if cluster_labels.ndim > 1:
        cluster_labels = cluster_labels.flatten()
    if market_returns.ndim > 1:
        market_returns = market_returns.values.flatten()

    dates = pc_df.index

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plot clusters on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cluster', color=color)
    ax1.plot(dates, cluster_labels, color=color, label='Cluster Group')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second y-axis for returns
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Market Returns', color=color)
    ax2.plot(dates, market_returns, color=color, alpha=0.7, label='Market Returns')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Cluster Groups vs Market Returns')
    fig.tight_layout()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()

    # Calculate correlation
    correlation = np.corrcoef(cluster_labels, market_returns)[0, 1]
    print(f"Correlation between cluster groups and market returns: {correlation:.4f}")

    # Zoomed in plot (optional)
    if zoom_dates is not None:
        start_date, end_date = zoom_dates
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        zoom_dates_range = dates[mask]
        zoom_clusters = cluster_labels[mask]
        zoom_returns = market_returns[mask]

        fig, ax1 = plt.subplots(figsize=(15, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cluster', color=color)
        ax1.plot(zoom_dates_range, zoom_clusters, color=color, label='Cluster Group')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Market Returns', color=color)
        ax2.plot(zoom_dates_range, zoom_returns, color=color, alpha=0.7, label='Market Returns')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Zoomed: Cluster Groups vs Market Returns\n({start_date} to {end_date})')
        fig.tight_layout()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.show()

    return correlation


def plot_rolling_r2_clusters_returns(cluster_labels, market_returns, window=63):
    """
    Rolling window R² between cluster groups and market returns without normalization

    Parameters:
        cluster_labels: array-like of cluster assignments
        market_returns: pandas Series of market returns with datetime index
        window: rolling window size in days
    """
    # Convert to numpy arrays if they aren't already
    cluster_labels = np.asarray(cluster_labels)
    market_returns_values = market_returns.values

    # Validation
    if len(cluster_labels) != len(market_returns_values):
        raise ValueError(f"Length mismatch: clusters ({len(cluster_labels)}) != returns ({len(market_returns_values)})")

    # Calculate rolling R²
    r2_values = []
    dates = market_returns.index[window - 1:]

    for i in range(window, len(cluster_labels) + 1):
        y = market_returns_values[i - window:i]
        X = sm.add_constant(cluster_labels[i - window:i])
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)

    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(dates, r2_values, color='tab:green')
    plt.title(f'Rolling R² of Cluster Groups vs Market Returns (window={window} days)')
    plt.xlabel('Date')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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

def get_average_returns(dates, returns_df):
    """
    Calculate the average return for each ETF over the given list of dates.

    Parameters:
    - dates: list of dates (string 'YYYY-MM-DD' or pandas Timestamps)
    - returns_df: DataFrame of daily returns for all ETFs

    Returns:
    - Series of average returns per ETF
    """
    # Ensure dates are Timestamps so we can index properly
    date_index = pd.to_datetime(dates)

    # Filter for available dates only (avoids missing data errors)
    valid_dates = date_index.intersection(returns_df.index)

    if len(valid_dates) == 0:
        raise ValueError("None of the provided dates are in the returns DataFrame index.")

    # Subset and compute mean return per ETF
    avg_returns = returns_df.loc[valid_dates].mean()
    return avg_returns

# -----------------------------
# Main workflow example:
# -----------------------------
plot_explained_variance(pca)
plot_all_pcs_over_time(pc_df)

# 1. Cluster PCA scores
cluster_labels, kmeans_model = cluster_pca_scores(pc_df, n_clusters=2)
cluster_dates = get_dates_by_cluster(pc_df, cluster_labels)

# 2. Plot clusters in 2D and 3D
plot_pca_clusters(pc_df, cluster_labels, kmeans_model)
# plot_pca_clusters_3d(pc_df, cluster_labels, kmeans_model)

# 3. Interpret clusters
cluster_means = interpret_clusters(pc_df, cluster_labels)

# 4. Compute rolling R² correlations
r2_dict = compute_rolling_r2(returns, pc_df, sector_lists, window=63)

# 5. Generate buy signals example for a date
print()
day = cluster_dates[1][3]
print(f"The day is: {day}")
print()
try:
    current_date = pd.Timestamp('2025-01-02')
    day = cluster_dates[1][-1]

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
 #hi f

spy_data = yf.download('SPY', period='20y', auto_adjust=True)['Close']
spy_returns = spy_data.pct_change().reindex(returns.index).dropna()

# 2. Compute and print R² over full period
r2_full = compute_and_print_r2_pc1_market(pc_df, spy_returns)

# 3. Plot rolling R²
plot_rolling_r2_pc1_market(pc_df, spy_returns, window=63)

spy_returns_aligned = spy_returns.reindex(pc_df.index).ffill().bfill()

# Call the plot function

# Get market returns
spy_returns = yf.download('SPY', period='20y', auto_adjust=True)['Close'].pct_change().dropna()

# Plot and calculate R²

spy_returns_aligned = spy_returns.reindex(pc_df.index).ffill().bfill()

# Call the plot function

full_r2 = plot_clusters_vs_returns(
    pc_df=pc_df,
    cluster_labels=cluster_labels,  # From your KMeans clustering
    market_returns=spy_returns,
    zoom_dates=('2023-01-01', '2023-06-01')
)

print(f"The full R2 is: {full_r2}")

# Show rolling R²
plot_rolling_r2_clusters_returns(
    cluster_labels=cluster_labels,
    market_returns=spy_returns.reindex(pc_df.index).ffill().bfill()
)



cluster_0_dates = cluster_dates[0]
cluster_1_dates = cluster_dates[1]




avg_cluster1_returns = get_average_returns(cluster_1_dates, returns)
print("cluster 1 average returns")
print(f"{avg_cluster1_returns}")

avg_cluster0_returns = get_average_returns(cluster_0_dates, returns)
print("cluster 0 average returns")
print(f"{avg_cluster0_returns}")
print()
print()

#=================================
#buy and sell signal momentum code
#=================================
print("====================")
print("Momentum Information")
print("====================")
import pandas as pd

def get_forward_returns_x_days(dates, returns_df, x):
    """
    Calculate average returns for each ETF x days AFTER the given dates.
    Returns a text summary string.
    """
    date_index = pd.to_datetime(dates)
    valid_dates = date_index.intersection(returns_df.index)

    if len(valid_dates) == 0:
        return f"No valid dates found for forward period of {x} days."

    forward_returns = []
    for date in valid_dates:
        if date in returns_df.index:
            pos = returns_df.index.get_loc(date)
            if pos + x < len(returns_df.index):
                future_date = returns_df.index[pos + x]
                forward_returns.append(returns_df.loc[future_date])

    if not forward_returns:
        return f"No forward returns available for {x} days ahead."

    avg_forward = pd.concat(forward_returns, axis=1).mean(axis=1)
    return f"Average returns {x} days after given dates:\n{avg_forward.to_string()}"


def get_forward_returns_y_days(dates, returns_df, y):
    """
    Calculate average returns for each ETF y days AFTER the given dates.
    Returns a text summary string.
    """
    date_index = pd.to_datetime(dates)
    valid_dates = date_index.intersection(returns_df.index)

    if len(valid_dates) == 0:
        return f"No valid dates found for forward period of {y} days."

    forward_returns = []
    for date in valid_dates:
        if date in returns_df.index:
            pos = returns_df.index.get_loc(date)
            if pos + y < len(returns_df.index):
                future_date = returns_df.index[pos + y]
                forward_returns.append(returns_df.loc[future_date])

    if not forward_returns:
        return f"No forward returns available for {y} days ahead."

    avg_forward = pd.concat(forward_returns, axis=1).mean(axis=1)
    return f"Average returns {y} days after given dates:\n{avg_forward.to_string()}"


def get_cluster_positive_return_percentages(dates, returns_df, x, y):
    """
    Calculate percentage of days with positive average returns:
     - 1 day after
     - x days after
     - y days after
    for the given list of dates.
    Returns a text summary string.
    """
    date_index = pd.to_datetime(dates)
    valid_dates = date_index.intersection(returns_df.index)

    def pct_positive(dates_list, forward_days):
        count_positive = 0
        count_total = 0
        for date in dates_list:
            if date in returns_df.index:
                pos = returns_df.index.get_loc(date)
                if pos + forward_days < len(returns_df.index):
                    future_date = returns_df.index[pos + forward_days]
                    avg_return = returns_df.loc[future_date].mean()
                    count_total += 1
                    if avg_return > 0:
                        count_positive += 1
        return (count_positive / count_total * 100) if count_total > 0 else 0

    pos_1d = pct_positive(valid_dates, 1)
    pos_xd = pct_positive(valid_dates, x)
    pos_yd = pct_positive(valid_dates, y)

    return (
        f"Percentage of positive average returns after given dates:\n"
        f"  1 day after: {pos_1d:.2f}%\n"
        f"  {x} days after: {pos_xd:.2f}%\n"
        f"  {y} days after: {pos_yd:.2f}%"
    )

# Example usage calls for your clusters — exactly like your style:
cluster_0_dates = cluster_dates[0]
cluster_1_dates = cluster_dates[1]

print("cluster 1 average returns (y days after):")
print(get_forward_returns_y_days(cluster_1_dates, returns, y=20))

print("cluster 0 average returns (y days after):")
print(get_forward_returns_y_days(cluster_0_dates, returns, y=20))

print("cluster 1 positive return percentages (1 day, x days, y days):")
print(get_cluster_positive_return_percentages(cluster_1_dates, returns, x=5, y=20))

print("cluster 0 positive return percentages (1 day, x days, y days):")
print(get_cluster_positive_return_percentages(cluster_0_dates, returns, x=5, y=20))


print()
print("===========================")
print()

def cluster_profitability_with_pc1_momentum(pc_df, cluster_labels, returns_df, ma_window=5, lookahead_days=[1, 5, 10, 20]):
    """
    Calculate profitability % for each cluster at various lookahead days after dates where PC1 momentum is up.

    Parameters:
    - pc_df: DataFrame with PCs, index = dates
    - cluster_labels: array-like cluster assignments aligned with pc_df index
    - returns_df: DataFrame of daily returns for ETFs, indexed by date
    - ma_window: window size for moving average to define PC1 momentum (default 5)
    - lookahead_days: list of ints for days after to check returns (default [1,5,10,20])

    Returns:
    - text summary of profitability percentages per cluster and lookahead day
    """

    pc1 = pc_df['PC1']
    # Calculate moving average of PC1
    pc1_ma = pc1.rolling(window=ma_window).mean()

    # Momentum up = pc1_ma today > pc1_ma yesterday
    momentum_up = pc1_ma > pc1_ma.shift(1)

    # Filter dates where momentum is up and drop NAs from rolling mean
    valid_dates = pc_df.index[momentum_up.fillna(False)]

    # Prepare results dict: cluster -> lookahead_day -> profitability %
    results = {}

    # Map cluster labels to index for convenience
    cluster_labels = np.array(cluster_labels)

    for cluster in np.unique(cluster_labels):
        # Get all dates in this cluster where momentum is up
        cluster_dates = [date for i, date in enumerate(pc_df.index) if cluster_labels[i] == cluster and date in valid_dates]

        # Initialize nested dict for this cluster
        results[cluster] = {}

        for days_ahead in lookahead_days:
            positive_count = 0
            total_count = 0

            for date in cluster_dates:
                lookahead_date = date + pd.Timedelta(days=days_ahead)
                # Find closest available date >= lookahead_date in returns_df
                future_dates = returns_df.index[returns_df.index >= lookahead_date]
                if len(future_dates) == 0:
                    continue  # no future data
                actual_date = future_dates[0]

                # Calculate average return across ETFs on that day
                avg_return = returns_df.loc[actual_date].mean()
                total_count += 1
                if avg_return > 0:
                    positive_count += 1

            profitability = (positive_count / total_count * 100) if total_count > 0 else float('nan')
            results[cluster][days_ahead] = profitability

    # Build summary text
    summary_lines = ["Profitability % per cluster when PC1 momentum is UP (by days after):\n"]
    for cluster in sorted(results.keys()):
        summary_lines.append(f"Cluster {cluster}:")
        for days_ahead in sorted(results[cluster].keys()):
            profit_pct = results[cluster][days_ahead]
            if np.isnan(profit_pct):
                summary_lines.append(f"  {days_ahead} days after: No data")
            else:
                summary_lines.append(f"  {days_ahead} days after: {profit_pct:.2f}% positive returns")
        summary_lines.append("")  # blank line for spacing

    return "\n".join(summary_lines)


print(cluster_profitability_with_pc1_momentum(pc_df, cluster_labels, returns))



#+++++++++++++++++++++++++

