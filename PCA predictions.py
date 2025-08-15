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
data = yf.download(all_etfs, period="7y", auto_adjust=True)
data = data['Close']
print(f"Using {len(data.columns)} ETFs after filtering.")

# Compute daily returns for ETFs
returns = data.pct_change().dropna()

# -----------------------------
# 3. FIXED: Train/Test Split (No data leakage)
# -----------------------------
# Define split date (e.g., 80% train, 20% test)
split_date = returns.index[int(len(returns) * 0.8)]
print(f"Split date: {split_date}")

train_returns = returns.loc[:split_date]
test_returns = returns.loc[split_date:]

print(f"Train samples: {len(train_returns)}")
print(f"Test samples: {len(test_returns)}")

# STEP 1: Standardize ETF returns using ONLY training statistics
train_mean = train_returns.mean()
train_std = train_returns.std()

# Apply same standardization to both train and test
train_returns_std = (train_returns - train_mean) / train_std
test_returns_std = (test_returns - train_mean) / train_std

# -----------------------------
# 4. FIXED: Fit PCA on training data only
# -----------------------------
pca = PCA()
Y_train = pca.fit_transform(train_returns_std)  # Fit only on training data
V = pca.components_.T  # Loadings matrix: rows = ETFs, cols = PCs

# Transform test data using the same PCA fitted on training data
Y_test = pca.transform(test_returns_std)


# -----------------------------
# 5. FIXED: Deterministic sign alignment
# -----------------------------
def align_pca_signs(V, Y_train, Y_test=None):
    """
    Align PC signs deterministically by ensuring the largest absolute loading
    in each PC is positive.
    """
    V_aligned = V.copy()
    Y_train_aligned = Y_train.copy()
    Y_test_aligned = Y_test.copy() if Y_test is not None else None

    for i in range(V.shape[1]):  # For each PC
        # Find ETF with largest absolute loading
        max_loading_idx = np.argmax(np.abs(V[:, i]))
        if V[max_loading_idx, i] < 0:
            # Flip signs if largest loading is negative
            V_aligned[:, i] *= -1
            Y_train_aligned[:, i] *= -1
            if Y_test_aligned is not None:
                Y_test_aligned[:, i] *= -1

    return V_aligned, Y_train_aligned, Y_test_aligned


V_aligned, Y_train_aligned, Y_test_aligned = align_pca_signs(V, Y_train, Y_test)

# Create DataFrames for PC scores
train_pc_df = pd.DataFrame(Y_train_aligned, index=train_returns_std.index,
                           columns=[f"PC{i + 1}" for i in range(Y_train_aligned.shape[1])])
test_pc_df = pd.DataFrame(Y_test_aligned, index=test_returns_std.index,
                          columns=[f"PC{i + 1}" for i in range(Y_test_aligned.shape[1])])

# -----------------------------
# 6. FIXED: Standardize PC scores using training statistics
# -----------------------------
# Compute PC scaling parameters from training data only
train_pc_mean = train_pc_df.mean()
train_pc_std = train_pc_df.std()

# Apply same scaling to both train and test PC scores
train_pc_scaled = (train_pc_df - train_pc_mean) / train_pc_std
test_pc_scaled = (test_pc_df - train_pc_mean) / train_pc_std

print("Train PC scaled stats (should be ~0 mean, ~1 std):")
print(f"Mean: {train_pc_scaled.mean().round(3)}")
print(f"Std: {train_pc_scaled.std().round(3)}")


# -----------------------------
# 7. FIXED: Save all preprocessing artifacts
# -----------------------------
def save_preprocessing_artifacts(train_mean, train_std, pca, V_aligned,
                                 train_pc_mean, train_pc_std, save_dir="pca_artifacts"):
    """
    Save all preprocessing steps for reproducible inference.
    """
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

    # Save artifacts
    artifact_path = os.path.join(save_dir, 'pca_preprocessing.joblib')
    joblib.dump(artifacts, artifact_path)
    print(f"Preprocessing artifacts saved to: {artifact_path}")

    return artifacts


artifacts = save_preprocessing_artifacts(train_mean, train_std, pca, V_aligned,
                                         train_pc_mean, train_pc_std)


# -----------------------------
# 8. Function to load and apply preprocessing (for inference)
# -----------------------------
def load_and_transform_new_data(new_returns, artifact_path="pca_artifacts/pca_preprocessing.joblib"):
    """
    Load preprocessing artifacts and transform new data.
    Use this for inference on new data.
    """
    # Load saved artifacts
    artifacts = joblib.load(artifact_path)

    # Extract components
    train_etf_mean = artifacts['train_etf_mean']
    train_etf_std = artifacts['train_etf_std']
    pca_object = artifacts['pca_object']
    V_aligned = artifacts['pca_loadings_aligned']
    train_pc_mean = artifacts['train_pc_mean']
    train_pc_std = artifacts['train_pc_std']

    # Apply same preprocessing pipeline
    # 1. Standardize using training ETF statistics
    new_returns_std = (new_returns - train_etf_mean) / train_etf_std

    # 2. Transform using fitted PCA
    Y_new = pca_object.transform(new_returns_std)

    # 3. Apply sign alignment (loadings already contain the alignment)
    # Note: We need to manually apply the sign flips from V_aligned
    for i in range(V_aligned.shape[1]):
        original_loading = pca_object.components_.T[:, i]
        aligned_loading = V_aligned[:, i]
        if not np.allclose(original_loading, aligned_loading):
            Y_new[:, i] *= -1

    # 4. Standardize PC scores using training PC statistics
    pc_df = pd.DataFrame(Y_new, index=new_returns_std.index,
                         columns=[f"PC{i + 1}" for i in range(Y_new.shape[1])])
    pc_scaled = (pc_df - train_pc_mean) / train_pc_std

    return pc_scaled, artifacts


# Test the inference function on test data
test_pc_inference, _ = load_and_transform_new_data(test_returns)
print(f"\nInference test - Max difference between direct and loaded transform: "
      f"{np.max(np.abs(test_pc_scaled.values - test_pc_inference.values)):.6f}")

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

# Check train vs test PC correlations
print("\nTrain PC correlations (should be ~diagonal):")
train_corr = train_pc_df.corr()
print(f"Off-diagonal max: {np.max(np.abs(train_corr.values - np.eye(len(train_corr)))):.3f}")

print("\nTest PC correlations (should be near-diagonal):")
test_corr = test_pc_df.corr()
print(f"Off-diagonal max: {np.max(np.abs(test_corr.values - np.eye(len(test_corr)))):.3f}")


# -----------------------------
# 10. FIXED: Separate rolling analysis from forecasting
# -----------------------------
def analyze_pca_stability_over_time(returns, sector_lists, window_size=63,
                                    etfs_to_plot=None, num_graphs=5):
    """
    ANALYSIS ONLY: Rolling-window R² over time for ETFs vs PCs.
    This refits PCA in each window for diagnostic purposes.
    DO NOT use this for forecasting - it's just for understanding stability.
    """
    print("\n" + "=" * 60)
    print("RUNNING ROLLING PCA ANALYSIS (FOR DIAGNOSTICS ONLY)")
    print("This refits PCA in each window - DO NOT use for forecasting!")
    print("=" * 60)

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

        # Refit PCA for this window (analysis only!)
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

    # Plot results
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


# Run the diagnostic analysis
analyze_pca_stability_over_time(returns, sector_lists)

# -----------------------------
# 11. Example forecasting setup (using fixed PCA)
# -----------------------------
def setup_forecasting_model(train_pc_scaled, lookback=5):
    """
    Example: Set up a simple forecasting model using the properly preprocessed PC data.
    This uses the FIXED PCA basis throughout.
    """
    print(f"\nSetting up forecasting model with lookback={lookback} days")

    # Create lagged features for forecasting
    X_list = []
    y_list = []

    for i in range(lookback, len(train_pc_scaled)):
        # Features: previous `lookback` days of PC scores
        X_list.append(train_pc_scaled.iloc[i - lookback:i].values.flatten())
        # Target: next day PC scores
        y_list.append(train_pc_scaled.iloc[i].values)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Forecasting dataset shape: X={X.shape}, y={y.shape}")
    return X, y




# Example usage
X_forecast, y_forecast = setup_forecasting_model(train_pc_scaled)

print("\n" + "=" * 60)
print("SUMMARY OF FIXES IMPLEMENTED:")
print("=" * 60)
print("✅ Step 1: Train/test split with PCA fit only on training data")
print("✅ Step 2: Deterministic sign alignment based on largest absolute loading")
print("✅ Step 3: PC standardization using training statistics only")
print("✅ Step 4: Rolling PCA separated from forecasting pipeline")
print("✅ Step 5: All preprocessing artifacts saved for reproducible inference")
print("\nKey artifacts saved:")
print("- ETF standardization parameters (train_etf_mean, train_etf_std)")
print("- PCA object and aligned loadings")
print("- PC standardization parameters (train_pc_mean, train_pc_std)")
print("- Full pipeline for inference on new data")

# ... existing code ...

print("\n" + "=" * 60)
print("SUMMARY OF FIXES IMPLEMENTED:")
print("=" * 60)
print("✅ Step 1: Train/test split with PCA fit only on training data")
print("✅ Step 2: Deterministic sign alignment based on largest absolute loading")
print("✅ Step 3: PC standardization using training statistics only")
print("✅ Step 4: Rolling PCA separated from forecasting pipeline")
print("✅ Step 5: All preprocessing artifacts saved for reproducible inference")
print("\nKey artifacts saved:")
print("- ETF standardization parameters (train_etf_mean, train_etf_std)")
print("- PCA object and aligned loadings")
print("- PC standardization parameters (train_pc_mean, train_pc_std)")
print("- Full pipeline for inference on new data")

# -----------------------------
# 12. PCA VISUALIZATION PLOTS - The 3 requested plots
# -----------------------------

# Combine train and test PC data for full time series visualization
full_pc_df = pd.concat([train_pc_df, test_pc_df])


def plot_all_pcs_over_time(pc_df):
    """Plot all PCs on one graph over time."""
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
    """Compute rolling R² between an ETF's returns and each PC."""
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

        # Pad with NaN for the first (window-1) observations
        r2_series = [np.nan] * (window - 1) + r2_series
        r2_dict[pc] = r2_series

    return pd.DataFrame(r2_dict, index=pc_df.index)


def plot_etf_r2_over_time(returns, pc_df, sector_lists, window=63):
    """Plot rolling R² between each ETF and all PCs over time."""
    ticker_to_sector = {}
    for sector, tickers in sector_lists.items():
        for ticker in tickers:
            ticker_to_sector[ticker] = sector

    for etf in returns.columns:
        sector_name = ticker_to_sector.get(etf, "Unknown Sector")

        # Compute rolling R² for this ETF
        r2_df = compute_rolling_r2_for_etf(returns, pc_df, etf, window)

        # Plot
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
    """Plot bar chart of eigenvalues for each PC."""
    eigenvalues = pca.explained_variance_
    pc_names = [f'PC{i + 1}' for i in range(len(eigenvalues))]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(pc_names, eigenvalues, alpha=0.7, color='steelblue')
    plt.title("Eigenvalues by Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, eigenvalues):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 13. Generate the 3 requested plots
# -----------------------------
print("\n" + "=" * 60)
print("GENERATING THE 3 REQUESTED PCA PLOTS")
print("=" * 60)

# 1. Plot all PCs over time on one graph
print("1. Plotting all PCs over time...")
plot_all_pcs_over_time(full_pc_df)

# 2. Plot rolling R² for each ETF vs all PCs
# print("2. Plotting rolling R² for each ETF...")
# plot_etf_r2_over_time(returns, full_pc_df, sector_lists, window=63)

# 3. Plot eigenvalues bar chart
print("3. Plotting eigenvalues bar chart...")
plot_eigenvalues_bar(pca)

print("\n" + "=" * 60)
print("ALL 3 REQUESTED PLOTS GENERATED SUCCESSFULLY!")
print("=" * 60)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def create_kmeans_clusters(prices_df, lookback, n_clusters=5, plot_title="K-Means Clustering"):
    """
    Create K-means clusters from lookback-period flattened returns.

    prices_df: DataFrame of ETF prices
    lookback: lookback period in days
    n_clusters: number of clusters
    plot_title: title for the plot
    """
    # Compute returns
    returns_df = prices_df.pct_change().dropna()

    # Create lookback matrix
    lookback_matrix = []
    for i in range(len(returns_df) - lookback):
        # Flatten the last 'lookback' days of returns into a single row
        window = returns_df.iloc[i:i+lookback].values.flatten()
        lookback_matrix.append(window)

    lookback_matrix = np.array(lookback_matrix)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(lookback_matrix)

    # Plot first 2 dimensions of lookback matrix (just for visualization)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(lookback_matrix[:, 0], lookback_matrix[:, 1],
                          c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")

    # Plot centroids ("X" at mean of each cluster)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=200, c='red', linewidths=3, label='Centroids')

    plt.title(f"{plot_title} (Lookback={lookback} days)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

    return kmeans, clusters, lookback_matrix


# Example usage:
# Assuming 'prices_df' is your ETF price dataframe
kmeans_15, clusters_4, matrix_15 = create_kmeans_clusters(data, lookback=14, n_clusters=2, plot_title="15-Day KMeans")
kmeans_60, clusters_14, matrix_60 = create_kmeans_clusters(data, lookback=60, n_clusters=4, plot_title="60-Day KMeans")


def print_cluster_avg_returns(prices_df, clusters, lookback, label=""):
    """
    Print the average forward return for each cluster.

    prices_df : DataFrame of ETF prices (same as used in KMeans)
    clusters  : Cluster labels from KMeans
    lookback  : Lookback period used to generate the clusters
    label     : Optional label to identify the run (e.g., '5-day lookback')
    """
    returns_df = prices_df.pct_change().dropna()
    avg_returns = []

    # Forward returns start after the lookback window
    for cluster_id in np.unique(clusters):
        # Get indices in the original returns DataFrame for this cluster
        idxs = np.where(clusters == cluster_id)[0]

        # Calculate forward returns for each occurrence
        fwd_returns = []
        for i in idxs:
            start_price = prices_df.iloc[i + lookback - 1]  # price at end of lookback
            end_price = prices_df.iloc[i + lookback]        # price one day later
            total_return = (end_price - start_price) / start_price
            fwd_returns.append(total_return.mean())  # mean across ETFs

        avg_return = np.mean(fwd_returns)
        avg_returns.append((cluster_id, avg_return))

    print(f"\nAverage Forward Return by Cluster ({label}):")
    for cid, avg_ret in avg_returns:
        print(f"  Cluster {cid}: {avg_ret:.4%}")


# Example calls for your two lookbacks
print_cluster_avg_returns(data, clusters_4, lookback=14, label="5-Day Lookback")
print_cluster_avg_returns(data, clusters_14, lookback=60, label="14-Day Lookback")


def plot_cluster_trajectories(prices_df, lookback, clusters_list, lookback_labels=None):
    """
    Plot cluster movement over time for one or more KMeans clusterings.

    prices_df: DataFrame of ETF prices
    lookback: lookback period used to compute clusters
    clusters_list: list of tuples [(clusters_array, label), ...]
    lookback_labels: optional list of labels for legend
    """
    import matplotlib.dates as mdates

    plt.figure(figsize=(14, 5))

    for i, (clusters, label) in enumerate(clusters_list):
        # Calculate dates to match the exact length of clusters
        cluster_length = len(clusters)

        # The clusters correspond to the last cluster_length trading days
        # after accounting for the lookback window and pct_change()
        dates = prices_df.index[-(cluster_length):]

        # Debug print to verify lengths match
        print(f"Debug - {label}: dates length = {len(dates)}, clusters length = {len(clusters)}")

        plt.plot(dates, clusters, marker='o', linestyle='-', alpha=0.7, label=label)

    plt.title(f"Cluster Movement Over Time (Lookback={lookback} days)")
    plt.xlabel("Date")
    plt.ylabel("Cluster")
    plt.yticks(range(max(max(clusters) for clusters, _ in clusters_list) + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_cluster_trajectories(
    data,
    lookback=14,
    clusters_list=[(clusters_4, "4-Day KMeans"), (clusters_14[:len(clusters_4)], "60-Day KMeans")]
)


def calculate_cluster_market_r2(clusters_4, clusters_14, data):
    """
    Calculate R² values between cluster changes and SPY rolling returns.

    clusters_4: cluster assignments from 4-day lookback
    clusters_14: cluster assignments from 14-day lookback
    data: DataFrame of ETF prices (should include SPY or we'll download it)
    """
    from sklearn.metrics import r2_score

    # Download SPY data if not in the data DataFrame
    if 'XLI' not in data.columns:
        print("Downloading SPY data...")
        spy_data = yf.download('XLI', period="7y", auto_adjust=True)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data = spy_data['Close']
        spy_prices = spy_data.squeeze()
    else:
        spy_prices = data['XLI']

    # Calculate SPY daily returns
    spy_daily_returns = spy_prices.pct_change().dropna()

    # Calculate rolling SPY returns that match the lookback periods
    # 4-day rolling returns (sum of 4 daily returns)
    spy_4day_returns = spy_daily_returns.rolling(window=14).sum().dropna()

    # 14-day rolling returns (sum of 14 daily returns)
    spy_14day_returns = spy_daily_returns.rolling(window=60).sum().dropna()

    # Calculate cluster changes (difference from previous day)
    cluster_changes_4 = np.diff(clusters_4)
    cluster_changes_14 = np.diff(clusters_14)

    # Align the data by taking the last N observations to match cluster lengths
    spy_4day_aligned = spy_4day_returns.iloc[-(len(cluster_changes_4)):].values
    spy_14day_aligned = spy_14day_returns.iloc[-(len(cluster_changes_14)):].values

    # Debug: Print lengths to verify alignment
    print(f"Debug lengths:")
    print(f"  cluster_changes_4: {len(cluster_changes_4)}")
    print(f"  spy_4day_aligned: {len(spy_4day_aligned)}")
    print(f"  cluster_changes_14: {len(cluster_changes_14)}")
    print(f"  spy_14day_aligned: {len(spy_14day_aligned)}")

    # Calculate R² values
    r2_4_day = r2_score(spy_4day_aligned, cluster_changes_4)
    r2_14_day = r2_score(spy_14day_aligned, cluster_changes_14)

    # Also calculate correlation coefficients for additional insight
    corr_4_day = np.corrcoef(spy_4day_aligned, cluster_changes_4)[0, 1]
    corr_14_day = np.corrcoef(spy_14day_aligned, cluster_changes_14)[0, 1]

    print("\n=== Cluster Changes vs SPY Rolling Returns Analysis ===")
    print(f"4-day lookback clusters vs 4-day SPY rolling returns:")
    print(f"  R² with SPY 4-day returns: {r2_4_day:.4f}")
    print(f"  Correlation with SPY 4-day returns: {corr_4_day:.4f}")
    print(f"  Number of data points: {len(cluster_changes_4)}")

    print(f"\n14-day lookback clusters vs 14-day SPY rolling returns:")
    print(f"  R² with SPY 14-day returns: {r2_14_day:.4f}")
    print(f"  Correlation with SPY 14-day returns: {corr_14_day:.4f}")
    print(f"  Number of data points: {len(cluster_changes_14)}")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: 4-day cluster changes vs 4-day SPY returns
    ax1.scatter(cluster_changes_4, spy_4day_aligned, alpha=0.6, color='blue')
    ax1.set_xlabel('4-Day Cluster Changes')
    ax1.set_ylabel('SPY 4-Day Rolling Returns')
    ax1.set_title(f'4-Day Cluster Changes vs SPY 4-Day Returns (R² = {r2_4_day:.4f})')
    ax1.grid(True, alpha=0.3)

    # Plot 2: 14-day cluster changes vs 14-day SPY returns
    ax2.scatter(cluster_changes_14, spy_14day_aligned, alpha=0.6, color='red')
    ax2.set_xlabel('14-Day Cluster Changes')
    ax2.set_ylabel('SPY 14-Day Rolling Returns')
    ax2.set_title(f'14-Day Cluster Changes vs SPY 14-Day Returns (R² = {r2_14_day:.4f})')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time series of 4-day cluster changes and SPY returns
    dates_4 = spy_4day_returns.index[-(len(cluster_changes_4)):]
    ax3.plot(dates_4, cluster_changes_4, label='Cluster Changes', alpha=0.7, color='blue')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(dates_4, spy_4day_aligned, label='SPY 4-Day Returns', alpha=0.7, color='orange')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('4-Day Cluster Changes', color='blue')
    ax3_twin.set_ylabel('SPY 4-Day Returns', color='orange')
    ax3.set_title('4-Day Cluster Changes vs SPY 4-Day Returns Over Time')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Time series of 14-day cluster changes and SPY returns
    dates_14 = spy_14day_returns.index[-(len(cluster_changes_14)):]
    ax4.plot(dates_14, cluster_changes_14, label='Cluster Changes', alpha=0.7, color='red')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(dates_14, spy_14day_aligned, label='SPY 14-Day Returns', alpha=0.7, color='orange')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('14-Day Cluster Changes', color='red')
    ax4_twin.set_ylabel('SPY 14-Day Returns', color='orange')
    ax4.set_title('14-Day Cluster Changes vs SPY 14-Day Returns Over Time')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return {
        '4_day_r2': r2_4_day,
        '14_day_r2': r2_14_day,
        '4_day_corr': corr_4_day,
        '14_day_corr': corr_14_day
    }


# Call the function after creating your clusters
results = calculate_cluster_market_r2(clusters_4, clusters_14, data)


def calculate_cluster_etf_performance(clusters, data, lookback, label=""):
    """Calculate average forward return for each cluster against all ETFs."""
    returns_df = data.pct_change().dropna()

    # Get the dates that correspond to our clusters
    cluster_dates = returns_df.index[lookback:lookback + len(clusters)]

    print(f"\n=== {label} - Cluster Performance vs All ETFs ===")

    results = {}
    for etf in data.columns:
        etf_results = []

        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]

            fwd_returns = []
            for i in cluster_indices:
                if i + lookback < len(returns_df):
                    # Forward return for this ETF
                    fwd_ret = returns_df[etf].iloc[i + lookback]
                    fwd_returns.append(fwd_ret)

            if fwd_returns:
                avg_return = np.mean(fwd_returns)
                etf_results.append((cluster_id, avg_return, len(fwd_returns)))

        results[etf] = etf_results

        # Print results for this ETF
        print(f"\n{etf}:")
        for cluster_id, avg_ret, count in etf_results:
            print(f"  Cluster {cluster_id}: {avg_ret:.4%} (n={count})")

    return results


def calculate_etf_cluster_correlations(clusters, data, lookback, label=""):
    """Calculate correlations between ETFs in same clusters on same dates."""
    returns_df = data.pct_change().dropna()

    # Get the dates that correspond to our clusters
    cluster_dates = returns_df.index[lookback:lookback + len(clusters)]

    print(f"\n=== {label} - ETF Correlations within Same Clusters ===")

    cluster_correlations = {}

    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_dates_subset = [cluster_dates[i] for i in cluster_indices]

        # Get returns for all ETFs on these specific dates
        cluster_returns = returns_df.loc[cluster_dates_subset]

        if len(cluster_returns) > 1:  # Need at least 2 observations for correlation
            # Calculate correlation matrix for this cluster
            corr_matrix = cluster_returns.corr()

            # Calculate average correlation (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            avg_correlation = corr_matrix.values[mask].mean()

            cluster_correlations[cluster_id] = {
                'correlation_matrix': corr_matrix,
                'average_correlation': avg_correlation,
                'n_observations': len(cluster_returns)
            }

            print(f"\nCluster {cluster_id} (n={len(cluster_returns)} dates):")
            print(f"  Average correlation: {avg_correlation:.4f}")
            print(
                f"  Date range: {cluster_dates_subset[0].strftime('%Y-%m-%d')} to {cluster_dates_subset[-1].strftime('%Y-%m-%d')}")

    return cluster_correlations


def plot_cluster_correlation_comparison(corr_results_4, corr_results_14, clusters_4, clusters_14, data, lookback_4=4,
                                        lookback_14=14):
    """Compare correlation patterns between different lookback periods."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Average correlations by cluster for 4-day lookback
    clusters_4_unique = sorted(corr_results_4.keys())
    avg_corrs_4 = [corr_results_4[c]['average_correlation'] for c in clusters_4_unique]

    ax1.bar([f'Cluster {c}' for c in clusters_4_unique], avg_corrs_4, alpha=0.7, color='blue')
    ax1.set_title(f'{lookback_4}-Day Lookback: Average ETF Correlations by Cluster')
    ax1.set_ylabel('Average Correlation')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average correlations by cluster for 14-day lookback
    clusters_14_unique = sorted(corr_results_14.keys())
    avg_corrs_14 = [corr_results_14[c]['average_correlation'] for c in clusters_14_unique]

    ax2.bar([f'Cluster {c}' for c in clusters_14_unique], avg_corrs_14, alpha=0.7, color='red')
    ax2.set_title(f'{lookback_14}-Day Lookback: Average ETF Correlations by Cluster')
    ax2.set_ylabel('Average Correlation')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cluster transitions over time for 4-day
    returns_df = data.pct_change().dropna()
    dates_4 = returns_df.index[lookback_4:lookback_4 + len(clusters_4)]

    ax3.plot(dates_4, clusters_4, marker='o', linestyle='-', alpha=0.7, color='blue', markersize=2)
    ax3.set_title(f'{lookback_4}-Day Lookback: Cluster Assignments Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cluster')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Cluster transitions over time for 14-day
    dates_14 = returns_df.index[lookback_14:lookback_14 + len(clusters_14)]

    ax4.plot(dates_14, clusters_14, marker='o', linestyle='-', alpha=0.7, color='red', markersize=2)
    ax4.set_title(f'{lookback_14}-Day Lookback: Cluster Assignments Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cluster')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_cluster_date_overlap(clusters_4, clusters_14, data, lookback_4=4, lookback_14=14):
    """Analyze how well cluster assignments match on overlapping dates."""
    returns_df = data.pct_change().dropna()

    dates_4 = returns_df.index[lookback_4:lookback_4 + len(clusters_4)]
    dates_14 = returns_df.index[lookback_14:lookback_14 + len(clusters_14)]

    # Find overlapping dates
    overlap_dates = dates_4.intersection(dates_14)

    if len(overlap_dates) == 0:
        print("No overlapping dates found between cluster assignments")
        return

    # Get cluster assignments for overlapping dates
    clusters_4_overlap = []
    clusters_14_overlap = []

    for date in overlap_dates:
        idx_4 = dates_4.get_loc(date)
        idx_14 = dates_14.get_loc(date)
        clusters_4_overlap.append(clusters_4[idx_4])
        clusters_14_overlap.append(clusters_14[idx_14])

    clusters_4_overlap = np.array(clusters_4_overlap)
    clusters_14_overlap = np.array(clusters_14_overlap)

    # Calculate correlation between cluster assignments
    if len(np.unique(clusters_4_overlap)) > 1 and len(np.unique(clusters_14_overlap)) > 1:
        correlation = np.corrcoef(clusters_4_overlap, clusters_14_overlap)[0, 1]
    else:
        correlation = np.nan

    print(f"\n=== Cluster Assignment Overlap Analysis ===")
    print(f"Overlapping dates: {len(overlap_dates)}")
    print(f"Date range: {overlap_dates[0].strftime('%Y-%m-%d')} to {overlap_dates[-1].strftime('%Y-%m-%d')}")
    print(f"Correlation between {lookback_4}-day and {lookback_14}-day cluster assignments: {correlation:.4f}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.scatter(clusters_4_overlap, clusters_14_overlap, alpha=0.6)
    plt.xlabel(f'{lookback_4}-Day Cluster Assignment')
    plt.ylabel(f'{lookback_14}-Day Cluster Assignment')
    plt.title(f'Cluster Assignment Comparison (r={correlation:.3f})')
    plt.grid(True, alpha=0.3)
    plt.show()

    return correlation


# ==============================================================================
# USAGE: Add these function calls at the end of your existing code:
# ==============================================================================

# Calculate performance against all ETFs
perf_results_4 = calculate_cluster_etf_performance(clusters_4, data, lookback=14, label="4-Day Lookback")
perf_results_14 = calculate_cluster_etf_performance(clusters_14, data, lookback=60, label="14-Day Lookback")

# Calculate correlations within clusters
corr_results_4 = calculate_etf_cluster_correlations(clusters_4, data, lookback=14, label="4-Day Lookback")
corr_results_14 = calculate_etf_cluster_correlations(clusters_14, data, lookback=60, label="14-Day Lookback")

# Plot correlation comparisons
plot_cluster_correlation_comparison(corr_results_4, corr_results_14, clusters_4, clusters_14, data)

# Analyze cluster date overlap
overlap_correlation = analyze_cluster_date_overlap(clusters_4, clusters_14, data)



#++++++++++++++++++++++++++++++
#Generate buys



def compute_rolling_r2_all_etfs(returns, pc_df, window=63):
    """
    Compute rolling R² between all ETFs and all PCs.
    Returns a MultiIndex DataFrame with (ETF, PC) columns.
    """
    returns_aligned = returns.loc[pc_df.index]
    results = {}

    for etf in returns.columns:
        for pc in pc_df.columns:
            r2_series = []
            for end_ix in range(window, len(pc_df) + 1):
                start_ix = end_ix - window
                y = returns_aligned[etf].iloc[start_ix:end_ix]
                X = pc_df[pc].iloc[start_ix:end_ix]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                r2_series.append(model.rsquared)

            # Pad with NaN for the first (window-1) observations
            r2_series = [np.nan] * (window - 1) + r2_series
            results[(etf, pc)] = r2_series

    # Create MultiIndex DataFrame
    columns = pd.MultiIndex.from_tuples(list(results.keys()), names=['ETF', 'PC'])
    r2_df = pd.DataFrame(results, index=pc_df.index, columns=columns)

    return r2_df


def generate_buy_signals_fixed(prices_df, clusters_model, lookback, test_returns,
                               pc_scaled, r2_df, positive_pc_thresh=0.3,
                               negative_pc_thresh=-0.3, high_r2_thresh=0.5,
                               low_r2_thresh=0.2):
    """
    Generate buy signals based on clusters and PC correlations.
    Fixed version that handles pandas Series properly.

    Parameters:
    -----------
    prices_df : DataFrame
        Full price DataFrame (used for returns)
    clusters_model : KMeans
        Trained KMeans object
    lookback : int
        Lookback period for KMeans
    test_returns : DataFrame
        Returns DataFrame for test period
    pc_scaled : DataFrame
        Scaled PC scores for test period
    r2_df : DataFrame
        MultiIndex DataFrame of rolling R² for each (ETF, PC) pair
    positive_pc_thresh : float
        PC score threshold to consider positive
    negative_pc_thresh : float
        PC score threshold to consider negative
    high_r2_thresh : float
        R² threshold for "high" correlation
    low_r2_thresh : float
        R² threshold for "low" correlation

    Returns:
    --------
    buy_signals : DataFrame
        DataFrame with buy signals (1=buy, 0=no signal)
    signal_details : DataFrame
        DataFrame with details about what triggered each signal
    """
    buy_signals = pd.DataFrame(0, index=test_returns.index, columns=test_returns.columns)
    signal_details = []

    # Compute returns for cluster assignment
    all_returns = prices_df.pct_change().dropna()

    for i, date in enumerate(test_returns.index):
        if i < lookback:
            continue  # not enough data for lookback

        # Check if we have R² data for this date
        if date not in r2_df.index:
            continue

        # Flatten last `lookback` days for KMeans
        try:
            # Get the correct date range for lookback
            lookback_start_idx = all_returns.index.get_loc(date) - lookback
            lookback_end_idx = all_returns.index.get_loc(date)

            if lookback_start_idx < 0:
                continue

            window_data = all_returns.iloc[lookback_start_idx:lookback_end_idx]
            window = window_data.values.flatten().reshape(1, -1)

            cluster_id = clusters_model.predict(window)[0]
        except (KeyError, IndexError):
            continue

        # Get PC scores for this date
        if date not in pc_scaled.index:
            continue

        pcs_today = pc_scaled.loc[date]

        # Only generate signals if PC1 is positive enough
        if pcs_today['PC1'] <= positive_pc_thresh:
            continue

        # Identify positive and negative PCs
        pos_pcs = [pc for pc in pcs_today.index if pcs_today[pc] > positive_pc_thresh]
        neg_pcs = [pc for pc in pcs_today.index if pcs_today[pc] < negative_pc_thresh]

        # Check each ETF
        for etf in test_returns.columns:
            try:
                # Get R² values for this ETF with all PCs
                etf_r2 = r2_df.loc[date, etf]  # This gives us R² for all PCs for this ETF

                # Check conditions using proper pandas operations
                # High R² with at least one positive PC
                high_r2_pos = False
                if pos_pcs:
                    high_r2_pos = (etf_r2[pos_pcs] > high_r2_thresh).any()

                # Low R² with all negative PCs
                low_r2_neg = True
                if neg_pcs:
                    low_r2_neg = (etf_r2[neg_pcs] < low_r2_thresh).all()

                if high_r2_pos and low_r2_neg:
                    buy_signals.loc[date, etf] = 1

                    # Store signal details for debugging
                    signal_details.append({
                        'date': date,
                        'etf': etf,
                        'cluster': cluster_id,
                        'pc1_score': pcs_today['PC1'],
                        'positive_pcs': pos_pcs,
                        'negative_pcs': neg_pcs,
                        'high_r2_pcs': [pc for pc in pos_pcs if etf_r2[pc] > high_r2_thresh],
                        'low_r2_pcs': [pc for pc in neg_pcs if etf_r2[pc] < low_r2_thresh]
                    })

            except (KeyError, IndexError):
                continue

    signal_details_df = pd.DataFrame(signal_details)
    return buy_signals, signal_details_df


def plot_buy_signals_enhanced(buy_signals, test_returns, signal_details_df=None):
    """
    Enhanced plotting of buy signals with optional signal details.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Buy signals over time
    for i, etf in enumerate(buy_signals.columns):
        signal_dates = buy_signals.index[buy_signals[etf] == 1]
        if len(signal_dates) > 0:
            ax1.scatter(signal_dates, [i] * len(signal_dates),
                        marker='^', s=60, alpha=0.7, label=etf)

    ax1.set_title("Buy Signals Over Time (Test Period)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("ETF Index")
    ax1.set_yticks(range(len(buy_signals.columns)))
    ax1.set_yticklabels(buy_signals.columns)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Signal frequency by ETF
    signal_counts = buy_signals.sum()
    signal_counts.plot(kind='bar', ax=ax2, color='green', alpha=0.7)
    ax2.set_title("Total Buy Signals by ETF")
    ax2.set_xlabel("ETF")
    ax2.set_ylabel("Number of Buy Signals")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_signals = buy_signals.sum().sum()
    print(f"\n=== Buy Signal Summary ===")
    print(f"Total buy signals generated: {total_signals}")
    print(f"Signals per ETF:")
    for etf, count in signal_counts.items():
        print(f"  {etf}: {count}")

    if signal_details_df is not None and len(signal_details_df) > 0:
        print(f"\nCluster distribution of signals:")
        print(signal_details_df['cluster'].value_counts().sort_index())


def generate_sell_signals(buy_signals, prices_df, clusters_model, lookback, test_returns,
                          pc_scaled, r2_df, positive_pc_thresh=0.3, high_r2_thresh=0.5):
    """
    Generate sell signals for existing positions.
    Sell when the cluster no longer shows predictive growth patterns.

    Parameters:
    -----------
    buy_signals : DataFrame
        DataFrame with buy signals (1=buy, 0=no signal)
    ... (other params same as generate_buy_signals_fixed)

    Returns:
    --------
    sell_signals : DataFrame
        DataFrame with sell signals (1=sell, 0=no signal)
    """
    sell_signals = pd.DataFrame(0, index=test_returns.index, columns=test_returns.columns)
    all_returns = prices_df.pct_change().dropna()

    # Track current positions for each ETF
    positions = {etf: False for etf in test_returns.columns}

    for i, date in enumerate(test_returns.index):
        if i < lookback:
            continue

        # Check if we have R² data for this date
        if date not in r2_df.index:
            continue

        # Get current cluster assignment
        try:
            lookback_start_idx = all_returns.index.get_loc(date) - lookback
            lookback_end_idx = all_returns.index.get_loc(date)

            if lookback_start_idx < 0:
                continue

            window_data = all_returns.iloc[lookback_start_idx:lookback_end_idx]
            window = window_data.values.flatten().reshape(1, -1)
            current_cluster = clusters_model.predict(window)[0]
        except (KeyError, IndexError):
            continue

        # Update positions based on buy signals
        for etf in test_returns.columns:
            if buy_signals.loc[date, etf] == 1 and not positions[etf]:
                positions[etf] = True

        # Check for sell conditions on existing positions
        if date not in pc_scaled.index:
            continue

        pcs_today = pc_scaled.loc[date]

        for etf in test_returns.columns:
            if positions[etf]:  # We have a position in this ETF
                try:
                    etf_r2 = r2_df.loc[date, etf]

                    # Sell conditions:
                    # 1. PC1 is no longer significantly positive
                    pc1_declining = pcs_today['PC1'] <= positive_pc_thresh

                    # 2. High R² PCs are no longer positive or R² has declined
                    pos_pcs = [pc for pc in pcs_today.index if pcs_today[pc] > positive_pc_thresh]
                    high_r2_declining = True

                    if pos_pcs:
                        high_r2_declining = not (etf_r2[pos_pcs] > high_r2_thresh).any()

                    # Generate sell signal if either condition is met
                    if pc1_declining or high_r2_declining:
                        sell_signals.loc[date, etf] = 1
                        positions[etf] = False  # Close position

                except (KeyError, IndexError):
                    continue

    return sell_signals


def calculate_trading_performance(buy_signals, sell_signals, test_returns, prices_df):
    """
    Calculate comprehensive trading performance metrics.
    Assumes one share per ETF with proper buy/sell timing.
    """
    # Get aligned price data for test period
    test_prices = prices_df.loc[test_returns.index]

    trades = []
    positions = {etf: None for etf in test_returns.columns}  # None = no position, dict = position info
    total_capital_deployed = 0

    # Process all trading signals chronologically
    for date in test_returns.index:
        for etf in test_returns.columns:
            # Process buy signals
            if buy_signals.loc[date, etf] == 1 and positions[etf] is None:
                buy_price = test_prices.loc[date, etf]
                positions[etf] = {
                    'buy_date': date,
                    'buy_price': buy_price,
                    'shares': 1
                }
                total_capital_deployed += buy_price

            # Process sell signals
            elif sell_signals.loc[date, etf] == 1 and positions[etf] is not None:
                sell_price = test_prices.loc[date, etf]
                position = positions[etf]

                # Calculate trade performance
                trade_return = (sell_price - position['buy_price']) / position['buy_price']
                trade_profit = sell_price - position['buy_price']
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
                    'annualized_return': (trade_return + 1) ** (365 / days_held) - 1 if days_held > 0 else 0
                })

                positions[etf] = None  # Close position

    # Handle any remaining open positions (mark-to-market at end)
    final_date = test_returns.index[-1]
    for etf, position in positions.items():
        if position is not None:
            final_price = test_prices.loc[final_date, etf]
            trade_return = (final_price - position['buy_price']) / position['buy_price']
            trade_profit = final_price - position['buy_price']
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
                'status': 'OPEN'
            })

    if not trades:
        print("No completed trades to analyze")
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)

    # Calculate comprehensive performance metrics
    total_trades = len(trades_df)
    profitable_trades = (trades_df['return_pct'] > 0).sum()
    total_profit = trades_df['profit_dollar'].sum()
    total_return_pct = trades_df['return_pct'].mean()
    win_rate = profitable_trades / total_trades

    # Calculate other metrics
    avg_profit_per_trade = total_profit / total_trades
    avg_return_per_trade = total_return_pct
    median_return = trades_df['return_pct'].median()
    best_trade = trades_df['return_pct'].max()
    worst_trade = trades_df['return_pct'].min()
    avg_days_held = trades_df['days_held'].mean()

    # Risk metrics
    return_std = trades_df['return_pct'].std()
    sharpe_ratio = avg_return_per_trade / return_std if return_std > 0 else 0

    # Profit factor (gross profit / gross loss)
    gross_profit = trades_df[trades_df['profit_dollar'] > 0]['profit_dollar'].sum()
    gross_loss = abs(trades_df[trades_df['profit_dollar'] < 0]['profit_dollar'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Maximum drawdown (simplified)
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
    """
    Print a comprehensive performance report.
    """
    print("\n" + "=" * 80)
    print("                        TRADING PERFORMANCE REPORT")
    print("=" * 80)

    # Overall Performance
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {performance_metrics['total_trades']}")
    print(f"   Profitable Trades: {performance_metrics['profitable_trades']}")
    print(f"   Win Rate: {performance_metrics['win_rate']:.2%}")
    print(f"   Total Profit: ${performance_metrics['total_profit_dollar']:,.2f}")
    print(f"   Average Return per Trade: {performance_metrics['avg_return_per_trade']:.2%}")
    print(f"   Median Return per Trade: {performance_metrics['median_return']:.2%}")

    # Risk Metrics
    print(f"\n📈 RISK METRICS:")
    print(f"   Return Standard Deviation: {performance_metrics['return_std']:.2%}")
    print(f"   Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
    print(f"   Maximum Drawdown: {performance_metrics['max_drawdown']:.2%}")
    print(f"   Profit Factor: {performance_metrics['profit_factor']:.2f}")

    # Trade Details
    print(f"\n⏱️  TRADE DETAILS:")
    print(f"   Average Days Held: {performance_metrics['avg_days_held']:.1f}")
    print(f"   Best Trade: {performance_metrics['best_trade']:.2%}")
    print(f"   Worst Trade: {performance_metrics['worst_trade']:.2%}")
    print(f"   Annualized Return: {performance_metrics['annualized_return']:.2%}")

    # Capital Deployment
    print(f"\n💰 CAPITAL METRICS:")
    print(f"   Total Capital Deployed: ${performance_metrics['total_capital_deployed']:,.2f}")
    print(f"   Average Profit per Trade: ${performance_metrics['avg_profit_per_trade']:,.2f}")

    # Performance by ETF
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
    """
    Plot comprehensive trading results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Buy and Sell Signals Timeline
    for i, etf in enumerate(buy_signals.columns):
        buy_dates = buy_signals.index[buy_signals[etf] == 1]
        sell_dates = sell_signals.index[sell_signals[etf] == 1]

        if len(buy_dates) > 0:
            ax1.scatter(buy_dates, [i] * len(buy_dates),
                        marker='^', s=60, color='green', alpha=0.7)
        if len(sell_dates) > 0:
            ax1.scatter(sell_dates, [i] * len(sell_dates),
                        marker='v', s=60, color='red', alpha=0.7)

    ax1.set_title("Buy (↑) and Sell (↓) Signals Timeline")
    ax1.set_ylabel("ETF")
    ax1.set_yticks(range(len(buy_signals.columns)))
    ax1.set_yticklabels(buy_signals.columns)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Trade Returns Distribution
    if len(trades_df) > 0:
        ax2.hist(trades_df['return_pct'] * 100, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(trades_df['return_pct'].mean() * 100, color='red', linestyle='--',
                    label=f'Mean: {trades_df["return_pct"].mean():.2%}')
        ax2.set_title("Distribution of Trade Returns")
        ax2.set_xlabel("Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Performance
    if len(trades_df) > 0:
        trades_df_sorted = trades_df.sort_values('buy_date')
        cumulative_returns = (1 + trades_df_sorted['return_pct']).cumprod()
        ax3.plot(range(len(cumulative_returns)), cumulative_returns, marker='o', linewidth=2)
        ax3.set_title("Cumulative Performance (Trade by Trade)")
        ax3.set_xlabel("Trade Number")
        ax3.set_ylabel("Cumulative Return Multiple")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5)

    # Plot 4: Performance by ETF
    if len(trades_df) > 0:
        etf_returns = trades_df.groupby('etf')['return_pct'].mean()
        colors = ['green' if x > 0 else 'red' for x in etf_returns.values]
        etf_returns.plot(kind='bar', ax=ax4, color=colors, alpha=0.7)
        ax4.set_title("Average Return by ETF")
        ax4.set_xlabel("ETF")
        ax4.set_ylabel("Average Return")
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Complete Trading System Implementation
# -----------------------------

# First, compute R² for all ETF-PC pairs properly
print("Computing rolling R² for all ETF-PC combinations...")
r2_all_etfs = compute_rolling_r2_all_etfs(returns, full_pc_df, window=63)

# Filter to test period only
r2_test_aligned = r2_all_etfs.loc[test_returns.index]

# Generate buy signals with the fixed function
print("Generating buy signals...")
buy_signals_fixed, buy_signal_details = generate_buy_signals_fixed(
    prices_df=data,
    clusters_model=kmeans_15,  # 4-day lookback model
    lookback=14,
    test_returns=test_returns,
    pc_scaled=test_pc_scaled,
    r2_df=r2_test_aligned,
    positive_pc_thresh=0.3,
    negative_pc_thresh=-0.3,
    high_r2_thresh=0.5,
    low_r2_thresh=0.2
)

# Generate sell signals
print("Generating sell signals...")
sell_signals_fixed = generate_sell_signals(
    buy_signals=buy_signals_fixed,
    prices_df=data,
    clusters_model=kmeans_15,
    lookback=14,
    test_returns=test_returns,
    pc_scaled=test_pc_scaled,
    r2_df=r2_test_aligned,
    positive_pc_thresh=0.3,
    high_r2_thresh=0.5
)

# Calculate comprehensive trading performance
print("Calculating trading performance...")
trades_df, performance_metrics = calculate_trading_performance(
    buy_signals_fixed,
    sell_signals_fixed,
    test_returns,
    data
)

# Print comprehensive performance report
print_performance_report(trades_df, performance_metrics)

# Plot comprehensive trading results
if len(trades_df) > 0:
    test_prices = data.loc[test_returns.index]
    plot_trading_results(buy_signals_fixed, sell_signals_fixed, trades_df, test_returns, test_prices)
else:
    print("No trades executed - cannot generate trading plots")

# Enhanced signal analysis
plot_buy_signals_enhanced(buy_signals_fixed, test_returns, buy_signal_details)

# Show signal details
if len(buy_signal_details) > 0:
    print(f"\n=== BUY SIGNAL DETAILS ===")
    print(f"Total buy signals: {len(buy_signal_details)}")
    print(buy_signal_details.head())
else:
    print("No buy signals were generated with current parameters")

# Summary statistics
total_buy_signals = buy_signals_fixed.sum().sum()
total_sell_signals = sell_signals_fixed.sum().sum()

print(f"\n=== TRADING SUMMARY ===")
print(f"Total buy signals: {total_buy_signals}")
print(f"Total sell signals: {total_sell_signals}")
print(f"Buy signals shape: {buy_signals_fixed.shape}")
print(f"Sell signals shape: {sell_signals_fixed.shape}")

if len(trades_df) > 0:
    print(f"Completed trades: {len(trades_df)}")
    print(f"Average trade duration: {trades_df['days_held'].mean():.1f} days")

    # Check for open positions properly
    if 'status' in trades_df.columns:
        open_positions = len(trades_df[trades_df['status'] == 'OPEN'])
        if open_positions > 0:
            print(f"Open positions at end: {open_positions}")
        else:
            print("All positions were closed during the test period")
    else:
        print("All positions were closed during the test period")
else:
    print("No completed trades with current parameters")