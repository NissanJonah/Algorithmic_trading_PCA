import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. Define sector ETFs
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
# 11. PCA Visualization Plots
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
print("GENERATING PCA PLOTS")
print("=" * 60)

print("1. Plotting all PCs over time...")
plot_all_pcs_over_time(full_pc_df)
print("2. Plotting eigenvalues bar chart...")
plot_eigenvalues_bar(pca)
print("\n" + "=" * 60)
print("PCA PLOTS GENERATED SUCCESSFULLY!")
print("=" * 60)

# -----------------------------
# 12. KMeans Clustering with Lookback Periods
# -----------------------------
def create_clusters_with_lookback(returns, pc_scaled, sector_lists, lookbacks=[5, 10, 60], n_clusters=3):
    print("\n" + "=" * 60)
    print("CREATING KMEANS CLUSTERS WITH LOOKBACK PERIODS")
    print("=" * 60)

    # Calculate momentum for each lookback period
    momentum_data = {}
    for lookback in lookbacks:
        print(f"Calculating {lookback}-day momentum...")
        momentum = pd.DataFrame(index=returns.index, columns=returns.columns)
        for i in range(lookback, len(returns)):
            period_returns = returns.iloc[i-lookback:i]
            momentum.iloc[i] = (1 + period_returns).prod() - 1
        momentum_data[f'{lookback}d'] = momentum

    # Combine PC scores with momentum data
    combined_features = pc_scaled.copy()
    for lookback in lookbacks:
        momentum = momentum_data[f'{lookback}d']
        momentum = momentum.loc[combined_features.index]  # Align indices
        for col in momentum.columns:
            combined_features[f'{col}_mom_{lookback}d'] = momentum[col]

    # Drop any rows with NaN values
    combined_features = combined_features.dropna()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(combined_features)

    # Create DataFrame with cluster assignments
    cluster_df = pd.DataFrame({
        'Cluster': clusters
    }, index=combined_features.index)

    # Analyze cluster characteristics
    cluster_stats = {}
    for cluster in range(n_clusters):
        cluster_data = combined_features[clusters == cluster]
        cluster_stats[cluster] = {
            'size': len(cluster_data),
            'mean_pc1': cluster_data['PC1'].mean(),
            'mean_pc2': cluster_data['PC2'].mean() if 'PC2' in cluster_data.columns else np.nan,
            'mean_pc3': cluster_data['PC3'].mean() if 'PC3' in cluster_data.columns else np.nan,
            'avg_momentum': {
                f'{lookback}d': cluster_data[[f'{col}_mom_{lookback}d' for col in returns.columns]].mean().mean()
                for lookback in lookbacks
            }
        }

    # Print cluster analysis
    print("\nCluster Analysis:")
    for cluster, stats in cluster_stats.items():
        print(f"\nCluster {cluster}:")
        print(f"  Size: {stats['size']} days")
        print(f"  Mean PC1: {stats['mean_pc1']:.3f}")
        print(f"  Mean PC2: {stats['mean_pc2']:.3f}")
        print(f"  Mean PC3: {stats['mean_pc3']:.3f}")
        print("  Average Momentum:")
        for lookback, mom in stats['avg_momentum'].items():
            print(f"    {lookback}: {mom:.4f}")

    # Plot clusters over time
    plt.figure(figsize=(14, 7))
    for cluster in range(n_clusters):
        cluster_dates = cluster_df[cluster_df['Cluster'] == cluster].index
        plt.scatter(cluster_dates, [cluster] * len(cluster_dates), label=f'Cluster {cluster}', alpha=0.6)
    plt.title("Cluster Assignments Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2D scatter of clusters (PC1 vs PC2)
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        mask = clusters == cluster
        plt.scatter(combined_features['PC1'][mask], combined_features['PC2'][mask],
                    label=f'Cluster {cluster}', alpha=0.6, s=50)
        # Add 'X' marker at cluster center
        center_pc1 = cluster_stats[cluster]['mean_pc1']
        center_pc2 = cluster_stats[cluster]['mean_pc2']
        plt.scatter([center_pc1], [center_pc2], marker='x', color='black', s=200, linewidths=3,
                    label=f'Cluster {cluster} Center' if cluster == 0 else None, zorder=10)
    plt.title(f"KMeans Clusters in PC1-PC2 Space\n(Clustering includes {lookbacks}-day momentum features)")
    plt.xlabel(f"PC1 (Explained Variance: {explained_var[0]:.2%})")
    plt.ylabel(f"PC2 (Explained Variance: {explained_var[1]:.2%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3D scatter of clusters (PC1 vs PC2 vs PC3)
    if 'PC3' in combined_features.columns:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in range(n_clusters):
            mask = clusters == cluster
            ax.scatter(combined_features['PC1'][mask], combined_features['PC2'][mask], combined_features['PC3'][mask],
                       label=f'Cluster {cluster}', alpha=0.6, s=50, zorder=5)
            # Add 'X' marker at cluster center
            center_pc1 = cluster_stats[cluster]['mean_pc1']
            center_pc2 = cluster_stats[cluster]['mean_pc2']
            center_pc3 = cluster_stats[cluster]['mean_pc3']
            ax.scatter([center_pc1], [center_pc2], [center_pc3], marker='x', color='black', s=200, linewidths=3,
                       label=f'Cluster {cluster} Center' if cluster == 0 else None, zorder=10)
        ax.set_title(f"KMeans Clusters in PC1-PC2-PC3 Space\n(Clustering includes {lookbacks}-day momentum features)")
        ax.set_xlabel(f"PC1 (Explained Variance: {explained_var[0]:.2%})")
        ax.set_ylabel(f"PC2 (Explained Variance: {explained_var[1]:.2%})")
        ax.set_zlabel(f"PC3 (Explained Variance: {explained_var[2]:.2%})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return cluster_df, cluster_stats, momentum_data

# Run clustering with lookback periods
cluster_df, cluster_stats, momentum_data = create_clusters_with_lookback(returns, full_pc_df, sector_lists, lookbacks=[5, 10, 60], n_clusters=3)

print("\n" + "=" * 60)
print("CLUSTERING COMPLETE!")
print("=" * 60)
print(f"Generated {len(cluster_df)} cluster assignments")
print(f"Clusters: {cluster_df['Cluster'].nunique()}")
print("=" * 60)