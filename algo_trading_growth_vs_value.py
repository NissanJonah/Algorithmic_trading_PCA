import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# 1. Define ETFs
# -----------------------------
growth_list = {
    "Technology": ["XLK", "VGT", "QQQ", "SPYG", "IWF", "ARKK", "TCHP", "VUG", "MGK", "IUSG", "FTEC", "XITK"]
}

value_list = {
    "Consumer Staples": ["XLP", "VDC", "FSTA", "KXI", "RHS", "IYK", "FXG", "VTV", "SCHV", "IWD", "VOE", "VBR"]
}

market = {
    "Market": ["SPY"]
}

all_etfs = growth_list["Technology"] + value_list["Consumer Staples"] + market["Market"]

# -----------------------------
# 2. Download adjusted close prices
# -----------------------------
prices = yf.download(all_etfs, start="2024-01-01", end="2025-08-01", auto_adjust=True)

# Flatten columns if necessary
if isinstance(prices.columns, pd.MultiIndex):
    prices = prices['Close']
prices.columns = prices.columns.str.strip()

# -----------------------------
# 3. Separate ETFs
# -----------------------------
prices_growth = prices[[t for t in growth_list["Technology"] if t in prices.columns]]
prices_value = prices[[t for t in value_list["Consumer Staples"] if t in prices.columns]]
prices_market = prices[market["Market"]]

# -----------------------------
# 4. Standardize for PCA
# -----------------------------
def normalize(df):
    return (df - df.mean()) / df.std()

prices_growth_std = normalize(prices_growth).dropna()
prices_value_std = normalize(prices_value).dropna()

# -----------------------------
# 5. PCA
# -----------------------------
def compute_pca(prices_std):
    pca = PCA()
    Y = pca.fit_transform(prices_std)  # Scores matrix
    V = pca.components_.T               # Loadings matrix
    pc_df = pd.DataFrame(Y, index=prices_std.index,
                         columns=[f"PC{i+1}" for i in range(Y.shape[1])])
    explained_var = pca.explained_variance_ratio_
    return pc_df, V, explained_var

pc_growth, V_growth, var_growth = compute_pca(prices_growth_std)
pc_value, V_value, var_value = compute_pca(prices_value_std)

print(f"Growth PC1 explains {var_growth[0]:.2%} of variance")
print(f"Value PC1 explains {var_value[0]:.2%} of variance")

# -----------------------------
# 6. Plots
# -----------------------------
# Plot first 2 PCs for growth and value
plt.figure(figsize=(14,6))
plt.plot(pc_growth["PC1"], label="Growth PC1")
plt.plot(pc_growth["PC2"], label="Growth PC2")
plt.title("Growth ETFs - First 2 PCs (Price Series)")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(pc_value["PC1"], label="Value PC1")
plt.plot(pc_value["PC2"], label="Value PC2")
plt.title("Value ETFs - First 2 PCs (Price Series)")
plt.legend()
plt.show()

# -----------------------------
# 7. First derivatives overlay
# -----------------------------
def first_derivative(series):
    return series.diff().dropna()
# -----------------------------
# 7. First derivatives separate plots
# -----------------------------
growth_deriv = first_derivative(pc_growth["PC1"])
value_deriv = first_derivative(pc_value["PC1"])
spy_deriv = first_derivative(prices_market["SPY"])

# Growth PC1 derivative
plt.figure(figsize=(14,6))
plt.plot(growth_deriv, label="Growth PC1 Derivative", color='blue')
plt.title("First Derivative - Growth PC1")
plt.legend()
plt.show()

# Value PC1 derivative
plt.figure(figsize=(14,6))
plt.plot(value_deriv, label="Value PC1 Derivative", color='green')
plt.title("First Derivative - Value PC1")
plt.legend()
plt.show()

# S&P 500 price derivative
plt.figure(figsize=(14,6))
plt.plot(spy_deriv, label="S&P 500 Price Derivative", color='red')
plt.title("First Derivative - S&P 500 Price")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(first_derivative(pc_growth["PC1"]), label="Growth PC1 Derivative")
plt.plot(first_derivative(pc_value["PC1"]), label="Value PC1 Derivative")
plt.plot(first_derivative(prices_market["SPY"]), label="S&P 500 Price Derivative")
plt.title("First Derivative: PC1s vs S&P Prices")
plt.legend()
plt.show()

# S&P daily returns
spy_returns = prices_market["SPY"].pct_change().dropna()

# When Growth PC1 > Value PC1
condition = pc_growth["PC1"] > pc_value["PC1"]

# Percent of days S&P goes up under this condition
percent_up = (spy_returns[condition] > 0).mean() * 100
print(f"S&P goes up {percent_up:.2f}% of the time when Growth PC1 > Value PC1")


plt.figure(figsize=(14,6))
plt.plot(pc_growth["PC1"], label="Growth PC1", color='blue')
plt.plot(pc_value["PC1"], label="Value PC1", color='green')
plt.plot(prices_market["SPY"], label="S&P 500", color='red', alpha=0.5)
plt.title("PC1s vs S&P 500")
plt.legend()
plt.show()
