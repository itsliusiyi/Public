# %% [markdown]
# ## 1. Calculate the Covariance and Correlations

# %% [markdown]
# This code analyzes dynamic correlations between five major asset classes using exponentially weighted moving average (EWMA) applied to 5-year rolling returns. It processes historical data from 1995 to present, calculating time-varying covariance and correlation matrices that emphasize recent market dynamics while maintaining long-term perspective.

# %%
# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta


# %%
# settings
tickers = {
    'US_Stocks': 'SPY',
    'Intl_Stocks': 'EFA', 
    'Bonds': 'AGG',
    'Real_Estate': 'VNQ',
    'Commodities': 'GSG'}
start_date = '1995-01-01'
end_date = (pd.Timestamp.now() - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')
lambda_param=0.98
window=60  # 5y window for monthly data

class EWMA5YearRollingCovariance:
    def __init__(self, lambda_param=lambda_param):
        self.lambda_param = lambda_param
        self.cov_matrix = None
        self.portfolio_names = None
        self.initialized = False
        
    def initialize(self, initial_returns):
        self.portfolio_names = initial_returns.columns.tolist()
        initial_data = initial_returns.dropna()
        if len(initial_data) < 2:
            raise ValueError # need at least two data points to initialize
        initial_data = initial_data.iloc[:2]
        self.cov_matrix = initial_data.cov().values
        self.initialized = True
        return self.cov_matrix
    
    def update_cov_matrix(self, current_returns):
        if not self.initialized:
            self.initialize(current_returns)
            return self.cov_matrix
        if current_returns.isna().any():
            return self.cov_matrix
        r_t = current_returns.values.reshape(-1, 1)
        mean_returns = current_returns.mean()
        deviations = r_t - mean_returns
        outer_product = np.dot(deviations, deviations.T)
        self.cov_matrix = (self.lambda_param * self.cov_matrix + (1 - self.lambda_param)*outer_product)
        return self.cov_matrix
    
    def calculate_cov(self, price_data, window=window):
        dates = []
        cov_matrices = []
        corr_matrices = []

        # calculate the 5y rolling returns
        returns_rolling = (price_data / price_data.shift(window)) - 1

        # find the first valid date
        first_valid_date = returns_rolling.dropna().index[0]
        start_idx = returns_rolling.index.get_loc(first_valid_date)
        print(f"First valid 5y return data point: {first_valid_date}")
        
        # initialize EWMA
        self.initialize(returns_rolling.iloc[start_idx:start_idx+2])
        
        # calculate cov and corr
        for i in range(start_idx, len(returns_rolling)):
            current_date = returns_rolling.index[i]
            current_returns = returns_rolling.iloc[i]
            if current_returns.isna().any():
                continue
            # calculate cov
            cov_matrix = self.update_cov_matrix(current_returns)
            dates.append(current_date)
            cov_matrices.append(cov_matrix.copy())
            # calculate corr
            std_dev = np.sqrt(np.diag(cov_matrix))
            std_dev[std_dev == 0] = 1e-8
            corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
            np.fill_diagonal(corr_matrix, 1.0)
            corr_matrix = np.clip(corr_matrix, -1, 1)
            corr_matrices.append(corr_matrix)
            
        return dates, cov_matrices, corr_matrices, returns_rolling

# download the data
data = yf.download(list(tickers.values()), start=start_date, end=end_date, auto_adjust=False)['Adj Close']
data.columns = list(tickers.keys())
monthly_data = data.resample('M').last()
monthly_data = monthly_data[monthly_data.index >= start_date]
print(f"Raw data period: {monthly_data.index[0]} 到 {monthly_data.index[-1]}")
print(f"Raw data points: {len(monthly_data)}")
portfolio_names = monthly_data.columns.tolist()

# calculate cov and corr
ewma_calc = EWMA5YearRollingCovariance(lambda_param=lambda_param)
dates, cov_matrices, corr_matrices, returns_rolling = ewma_calc.calculate_cov(monthly_data, window=window)


# %% [markdown]
# ## 2. Statistical Summary

# %%
# summarize the statisticals
print("\n" + "="*60)
print("EWMA Covariance Matrix Based on 5-Year Rolling Returns - Statistical Summary")
print("="*60)
print(f"Analysis period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
print(f"Number of portfolios: {len(portfolio_names)}")

# the lastest covariance matrix
print("\nLatest covariance matrix:")
latest_cov = cov_matrices[-1]
latest_cov_df = pd.DataFrame(latest_cov, 
                            index=portfolio_names, 
                            columns=portfolio_names)
print(latest_cov_df.round(4))

# the lastest correlation matrix
print("\nLatest correlation matrix:")
latest_corr = corr_matrices[-1]
latest_corr_df = pd.DataFrame(latest_corr, 
                            index=portfolio_names, 
                            columns=portfolio_names)
print(latest_corr_df.round(4))


# %% [markdown]
# ## 3. Visualize the Results

# %%
# select key pairs
key_pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]

# plot the time-series correlations for key pairs
plt.figure(figsize=(10, 6))
for i, (idx1, idx2) in enumerate(key_pairs):
        # create 2×2 subplots
        plt.subplot(2, 2, i+1) 
        corr_series = [corr_matrix[idx1, idx2] for corr_matrix in corr_matrices]
        plt.plot(dates, corr_series, linewidth=2, label=f'{portfolio_names[idx1]} vs {portfolio_names[idx2]}')
        plt.title(f'{portfolio_names[idx1]} vs {portfolio_names[idx2]}')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # include average correlations
        avg_corr = np.mean(corr_series)
        plt.axhline(y=avg_corr, color='green', linestyle='-', alpha=1.0, label=f'Avg: {avg_corr:.2f}')
        plt.legend()
plt.tight_layout()
plt.suptitle('EWMA Correlation based on 5-Year Rolling Returns', y=1.02)
plt.show()

# plot the correlation matrix heatmap as of the latest date
plt.figure(figsize=(10, 6))
mask = np.triu(np.ones_like(latest_corr, dtype=bool))
sns.heatmap(latest_corr, mask=~mask,
            xticklabels=portfolio_names, 
            yticklabels=portfolio_names,
            annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation'})
plt.title(f'Correlation Matrix as of {end_date}\n(Based on 5-Year Rolling Returns)')
plt.tight_layout()
plt.show()



