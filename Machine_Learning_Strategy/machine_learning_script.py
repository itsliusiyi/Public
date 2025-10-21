# Python

# This quantitative trading strategy employed unsupervised learning to systematically select and weight S&P 500 stocks.
# The process started by downloading historical price data and computing technical indicators including ATR, Bollinger Bands, and RSI.
# After filtering for the top 150 most liquid stocks, the strategy incorporated Fama-French five-factor models to assess risk exposures.
# It selected stocks based on K-Means clustering with RSI-based initialization.
# Using mean-variance optimization, the strategy maximizes the Sharpe ratio while implementing weight constraints to ensure diversification.
# Compared the cumulative returns of this strategy vs. S&P 500.


# %% [markdown]
# ## 1.Download S&P 500 price data

# %%
# import necessary libraries
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np 
import datetime as dt
import yfinance as yf
import pandas_ta
import ssl
import urllib.request
from io import StringIO
import urllib3
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")



# %%
# function to get data from HTML table
def get_html_data(url,number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, headers=headers, verify=False, timeout=50)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[number]
        print(f"Number of Rows: {len(table)}")
        print("All column names:", table.columns.tolist())
        print(table)
        return table
    except Exception as e:
        print(f"Error: {e}")
        return None

# get the S&P 500 data from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
number=0
sp500 = get_html_data(url,number)
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)
symbols_list = sp500['Symbol'].unique().tolist()


# %%
# download the S&P 500 price data from yahoo finance
end_date = (pd.to_datetime(dt.date.today()) - pd.offsets.BDay(1)).date() # last business day
start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=8)).date()  # last 8 years
ssl._create_default_https_context = ssl._create_unverified_context
df = yf.download(tickers=symbols_list, start=start_date, end=end_date, auto_adjust=False).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()


# %% [markdown]
# ## 2.Calculate features and technical indicators for each stock

# %%
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level='ticker', group_keys=False).apply(compute_atr)
df['bb_low'] = df.groupby(level='ticker')['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level='ticker')['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level='ticker')['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
df['garman_klass_vol'] = ((np.log(df['high'] / df['low']))**2)/2-(2*np.log(2)-1)*(np.log(df['adj close']/df['open']))**2
def compute_macd(stock_data):
    macd = pandas_ta.macd(close=stock_data['close'], length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd'] = df.groupby(level='ticker', group_keys=False).apply(compute_macd)
df['rsi'] = df.groupby(level='ticker')['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6


# %% [markdown]
# ## 3. Filter for the most liquid stocks and compute monthly returns

# %%
# resample data to monthly frequency
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open','high', 'low', 'close' ]]
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                   axis=1)).dropna()

# select top 150 liquid stocks by 5-year rolling average of dollar volume
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


# %%
# compute monthly returns with outlier clipping
def compute_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close'].pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x. quantile(1-outlier_cutoff)))
                                                       .add(1).pow(1/lag).sub(1))
    return df
data = data.groupby(level='ticker', group_keys=False).apply(compute_returns).dropna()


# %% [markdown]
# ## 4. Download Fama-Fecher factors and calcute rolling factor betas

# %%
# download Fama-Fecher factors and filter stocks with at least 10 observations
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2012')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data.index.name = 'date'
factor_data = factor_data.resample('M').last().div(100)
factor_data = factor_data.join(data['return_1m']).sort_index()
observations = factor_data.groupby(level='ticker').size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

# calculate rolling factor betas
betas = (factor_data.groupby(level='ticker', group_keys=False).apply(lambda x: RollingOLS(endog=x['return_1m'],
                                                                                exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                                                                window=min(24,x.shape[0]),
                                                                                min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const', axis=1)))

data = (data.join(betas.groupby('ticker').shift()))
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()


# %% [markdown]
# ## 5.Fit a K-Means Clustering Algorithm to group similar assets

# %%
data_1 = data.copy()
# data = data_1.copy()


# %%
# fit a K-Means Clustering Algorithm to group similar assets
if 'cluster' in data.columns:
    data = data.drop('cluster', axis=1)

def get_clusters(df, n_clusters, target_rsi_values):
    initial_centroids = np.zeros((len(target_rsi_values), df.shape[1]))
    initial_centroids[:, df.columns.get_loc('rsi')] = target_rsi_values
    df['cluster'] = KMeans(n_clusters=n_clusters,
                           init=initial_centroids,
                           n_init=1,
                           random_state=0).fit(df).labels_
    return df
n_clusters=4
target_rsi_values = [30, 45, 55, 70]
data = data.dropna().groupby('date', group_keys=False).apply(lambda x: get_clusters(x, n_clusters, target_rsi_values))


# %%
# plot the clusters for each month
def plot_clusters(data):
    cluster_0 = data[data['cluster'] == 0]
    cluster_1 = data[data['cluster'] == 1]
    cluster_2 = data[data['cluster'] == 2]
    cluster_3 = data[data['cluster'] == 3]
    y_col = data.columns.get_loc('rsi')
    x_col = data.columns.get_loc('atr')
    plt.scatter(cluster_0.iloc[:, x_col], cluster_0.iloc[:, y_col], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:, x_col], cluster_1.iloc[:, y_col], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:, x_col], cluster_2.iloc[:, y_col], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:, x_col], cluster_3.iloc[:, y_col], color='black', label='cluster 3')
    plt.legend()
    plt.show()
    return
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().to_list():
    g = data.xs(i, level=0)
    plt.title(f'Date {i}')
    plot_clusters(g)


# %% [markdown]
# ## 6. Select assets based on clusters for eaxh month

# %%
# create a dictionary of stocks for each month
filtered_df = data[data['cluster'] == 3].copy()
filtered_df = filtered_df.reset_index(level='ticker')
filtered_df.index = filtered_df.index+pd.DateOffset(days=1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().to_list()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level='date').index.tolist()

# download price data for the selected stocks
stocks = data.index.get_level_values('ticker').unique().tolist()
new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique().min()-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique().max(),
                     auto_adjust=False)
returns_df = np.log(new_df['Adj Close']).diff()
portfolio_df = pd.DataFrame()

# define portfolio optimization function
def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    cov = risk_models.sample_cov(prices=prices,
                                        frequency=252)
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    weights = ef.max_sharpe()
    return ef.clean_weights()

# perform optimization and calculate portfolio returns
for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
#        print(start_date, end_date, optimization_start_date, optimization_end_date, cols)
        optimization_df = new_df['Adj Close'].loc[optimization_start_date:optimization_end_date][cols]
        success = False
        try:
            weights = optimize_weights(prices=optimization_df,
                                       lower_bound=round(1/(len(optimization_df.columns)*2),3))
            weights = pd.DataFrame(weights, index=pd.Series(0))
            success = True
        except:
            print(f"Optimization failed for {start_date}. Continuing with equal weights.")
        if not success:
            weights = pd.DataFrame([1/len(optimization_df.columns)for i in range(len(optimization_df.columns))],
                                   index=optimization_df.columns.to_list(),
                                   columns=pd.Series(0)).T
        temp_df = returns_df[start_date:end_date]
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                    .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                           left_index=True, right_index=True)\
                    .reset_index().set_index(['Date', 'Ticker']).unstack().stack()
        temp_df.index.names = ['date', 'ticker']
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
        temp_df = temp_df.groupby(level='date')['weighted_return'].sum().to_frame('Strategy Return')
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
    except Exception as e:
        print(e)

portfolio_df = portfolio_df.drop_duplicates()

# # test code for a single month
# e_date = next(iter(fixed_dates))
# s_date = (pd.to_datetime(next(iter(fixed_dates)))-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
# optimization_df = new_df['Adj Close'].loc[s_date:e_date][fixed_dates[e_date]]
# temp_df = returns_df['2019-11-01':'2019-11-30']
# temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
#                 .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
#                        left_index=True, right_index=True)\
#                 .reset_index().set_index(['Date', 'Ticker']).unstack().stack()
# temp_df.index.names = ['date', 'ticker']
# temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
# temp_df = temp_df.groupby(level='date')['weighted_return'].sum().to_frame('Strategy Return')


# %% [markdown]
# ## 7. Visualize portfolio returns and compare to S&P500 returns

# %%
portfolio_df_copy = portfolio_df.copy()
# portfolio_df = portfolio_df_copy.copy()

# portfolio cumulative returns vs S&P 500
spy = yf.download(tickers='SPY',
                  start=portfolio_df.index.min(),
                  end=dt.date.today(),
                  auto_adjust=False)
spy_ret = np.log(spy['Adj Close']).diff().dropna().rename({'Adj Close':'SPY'}, axis=1)
portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='inner')
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1


# %%
# plot cumulative returns
plt.style.use('ggplot')
portfolio_cumulative_return[:'2025-09-30'].plot(figsize=(16,6))
plt.title('Cumulative Returns: Unsupervised Learning Trading Strategy vs S&P 500')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Cumulative Return')
plt.show()


# %% [markdown]
# 


