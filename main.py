# %%
from numpy import disp
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib

plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
# %%
out = "AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}"
# %%
df = pd.read_csv(".lesson/assets/Daily_Demand_Forecasting_Orders.csv", sep=";")
df.rename(
    columns={df.columns[0]: "week", df.columns[10]: "banking_orders"}, inplace=True
)
# %%
df.plot(y="banking_orders")
# %%
plot_pacf(df.banking_orders, lags=20)
pacf_values = pacf(df.banking_orders)
print(pacf_values)
# %% we choose the lag equal to the value of the PACF that crosses the significance threshold
for lags in range(5):
    res = AutoReg(df.banking_orders, lags=lags, seasonal=False).fit()
    print(out.format(res.aic, res.hqic, res.bic))  # low AIC is good
    print(res.params)
# %%
model = ARIMA(df.banking_orders, order=(3, 0, 0))
results = model.fit()
plt.plot(df.banking_orders)
plt.plot(results.fittedvalues, color="red")
print(results.params)
# %%
plot_acf(results.resid)
# %% p-value not zero: we cannot reject the H0 (H0=data does not exhibit serial correlation)
sm.stats.acorr_ljungbox(results.resid, lags=[10], return_df=True)
# %% Forecast
df.banking_orders.corr(results.fittedvalues)
print(results.forecast())  # one-step ahead
print(results.forecast(steps=2))
# %%
plt.plot(df.banking_orders, color="blue")
plt.plot(pd.concat([results.fittedvalues, results.forecast(steps=30)]), color="red")
# %%
