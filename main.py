# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sleep
# %%
from ast import Param
from random import seed
from sqlite3 import paramstyle
from numpy import disp
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# %%
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200

# %%
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# %%
out = "AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}"

# %%
# https://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders
df = pd.read_csv(".lesson/assets/Daily_Demand_Forecasting_Orders.csv", sep=";")
df.rename(
    columns={df.columns[0]: "week", df.columns[10]: "banking_orders"}, inplace=True
)

# %%
plt.plot(df.banking_orders, label='actuals')
plt.legend()
plt.savefig(directory + "/banking_orders")

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
model = ARIMA(
    df.banking_orders, order=(3, 0, 0), enforce_stationarity=False
)  # if enforced, then we cannot constrain parameters
results = model.fit()
plt.plot(df.banking_orders, label="actuals")
plt.plot(results.fittedvalues, color="red", label="AR")
plt.legend()
plt.savefig(directory + "/AR_unconstrained")
print(results.params)

# %%
with model.fix_params({"ar.L1": 0}):
    results = model.fit()
    plt.plot(df.banking_orders, label="actuals")
    plt.plot(results.fittedvalues, color="purple", label="AR constrained")
    plt.legend()
    plt.savefig(directory + "/AR_constrained")

# %%
plot_acf(results.resid)

# %% p-value not zero: we cannot reject the H0 (H0=data does not exhibit serial correlation)
sm.stats.acorr_ljungbox(results.resid, lags=[10], return_df=True)

# %% Forecast
df.banking_orders.corr(results.fittedvalues)
print(results.forecast())  # one-step ahead
print(results.forecast(steps=2))

# %%
plt.plot(df.banking_orders, color="blue", label="actuals")
plt.plot(
    pd.concat([results.fittedvalues, results.forecast(steps=30)]),
    color="red",
    label="AR forecast",
)
plt.legend()
plt.savefig(directory + "/AR_forecast")

# %%
plot_acf(df.banking_orders)

# %% MA
model = ARIMA(
    df.banking_orders, order=(0, 0, 9), enforce_invertibility=False
)  # invertibility prevents constraining parameters later on
results = model.fit()
plt.plot(df.banking_orders, label="actuals")
plt.plot(results.fittedvalues, color="orange", label="MA unconstrained")
plt.legend()
plt.savefig(directory + "/MA_unconstrained")
print(results.params)

# %%
with model.fix_params(
    {"ma.L1": 0, "ma.L2": 0, "ma.L4": 0, "ma.L5": 0, "ma.L6": 0, "ma.L7": 0, "ma.L8": 0}
):
    results = model.fit()
    plt.plot(df.banking_orders, label="actuals")
    plt.plot(results.fittedvalues, color="black", label="MA constrained")
    plt.legend()
    plt.savefig(directory + "/MA_constrained")
    print(results.params)

# %%
print(results.forecast())  # one-step ahead
print(results.forecast(steps=2))
print(results.forecast(steps=20)) # note how the pedictions only last 10 steps because our model is order 9
# %%
plt.plot(df.banking_orders, color="blue", label="actuals")
plt.plot(
    pd.concat([results.fittedvalues, results.forecast(steps=30)]),
    color="red",
    label="MA forecast",
)
plt.legend()
plt.savefig(directory + "/MA_forecast")

# %%
s = 12345
np.random.seed(s)
# %%
arparams = np.array([0.8, -0.4])
maparams = np.array([-0.7])
n=1000
y = sm.tsa.arma_generate_sample(nsample=n, ar=arparams, ma=maparams)
# %%
plt.plot(y)
# %%
plot_acf(y, title=f'ACF simulation seed {s}')
plot_pacf(y, title = f'PACF simulation seed {s}')
# %%
for k in [(1,0,1),(2,0,1)]:
    print(f'{k}')
    model = ARIMA(
        y, order=k, enforce_invertibility=False
    )
    results = model.fit()
    plt.plot(y, label="simulation")
    plt.plot(results.fittedvalues, color="orange", label="ARMA unconstrained")
    plt.title(f'{k} ARMA')
    plt.legend()
    plt.savefig(f'{directory}/simulation_ARMA_unconstrained_{k}')
    plt.clf()
    print(results.param_names)
    print(results.params)
    plot_acf(results.resid, title=f'ACF {k} AMRA')
    plot_pacf(results.resid, title = f'PACF {k} ARMA')
    plt.show()
    
# %%
# %%
