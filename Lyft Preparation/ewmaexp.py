import numpy as np
import pandas as pd
import statsmodels.api as smapi
import statsmodels as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
path = r'c:/Users/runger/Desktop/Courses/ClassSpring2022/Beamer/'

# mydata = smapi.datasets.sunspots.load_pandas()
# y = pd.DataFrame(dataobj.data)['SUNACTIVITY']
# CO2 monthly
# mydata = smapi.datasets.co2.load_pandas()
# y = mydata.data
# y = y['co2'].resample('MS').mean()
# y = y.fillna(y.bfill())
per=365

#Manning log page views
filename =r'c:/Users/runger/Desktop/Manning_playoff_superbowl.csv'
y = pd.read_csv(filename)
y= y['views']

l1 = 0.05
l2 = 0.2
l3 = 0.6
fit1 = SimpleExpSmoothing(y, initialization_method = "heuristic").fit(smoothing_level=l1)
fit2 = SimpleExpSmoothing(y, initialization_method = "heuristic").fit(smoothing_level=l2)
fit3 = SimpleExpSmoothing(y,initialization_method = "heuristic").fit(smoothing_level=l3)
fit4 = SimpleExpSmoothing(y, initialization_method = "estimated").fit()
# print("Estimate is ", str(fit4.model.params["smoothing_level"]))
h= 60
s = len(y) - h
e = len(y)
plt.figure()
plt.plot(y[s:e], marker="o", color="blue")
plt.plot(fit1.fittedvalues[s:e], marker="+", color="red", label = "Smoothing " + str(l1))
plt.plot(fit2.fittedvalues[s:e], marker="+", color="green", label = "Smoothing " + str(l2))
plt.plot(fit3.fittedvalues[s:e], marker="+", color="black", label = "Smoothing " + str(l3))
plt.legend()
plt.title("Simple Exponential Smoothing")

l1 = 0.05
l2 = 0.2
l3 = 0.2
t1 =0.1
t2=0.2
t3=0.8
fit1 = Holt(y, initialization_method = "estimated").fit(smoothing_level=l1, smoothing_trend=t1)
fit2 = Holt(y, initialization_method = "estimated").fit(smoothing_level=l2, smoothing_trend=t2)
fit3 = Holt(y,initialization_method = "estimated").fit(smoothing_level=l3, smoothing_trend=t3)

plt.figure()
plt.plot(y[s:e], marker="o", color="blue")
plt.plot(fit1.fittedvalues[s:e], marker="+", color="red", label = "Smoothing " + str(l1) + ", Trend " + str(t1))
plt.plot(fit2.fittedvalues[s:e], marker="+", color="green", label = "Smoothing " + str(l2) + ", Trend " + str(t2))
plt.plot(fit3.fittedvalues[s:e], marker="+", color="black", label = "Smoothing " + str(l3) + ", Trend " + str(t3))
plt.legend()
plt.title("Holt's Method)")


# import numpy as np
# rng = np.random.default_rng(0)
# e = rng.normal(0, 1, 500)
# ETS Decomposition
# result = seasonal_decompose(y, model='multiplicative')
result = STL(y, period=per, trend_deg=1, robust = False).fit()
# ETS plot
result.plot()

plt.figure()
stlf = STLForecast(y, ARIMA, period = per, model_kwargs=dict(order=(1, 1, 0), trend="t"))
stlf_result = stlf.fit()
forecast = stlf_result.forecast(96)
plt.plot(y)
plt.plot(forecast)
plt.show()

# example with anomaly from [s, e]
s = 200
e = 250
yanon = np.concatenate([y[0:s], 1.1*y[s:e], y[e:]])
result = STL(yanon, period=12, trend = 51, trend_deg=2, robust = False).fit()
# ETS plot
result.plot()