import numpy as np
import pandas as pd
import statsmodels.api as smapi
import statsmodels as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_model import ARIMAResults as arr

path = r'c:/Users/runger/Desktop/Courses/ClassSpring2022/Beamer/'

# mydata = smapi.datasets.sunspots.load_pandas()
# y = pd.DataFrame(dataobj.data)['SUNACTIVITY']
mydata = smapi.datasets.co2.load_pandas()
y= mydata.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())

plt.figure()
title = 'CO2 Monthly Average from 1958'
plt.title(title)
plt.plot(y)
plt.show()
sm.graphics.tsaplots.plot_acf(y)
sm.graphics.tsaplots.plot_pacf(y, method='ywm')

h = 120 #Test data horizon
train, test = y[0:len(y)-h], y[len(y)-h:]
model = ARIMA(train, order=(1,1,1), seasonal_order= (1,1,1,12))
fit=model.fit()
print(fit.summary)
print('Coefficients:', fit.params)
diag = fit.plot_diagnostics()
xhat_train = fit.predict(start=24, end=len(train), dynamic = False, typ = 'levels')
xhat_test = fit.predict(start = len(train), end=len(train) + len(test) -1, dynamic=True, typ = 'levels')

plt.figure()
title = 'CO2 Test Data'
plt.plot(test, color = 'blue', marker = 'o', label = 'test data')
plt.plot(xhat_test, color='red', marker = '+', label = 'predicted test data')
plt.title(title)
plt.show()

plt.figure()
# plt.savefig(path+ modelinfo + trendinfo)
plt.plot(train, color = 'blue', marker = 'o', label = 'train data')
plt.plot(xhat_train, color='red', marker = '+', label = 'predicted train data')
plt.legend()
title = 'CO2 Train Data'
plt.title(title)
plt.show()


