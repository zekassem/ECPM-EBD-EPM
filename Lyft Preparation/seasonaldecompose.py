import numpy as np
import pandas as pd
import statsmodels.api as smapi
import statsmodels as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

path = r'c:/Users/runger/Desktop/Courses/ClassSpring2022/Beamer/'

# mydata = smapi.datasets.sunspots.load_pandas()
# y = pd.DataFrame(dataobj.data)['SUNACTIVITY']
mydata = smapi.datasets.co2.load_pandas()
y= mydata.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
# ETS Decomposition
result = seasonal_decompose(y, model='multiplicative')
# ETS plot
result.plot()
