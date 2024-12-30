# By Ali Sarabi (alisarabi@asu.edu)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.models import TCNModel, RNNModel, TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, r2_score


df = pd.read_csv('./stock_value.csv')
df['Date'] = pd.to_datetime(df['Date'])
idx = pd.date_range(start='1/30/2019', end='10/30/2020', freq='D')
# Fill missing Values
df = df.set_index('Date').reindex(idx, method='ffill').reset_index().rename(columns={'index': 'Date'})
pd.set_option('display.max_columns', None)
print(df)

# train ,Validation & test data split
split1 = round(0.6*len(df))
split2 = round(0.8*len(df))

scaler = MinMaxScaler()
scaler.fit(df[['Open','High','Low','Volume']][:split1])
df[['Open','High','Low','Volume']] = scaler.transform(df[['Open','High','Low','Volume']])
scaler2 = MinMaxScaler()
scaler2.fit(np.transpose([df['Close'][:split1]]))
df['Close'] = scaler2.transform(np.transpose([df['Close']]))
# after scaling
print(df)

# Select Features (components)
#features=["Close"]
features=['Open','High','Low','Volume','Close']
lenght_of_interval=13
forecast_horizon=5
n_epochs=150

# We'll use the day as a covariate
day_series = datetime_attribute_timeseries(pd.DatetimeIndex(df['Date']), attribute="weekday", one_hot=True)
scaler_month = Scaler()
day_series = scaler_month.fit_transform(day_series)

# Create training, validation and test sets:
train, validate, test = np.split(df, [split1, split2])
train = TimeSeries.from_dataframe(train, "Date", features)
validate = TimeSeries.from_dataframe(validate, "Date", features)
test = TimeSeries.from_dataframe(test, "Date", features)
train_day, val_day, test_day = day_series[:split1], day_series[split1:split2], day_series[split2:],


model = TCNModel(
    input_chunk_length=lenght_of_interval,      # The number of past periods to be used for prediction
    output_chunk_length=forecast_horizon,     # The forecast horizon (in the backtest) is limited to the output length.
    n_epochs=150,
    dropout=0.1,                # The dropout level will help to simulate different network architectures.
    dilation_base=2,            # The dilation base factor makes the TCN reach out to nodes that are farther back in time.
    weight_norm=True,           # The Boolean parameter weight_norm determines whether or not the TCN will use weight normalization.
    kernel_size=5,              # The kernel size should be set at least as large as the chosen dilation factor.
    num_filters=5,              # The number of filters should reflect the supposed intricacy of the patterns inherent in the time series.
    random_state=0,
)


model.fit(
    series=train,
    past_covariates=train_day,
    val_series=validate,
    val_past_covariates=val_day,
    verbose=True,
)


df_series=TimeSeries.from_dataframe(df, "Date", features)
backtest = model.historical_forecasts(
    series=df_series,
    past_covariates=day_series,
    start=0.1,
    forecast_horizon=forecast_horizon,
    retrain=False,
    verbose=True,
)

df_series["Close"].plot(label="actual")
backtest[str(features.index("Close"))].plot(label="backtest")
plt.legend()
plt.axvline( df['Date'][split2])
plt.show()