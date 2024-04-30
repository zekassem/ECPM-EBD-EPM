from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Make tank data
rng = np.random.default_rng(0)
T=1
dt = 0.01
w = rng.normal(0, 1, 500)
b = 1 - np.exp(-dt/T)
x = np.zeros(1)
u= np.zeros(1)
for i in range(1, 200):
    u = np.append(u,x[i-1])
    temp = b*w[i] + (1-b)*x[i-1]
    x = np.append(x, temp)

h = 20
train, test = x[1:len(x)-h], x[len(x)-h:]
model = AutoReg(train, lags=1)
model_fit = model.fit()
print('Coefficients:', model_fit.params)
# make predictions
xhat_train = model_fit.predict(start=0, end=len(train), dynamic=False)
xhat_test = model_fit.predict(start = len(train), end=len(train) + len(test) -1, dynamic=False)
for i in range(len(xhat_test)):
	print('predicted=%f, expected=%f' % (xhat_test[i], test[i]))

# plot results
plt.figure(1)
plt.plot(test, color = 'blue', marker = 'o', label = 'test data')
plt.plot(xhat_test, color='red', marker = '+', label = 'predicted')
plt.legend()
plt.title("Tank AR(1) Model Test Data & Predictions")
plt.title
plt.show()
plt.figure(2)
plt.plot(train, color = 'blue', marker = 'o', label = 'train data')
plt.plot(xhat_train, color='red', marker = '+', label = 'predicted')
plt.legend()
plt.title("Tank AR(1) Model Train Data & Predictions")
plt.show()