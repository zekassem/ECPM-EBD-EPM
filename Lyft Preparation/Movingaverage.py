import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(0)
T = 1
t = np.linspace(0, 10, num=50)
v = 5*(1 - np.exp(-t/T))


e = rng.normal(0, 1, 500)
theta = -0.6
x = e - np.roll(e, 1)
x = x[1:]
u = np.roll(x, 1)
# # plt2.savefig()
title = 'MA(1)'+ str(theta)
# plt.figure(1, figsize=(15, 5))
plt.figure(1)
plt.subplot(221)
plt.plot(x, color='blue', marker='+', label='data')
# plt.plot(x, color='black', label='datapoints')
plt.title(title)
plt.subplot(222)
plt.title("Lag 1 Scatter Plot")
plt.scatter(u, x, color='red', linestyle='--', marker = '+', label='signal')
# plt.show()

plt.subplot(223)
plt.title("Autocorrelation Plot")
# Providing x-axis name.
plt.xlabel("Lags")
# Plotting the Autocorrelation plot.
plt.acorr(x, maxlags=20)
# Displaying the plot.
print("The Autocorrelation plot for the data is:")
plt.grid(True)
plt.show()

lags = [1, 2, 5, 10]
plt.figure(2)
for i in range(4):
    r = lags[i]
    plt.subplot(2,2,i+1)
    xroll = np.roll(x,r)[r:]
    xshort = x[r:]
    plt.title('Lag = ' + str(r))
    plt.scatter(xroll, xshort)
plt.show()
