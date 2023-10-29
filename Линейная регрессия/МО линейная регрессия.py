import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("FuelConsumption.csv")
df.head()

df.describe()
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head()
msk = np.random.rand(len(df)) > 0.8
train = cdf[msk]
test = cdf [~msk]

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

plt.scatter(train. ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],'-r')
plt.xlabel('EngineSize')
plt.ylabel('Emission')
plt.show()
y_pred = regr.predict (train_x)
print("предсказанное начение",y_pred, sep = '\n')


