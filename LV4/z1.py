from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#a)
numeric_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)']
x = data[numeric_features]
y = data['CO2 Emissions (g/km)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#b)
plt.scatter(x_train['Engine Size (L)'], y_train, c='blue')
plt.scatter(x_test['Engine Size (L)'], y_test, c='red')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Ovisnost emisije CO2 o veličini motora')
plt.legend()
plt.show()

#c)
plt.hist(x_train['Engine Size (L)'])
plt.title('Prije skaliranja (Engine Size)')
plt.show()

sc = MinMaxScaler()
X_train_n = sc.fit_transform(x_train)
X_test_n = sc.transform(x_test)
plt.hist(X_train_n[:, 0])
plt.title('Nakon skaliranja (Engine Size)')
plt.show()

#d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

#e)
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Procjena modela')
plt.title('Odnos stvarnih vrijednosti i procjene')
plt.show()

#f)
print(f"MAE: {mean_absolute_error(y_test, y_test_p)}")
print(f"MSE: {mean_squared_error(y_test, y_test_p)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_test_p)}")
print(f"RMSE: {mean_squared_error(y_test, y_test_p)}")
print(f"R2 score: {r2_score(y_test, y_test_p)}")