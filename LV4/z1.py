# Zadatak 4.5.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
# Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
# laboratorijskoj vježbi.
# a) Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite
# podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%. ˇ
# b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
# o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
# plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom. ˇ
# c) Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
# jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
# transformirajte ulazne velicine skupa podataka za testiranje. ˇ
# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
# povežite ih s izrazom 4.6.
# e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
# pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
# dobivene modelom.
# f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
# skupu podataka za testiranje.
# g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
# ulaznih velicina?


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