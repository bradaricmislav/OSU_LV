# Zadatak 4.5.2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku ˇ
# varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategori ˇ ckih ˇ
# velicina. Radi jednostavnosti nemojte skalirati ulazne veli ˇ cine. Komentirajte dobivene rezultate. ˇ
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
# vozila radi?

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import max_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# ✔ ispravno: Fuel Type nije numerički feature
numeric_features = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)'
]

x = data[numeric_features + ['Fuel Type']]
y = data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

# ispravi split (lakše i ispravno)
X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]

X_train_cat = X_train[['Fuel Type']]
X_test_cat = X_test[['Fuel Type']]



# OneHotEncoding
ohe = OneHotEncoder(handle_unknown='ignore')

X_cat_train_enc = ohe.fit_transform(X_train_cat).toarray()
X_cat_test_enc = ohe.transform(X_test_cat).toarray()

# spajanje feature-a
X_train_final = np.hstack((X_train_num.values, X_cat_train_enc))
X_test_final = np.hstack((X_test_num.values, X_cat_test_enc))

# model
model = lm.LinearRegression()
model.fit(X_train_final, y_train)

# predikcija
y_pred = model.predict(X_test_final)

# evaluacija
ME = max_error(y_test, y_pred)
print(f"Max Error: {ME:.2f} g/km")

# najveća pogreška
errors = np.abs(y_test.values - y_pred)
max_idx = np.argmax(errors)

print("\nVozilo s najvećom pogreškom:")
print(f"Stvarna emisija: {y_test.values[max_idx]:.2f}")
print(f"Predikcija:      {y_pred[max_idx]:.2f}")
print(f"Pogreška:        {errors[max_idx]:.2f}")


# Analiza dobivenih rješenja

# Model linearne regresije koristi numeričke varijable zajedno s kategorijskom varijablom Fuel Type koja je enkodirana pomoću 1-od-K kodiranja. 
# Time se omogućuje da model uzme u obzir različite tipove goriva prilikom procjene emisije CO2.

# Rezultati pokazuju da model uspijeva aproksimirati trend emisije, ali postoje odstupanja kod pojedinih vozila.

# Kritički osvrt

# Najveće pogreške javljaju se kod vozila s ekstremnim vrijednostima potrošnje ili emisije, što ukazuje da linearni model ne može u potpunosti opisati sve odnose među varijablama. 
# Također, utjecaj tipa goriva je ograničen u odnosu na potrošnju i veličinu motora.

# Moguća poboljšanja

# Točnost modela mogla bi se poboljšati korištenjem složenijih modela poput nelinearnih regresijskih metoda ili modela temeljenih na stablima odlučivanja. 
# Također, dodatna analiza podataka i inženjering značajki (feature engineering) mogli bi poboljšati kvalitetu predikcije.