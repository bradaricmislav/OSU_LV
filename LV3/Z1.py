# Zadatak 3.4.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
# Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
# a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
# duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
# category.
# b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
# ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja. ˇ
# c) Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
# plinova za ova vozila?
# d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
# plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara? ˇ
# e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
# broj cilindara?
# f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
# g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? ´
# h) Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)? ˇ
# i) Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat.

import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

# a)
print(f"Broj mjerenja: {data.shape[0]}")
print(f"Tipovi:\n{data.dtypes}")
print(f"Izostale vrijednosti:\n{data.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data.duplicated().sum()}")
data.drop_duplicates()

data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category")
data["Vehicle Class"] = data["Vehicle Class"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")
data["Fuel Type"] = data["Fuel Type"].astype("category")

#b)
print("Najveća potrošnja u gradu:")
print(data.nlargest(3, "Fuel Consumption City (L/100km)")[["Make","Model","Fuel Consumption City (L/100km)"]])

print("Najmanja potrošnja u gradu:")
print(data.nsmallest(3, "Fuel Consumption City (L/100km)")[["Make","Model","Fuel Consumption City (L/100km)"]])

#c)
filtered = data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)]
print(f"Broj vozila s s veličinom motora između 2.5 i 3.5L: {len(filtered)}")
print(f"Prosječna CO2 emisija plinova: {filtered["CO2 Emissions (g/km)"].mean()}")

#d)
audi = data[data["Make"] == "Audi"]
print(f"Broj Audi vozila: {len(audi)}")
audi4cylinders = audi[audi["Cylinders"] == 4]
print(f"Prosječna CO2 emisija za Audi vozila s 4 cilindra: {audi4cylinders["CO2 Emissions (g/km)"].mean()}")

#e) 
print(f"Broj vozila po cilindrima:")
print(data["Cylinders"].value_counts())
print(data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())

#f)
diesel = data[data["Fuel Type"] == "D"]
gasoline = data[data["Fuel Type"] == "X"]

print(f"Prosjecna gradska potrosnja dizela: {diesel["Fuel Consumption City (L/100km)"].mean()}")
print(f"Medijan dizela: {diesel["Fuel Consumption City (L/100km)"].median()}")

print(f"Prosjecna gradska potrosnja benzinca: {gasoline["Fuel Consumption City (L/100km)"].mean()}")
print(f"Medijan benzinca: {gasoline["Fuel Consumption City (L/100km)"].median()}")

#g)
diesel4cylinders = data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")]
print(f"Dizel s 4 cilindra koji najvise trosi:\n{diesel4cylinders.nlargest(1, "Fuel Consumption City (L/100km)")[["Make", "Model", "Fuel Consumption City (L/100km)"]]}")

#h)
manual = data[data["Transmission"].str.startswith("M")]
print(f"Broj vozila s rucnim mjenjacem: {len(manual)}")

#i)
print(f"Korelacija:\n{data.corr(numeric_only=True)}")