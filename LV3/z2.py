# Zadatak 3.4.2 Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije: ´
# a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´
# b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
# C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
# velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ
# c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
# goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´
# d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
# groupby.
# e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
# broj cilindara.

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#a)
plt.hist(data["CO2 Emissions (g/km)"], bins=30)

plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frekvencija")
plt.title("Distribucija CO2 emisije")

plt.show()

#b)
for fuel, group in data.groupby("Fuel Type"):
    plt.scatter(group["Fuel Consumption City (L/100km)"],
                group["CO2 Emissions (g/km)"],
                label=fuel)
plt.xlabel("Gradska potrosnja (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos potrosnje i CO2 emisije")
plt.legend()

plt.show()

#c)
data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")

plt.title("Izvangradska potrosnja po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Potrosnja (L/100km)")

plt.show()

#d)
fuel_counts = data.groupby("Fuel Type").size()
fuel_counts.plot(kind="bar")

plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.title("Broj vozila po tipu goriva")

plt.show()

#e)
avg_co2 = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
avg_co2.plot(kind="bar")

plt.xlabel("Broj cilindara")
plt.ylabel("Prosjecna CO2 emisija")
plt.title("CO2 emisija prema broju cilindara")

plt.show()