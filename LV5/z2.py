# Zadatak 5.5.2 Skripta zadatak_2.py ucitava podatkovni skup Palmer Penguins [1]. Ovaj ˇ
# podatkovni skup sadrži mjerenja provedena na tri razlicite vrste pingvina (’Adelie’, ’Chins- ˇ
# trap’, ’Gentoo’) na tri razlicita otoka u podru ˇ cju Palmer Station, Antarktika. Vrsta pingvina ˇ
# odabrana je kao izlazna velicina i pri tome su klase ozna ˇ cene s cjelobrojnim vrijednostima ˇ
# 0, 1 i 2. Ulazne velicine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’ ˇ flipper_length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
# funkcija plot_decision_region.
# a) Pomocu stup ´ castog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu ˇ
# pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy ˇ
# funkciju unique.
# b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
# taka za ucenje. ˇ
# c) Pronadite u atributima izgra ¯ denog modela parametre modela. Koja je razlika u odnosu na ¯
# binarni klasifikacijski problem iz prvog zadatka?
# d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za u ˇ cenje i ˇ
# izgradeni model logisti ¯ cke regresije. Kako komentirate dobivene rezultate? ˇ
# e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunajte to ˇ cnost. ˇ
# Pomocu c ´ lassification_report funkcije izracunajte vrijednost ˇ cetiri glavne metrike
# na skupu podataka za testiranje.
# f) Dodajte u model još ulaznih velicina. Što se doga ˇ da s rezultatima klasi ¯ fikacije na skupu
# podataka za testiranje?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


#a) 
speciesTrain, countsTrain = np.unique(y_train, return_counts=True)
speciesTest, countsTest = np.unique(y_test, return_counts=True)

plt.bar(speciesTest, countsTest)
plt.figure()
plt.bar(speciesTrain, countsTrain)
plt.show()

#b)
model = LogisticRegression()
model.fit(X_train, y_train)

#c)
print("Intercept:", model.intercept_)
print("Koeficijenti:\n", model.coef_)

#d)
plot_decision_regions(X_train, y_train.ravel(), model)
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.title("Granice odluke (train)")
plt.legend()
plt.show()

#e)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)
accuracy = accuracy_score(y_test, y_pred)
print("Točnost:", accuracy)

print(classification_report(y_test, y_pred, target_names=['Adelie', 'Chinstrap', 'Gentoo']))

#f)
input_variables = [
    'bill_length_mm',
    'flipper_length_mm',
    'bill_depth_mm',
    'body_mass_g'
]

X = df[input_variables].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)

print("Točnost:", accuracy_score(y_test, y_pred))