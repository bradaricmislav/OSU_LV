# Zadatak 6.5.1 Skripta zadatak_1.py ucitava ˇ Social_Network_Ads.csv skup podataka [2].
# Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
# Podaci o korisnicima su spol, dob i procijenjena placa. Razmatra se binarni klasi ´ fikacijski
# problem gdje su dob i procijenjena placa ulazne veli ´ cine, dok je kupovina (0 ili 1) izlazna ˇ
# velicina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija ˇ
# plot_decision_region [1]. Podaci su podijeljeni na skup za ucenje i skup za testiranje modela ˇ
# u omjeru 80%-20% te su standardizirani. Izgraden je model logisti ¯ cke regresije te je izra ˇ cunata ˇ
# njegova tocnost na skupu podataka za u ˇ cenje i skupu podataka za testiranje. Potrebno je: ˇ
# 1. Izradite algoritam KNN na skupu podataka za ucenje (uz ˇ K=5). Izracunajte to ˇ cnost ˇ
# klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite ˇ
# dobivene rezultate s rezultatima logisticke regresije. Što primje ˇ cujete vezano uz dobivenu ´
# granicu odluke KNN modela?
# 2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?
# Zadatak 6.5.2 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ K
# algoritma KNN za podatke iz Zadatka 1.
# Zadatak 6.5.3 Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
# te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
# ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje? ˇ
# Mijenjajte tip kernela koji se koristi. Što primjecujete? ´
# Zadatak 6.5.4 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ C i γ
# algoritma SVM za problem iz Zadatka 1.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#Zadatak 1
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)

print("KNN modeL:")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.title("KNN (K=5)")
plt.show()

#Zadatak 1 2. dio 
KNN_model1 = KNeighborsClassifier(n_neighbors = 1)
KNN_model1.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model1)
plt.title("KNN (K=1)")
plt.show()

KNN_model100 = KNeighborsClassifier(n_neighbors = 100)
KNN_model100.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model100)
plt.title("KNN (K=100)")
plt.show()

#Zadatak 2
param_grid = {"n_neighbors": np.arange(1, 101)}

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv=5)
grid.fit(X_train_n, y_train)

print("Najbolji K:", grid.best_params_)
print("Najbolja točnost:", grid.best_score_)

#Zadatak 3 
svm_model = svm.SVC(kernel='rbf', C=1, gamma=0.01)
svm_model.fit(X_train_n, y_train)

y_test_svm = svm_model.predict(X_test_n)

print("\nSVM:")
print("Tocnost test:", accuracy_score(y_test, y_test_svm))

plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.title("SVM (RBF)")
plt.show()

#Zadatak 4
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_n, y_train)

print("Najbolji parametri:", grid.best_params_)
print("Najbolja točnost:", grid.best_score_)

# Analiza dobivenih rješenja

# Najprije je primijenjena logistička regresija kao osnovni model za binarnu klasifikaciju kupovine oglasa. 
# Dobivena točnost na skupu za učenje iznosi 0.825, dok je točnost na testnom skupu 0.900. 
# Budući da je testna točnost čak veća od trening točnosti, ne može se govoriti o overfittingu. 
# Vjerojatnije je riječ o slučajnoj raspodjeli podataka ili nešto lakšem testnom skupu.

# Kod KNN algoritma za K = 5 dobiva se nelinearna granica odluke koja se bolje prilagođava stvarnoj raspodjeli podataka nego logistička regresija. 
# Za razliku od logističke regresije koja daje linearnu granicu odluke, 
# KNN omogućuje fleksibilniju klasifikaciju jer odluku donosi na temelju lokalnog susjedstva podataka.

# Za K = 1 granica odluke postaje vrlo nepravilna i jako prati pojedinačne primjere iz skupa za učenje. 
# Takav model često pokazuje vrlo visoku točnost na trening skupu, ali lošiju generalizaciju na testnom skupu, što ukazuje na overfitting.

# Za K = 100 granica odluke postaje vrlo glatka i pojednostavljena. 
# Model zanemaruje lokalne strukture u podacima te može doći do underfittinga jer previše generalizira.

# Primjenom unakrsne validacije određena je optimalna vrijednost hiperparametra K, 
# čime se smanjuje subjektivan odabir parametra i povećava pouzdanost modela.

# Kod SVM modela s RBF kernelom granica odluke također postaje nelinearna i često preciznija od logističke regresije. 
# Parametar C kontrolira koliko strogo model kažnjava pogreške klasifikacije.
# Veći C vodi složenijoj granici odluke i mogućem overfittingu, dok manji C daje jednostavniju granicu i veću toleranciju na pogreške.

# Parametar γ određuje koliko daleko utječe pojedini podatkovni primjer. 
# Velika vrijednost γ stvara vrlo kompleksnu granicu odluke i povećava rizik prenaučenosti, dok mala vrijednost γ daje glatkiju granicu i stabilniji model.

# GridSearchCV omogućuje pronalazak optimalnih vrijednosti parametara C i γ te poboljšava konačne performanse modela.

# Kritički osvrt na rezultate

# Rezultati pokazuju da logistička regresija daje vrlo dobre rezultate unatoč svojoj jednostavnosti. 
# To sugerira da podaci možda nisu izrazito nelinearno razdvojivi te da linearni model može biti sasvim dovoljan.

# KNN model pokazuje veću osjetljivost na odabir parametra K. 
# Premala vrijednost vodi prenaučenosti, dok prevelika vrijednost vodi podnaučenosti. 
# Zbog toga je pravilno određivanje optimalnog K ključno za uspješnost modela.

# SVM s RBF kernelom često daje najbolje rezultate, ali zahtijeva pažljivo podešavanje hiperparametara. 
# Bez optimizacije parametara model može biti lošiji od jednostavnijih metoda.

# Jedan od mogućih problema je relativno mali skup podataka, zbog čega rezultati mogu značajno ovisiti o slučajnoj podjeli na train i test skup. 
# To može objasniti zašto je testna točnost logističke regresije veća od trening točnosti.

# Također, korištena je samo metrika točnosti (accuracy), što nije uvijek dovoljno. 
# Ako su klase neuravnotežene, accuracy može dati varljivo dobar rezultat.

# Predlaganje mogućih poboljšanja

# Prvo poboljšanje bilo bi korištenje dodatnih evaluacijskih metrika poput precision, recall, F1-score i confusion matrix kako bi se dobila potpunija slika kvalitete modela.

# Drugo, bilo bi korisno koristiti k-fold cross-validation za sve modele, a ne samo za optimizaciju hiperparametara, kako bi procjena performansi bila stabilnija i pouzdanija.

# Treće, mogla bi se ispitati važnost dodatnih atributa poput spola korisnika, jer trenutni model koristi samo dob i procijenjenu plaću.

# Kod SVM modela mogla bi se testirati i linearna, polinomijalna te sigmoid kernel funkcija radi usporedbe performansi i složenosti granice odluke.

# Također, moguće je dodatno proširiti raspon hiperparametara u GridSearchCV kako bi se pronašlo još bolje rješenje.

# Na kraju, korisno bi bilo analizirati pogrešno klasificirane primjere kako bi se bolje razumjelo gdje modeli najčešće griješe i postoji li obrazac u tim pogreškama.