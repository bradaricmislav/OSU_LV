# Zadatak 5.5.1 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
# ulazne velicine. Podaci su podijeljeni na skup za u ˇ cenje i skup za testiranje modela. ˇ
# a) Prikažite podatke za ucenje u ˇ x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite ˇ
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je moguce de ´ finirati boju svake klase.
# b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
# taka za ucenje. ˇ
# c) Pronadite u atributima izgra ¯ denog modela parametre modela. Prikažite granicu odluke ¯
# naucenog modela u ravnini ˇ x1 − x2 zajedno s podacima za ucenje. Napomena: granica ˇ
# odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.
# d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunate to ˇ cnost, ˇ
# preciznost i odziv na skupu podataka za testiranje.
# e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznacite dobro klasi ˇ ficirane
# primjere dok pogrešno klasificirane primjere oznacite crnom bojom.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a)
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], cmap="bwr", c=y_train)
plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test, cmap="bwr", marker='x')

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Podaci za učenje i testiranje")
plt.show()

#b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

#c)
theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_[0]

print("theta0 =", theta0)
print("theta1 =", theta1)
print("theta2 =", theta2)

x = np.linspace(min(X_train[:,0]), max(X_train[:,0]), 100)
y = -(theta0 + theta1 * x) / theta2

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')

plt.plot(x, y, color='red')
plt.title("Granica odluke")
plt.show()

#d)
y_test_p = LogRegression_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_p)
print ("Matrica zabune: ", cm )

accuracy = accuracy_score(y_test, y_test_p)
precision = precision_score(y_test, y_test_p)
recall = recall_score(y_test, y_test_p)

print("Točnost:", accuracy)
print("Preciznost:", precision)
print("Odziv:", recall)

#e)
correct = (y_test == y_test_p)

plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Točno')

plt.scatter(X_test[~correct, 0], X_test[~correct, 1], c='black', label='Pogrešno')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rezultati klasifikacije (test)')
plt.legend()
plt.show()
