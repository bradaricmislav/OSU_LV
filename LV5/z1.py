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
