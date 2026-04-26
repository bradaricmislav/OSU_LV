# Zadatak 8.4.1 MNIST podatkovni skup za izgradnju klasifikatora rukom pisanih znamenki
# dostupan je u okviru Keras-a. Skripta zadatak_1.py ucitava MNIST podatkovni skup te podatke ˇ
# priprema za ucenje potpuno povezane mreže. ˇ
# 1. Upoznajte se s ucitanim podacima. Koliko primjera sadrži skup za u ˇ cenje, a koliko skup za ˇ
# testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne velicina? ˇ
# 2. Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za u ´ cenje te ispišite ˇ
# njezinu oznaku u terminal.
# 3. Pomocu klase ´ Sequential izgradite mrežu prikazanu na slici 8.5. Pomocu metode ´
# .summary ispišite informacije o mreži u terminal.
# 4. Pomocu metode ´ .compile podesite proces treniranja mreže.
# 5. Pokrenite ucenje mreže (samostalno de ˇ finirajte broj epoha i velicinu serije). Pratite tijek ˇ
# ucenja u terminalu. ˇ
# 6. Izvršite evaluaciju mreže na testnom skupu podataka pomocu metode ´ .evaluate.
# 7. Izracunajte predikciju mreže za skup podataka za testiranje. Pomo ˇ cu scikit-learn biblioteke ´
# prikažite matricu zabune za skup podataka za testiranje.
# 8. Pohranite model na tvrdi disk.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()




# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",
optimizer="adam",
metrics=["accuracy",])


# TODO: provedi ucenje mreze
batch_size = 32
epochs = 10

history = model.fit(
    x_train_s, y_train_s,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion matrix:\n", cm)


# TODO: spremi model
model.save("mnist_model.keras")
