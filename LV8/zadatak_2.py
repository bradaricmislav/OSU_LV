# Zadatak 8.4.2 Napišite skriptu koja ce u ´ citati izgra ˇ denu mrežu iz zadatka 1 i MNIST skup ¯
# podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasi ´ ficiranih slika iz
# skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu ¯
# mrežom.

import numpy as np
import keras
from matplotlib import pyplot as plt
from keras.models import load_model

model = load_model("mnist_model.keras")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

wrong_idx = np.where(y_pred_classes != y_test)[0]

print(f"Ukupno pogrešaka: {len(wrong_idx)}")

plt.figure(figsize=(10, 5))
for i, idx in enumerate(wrong_idx[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}\nPredviđeno: {y_pred_classes[idx]}")
    plt.axis('off')
plt.show()
