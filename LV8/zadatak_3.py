# Zadatak 8.4.3 Napišite skriptu koja ce u ´ citati izgra ˇ denu mrežu iz zadatka 1. Nadalje, skripta ¯
# treba ucitati sliku ˇ test.png sa diska. Dodajte u skriptu kod koji ce prilagoditi sliku za mrežu, ´
# klasificirati sliku pomocu izgra ´ dene mreže te ispisati rezultat u terminal. Promijenite sliku ¯
# pomocu nekog gra ´ fickog alata (npr. pomo ˇ cu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite ´
# skriptu. Komentirajte dobivene rezultate za razlicite napisane znamenke.

import numpy as np
import keras
from matplotlib import pyplot as plt
from PIL import Image 
from keras.models import load_model

model = load_model("mnist_model.keras")

img = Image.open("test.png").convert("L")  
img = img.resize((28,28))

img_array = np.array(img)

img_array = 255 - img_array

img_array = img_array.astype("float32") / 255
img_array = np.expand_dims(img_array, axis=(0,-1))

prediction = model.predict(img_array)
pred_class = np.argmax(prediction)

print("Predicted digit:", pred_class)

plt.imshow(img, cmap="gray")
plt.title(f"Predicted: {pred_class}")
plt.axis("off")
plt.show()