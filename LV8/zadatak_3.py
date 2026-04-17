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