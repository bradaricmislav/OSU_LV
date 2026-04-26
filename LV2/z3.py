# Zadatak 2.4.3 Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce´
# numpy matrice pokušajte:
# a) posvijetliti sliku,
# b) prikazati samo drugu cetvrtinu slike po širini, ˇ
# c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
# d) zrcaliti sliku.

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV1/LV2/road.jpg")

bright_img = np.clip(img.astype(int) + 50, 0, 255)
plt.imshow(bright_img)
plt.title("Posvijetljena slika")
plt.show()

width = img.shape[1]
quarter_image = img[:, width//4 : width//2]
plt.imshow(quarter_image)
plt.title("Druga četvrtina slike")
plt.show()

rotated_img = np.rot90(img, -1)
plt.imshow(rotated_img)
plt.title("Rotirana slika")
plt.show()

mirrored_img = np.fliplr(img)
plt.imshow(mirrored_img)
plt.title("Zrcaljena slika")
plt.show()
