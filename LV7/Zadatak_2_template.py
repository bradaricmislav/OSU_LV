import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1)
unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", len(unique_colors))

#2)
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(img_array)
centers = kmeans.cluster_centers_

#3)
img_array_aprox = centers[labels]
img_aprox = np.reshape(img_array_aprox, (w, h, d))
img_aprox = np.clip(img_aprox, 0, 1)
plt.figure()
plt.title(f"Kvantizirana slika")
plt.imshow(img_aprox)
plt.show()

#4)

#5)

#6)
J = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(img_array)
    J.append(km.inertia_)

plt.figure()
plt.plot(K_range, J, marker='o')
plt.xlabel("K")
plt.ylabel("J (inertia)")
plt.title("Elbow metoda")
plt.show()

#7)
k_final = 5
for i in range(k_final):
    mask = (labels == i)
    binary_img = mask.reshape(w, h)

    plt.figure()
    plt.title(f"Klaster {i}")
    plt.imshow(binary_img, cmap='gray')
    plt.show()