# Zadatak 7.5.2 Kvantizacija boje je proces smanjivanja broja razlicitih boja u digitalnoj slici, ali ˇ
# uzimajuci u obzir da rezultantna slika vizualno bude što sli ´ cnija originalnoj slici. Jednostavan ˇ
# nacin kvantizacije boje može se posti ˇ ci primjenom algoritma ´ K srednjih vrijednosti na RGB
# vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
# elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
# slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
# kvantizacije i koja sadrži samo 5 boja koje su odredene algoritmom ¯ K srednjih vrijednosti.
# 1. Otvorite skriptu zadatak_2.py. Ova skripta ucitava originalnu RGB sliku ˇ test_1.jpg
# te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri cemu je ˇ n
# broj elemenata slike, a m je jednak 3. Koliko je razlicitih boja prisutno u ovoj slici? ˇ
# 2. Primijenite algoritam K srednjih vrijednosti koji ce prona ´ ci grupe u RGB vrijednostima ´
# elemenata originalne slike.
# 3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom. ´
# 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
# rezultate.
# 5. Primijenite postupak i na ostale dostupne slike.
# 6. Graficki prikažite ovisnost ˇ J o broju grupa K. Koristite atribut inertia objekta klase
# KMeans. Možete li uociti lakat koji upu ˇ cuje na optimalni broj grupa? ´
# 7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
# primjecujete?

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