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

# Analiza dobivenih rješenja

# U ovom zadatku primijenjen je algoritam K-srednjih vrijednosti (K-Means) za kvantizaciju boje slike. 
# Cilj postupka bio je smanjiti broj različitih boja u slici, uz zadržavanje što sličnijeg vizualnog izgleda originalu.

# Najprije je učitana RGB slika te su vrijednosti piksela normalizirane u raspon od 0 do 1. 
# Svaki piksel slike promatra se kao jedan podatkovni primjer s tri značajke – crvena (R), zelena (G) i plava (B) komponenta. 
# Na taj način slika je transformirana u 2D matricu oblika (broj piksela × 3).

# Izračunom funkcije np.unique() dobiven je broj različitih boja prisutnih u originalnoj slici. 
# Taj broj je vrlo velik jer gotovo svaki piksel može imati blago različitu RGB vrijednost, posebno kod fotografija i prirodnih scena.

# Primjenom K-Means algoritma s odabranim brojem klastera K = 5 određeno je pet dominantnih boja slike. 
# Svakom pikselu dodijeljen je najbliži centroid te je njegova originalna boja zamijenjena bojom centra pripadajućeg klastera. 
# Rezultat je kvantizirana slika koja sadrži znatno manji broj boja, ali i dalje zadržava osnovni vizualni sadržaj.

# Povećanjem broja klastera K dobiva se kvalitetnija aproksimacija originalne slike jer model koristi više boja i bolje opisuje detalje slike. 
# Smanjenjem vrijednosti K slika postaje jednostavnija, gubi detalje i pojavljuju se vidljive prijelazne granice između boja.

# Grafički prikaz ovisnosti funkcije cilja J (inertia) o broju klastera K pokazuje opadanje pogreške povećanjem broja grupa. 
# U određenom trenutku smanjenje pogreške postaje znatno sporije, što predstavlja tzv. „lakat” i može pomoći pri odabiru optimalnog broja klastera.

# Prikaz binarnih slika pojedinih klastera omogućuje uvid u to koji dijelovi slike pripadaju određenoj dominantnoj boji. 
# Često se može primijetiti da jedan klaster odgovara pozadini, drugi objektima u prvom planu, a ostali različitim nijansama osvjetljenja i sjena.

# Kritički osvrt na rezultate

# Rezultati pokazuju da K-Means predstavlja vrlo jednostavnu i učinkovitu metodu za kvantizaciju boje, 
# posebno kada je cilj smanjenje memorijskih zahtjeva ili pojednostavljenje slike.

# Međutim, algoritam ima određena ograničenja. K-Means koristi euklidsku udaljenost u RGB prostoru, što ne odgovara uvijek ljudskoj percepciji boja. 
# Dvije boje koje su matematički bliske ne moraju nužno biti vizualno slične.

# Također, algoritam zanemaruje prostorni raspored piksela. Svaki piksel promatra se neovisno o svojoj poziciji u slici, 
# zbog čega se mogu pojaviti vizualno neprirodni prijelazi između područja.

# Odabir broja klastera K značajno utječe na kvalitetu rezultata. Premali K dovodi do velikog gubitka detalja, 
# dok preveliki K smanjuje korist kvantizacije jer slika ostaje gotovo jednako složena kao original.

# Algoritam je također osjetljiv na inicijalizaciju centroida, pa različita pokretanja mogu dati nešto različite rezultate, 
# posebno kod složenijih slika s velikim brojem nijansi.

# Predlaganje mogućih poboljšanja

# Jedno od mogućih poboljšanja bilo bi korištenje metode Elbow za sustavniji odabir optimalnog broja klastera K umjesto ručnog odabira vrijednosti poput K = 5.

# Također, mogla bi se koristiti Silhouette analiza kao dodatna metoda procjene kvalitete grupiranja boja.

# Umjesto RGB prostora, bolji rezultati često se mogu postići korištenjem perceptualno prikladnijih prostora boja poput LAB ili HSV, 
# jer oni bolje odgovaraju načinu na koji čovjek percipira razlike među bojama.

# Dodatno poboljšanje bilo bi uključivanje prostornih koordinata piksela (x, y) zajedno s RGB vrijednostima kako bi algoritam uzeo u obzir i 
# položaj piksela u slici, čime bi se smanjili neprirodni prijelazi.

# Moguće je isprobati i druge metode grupiranja poput Gaussian Mixture Models ili hijerarhijskog grupiranja koje ponekad daju kvalitetnije rezultate za složenije slike.

# Na kraju, korisno bi bilo usporediti rezultate na više različitih slika – portretima, pejzažima i objektima – 
# kako bi se bolje razumjelo u kojim situacijama K-Means daje najbolje rezultate za kvantizaciju boje.