# Zadatak 7.5.1 Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
# umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
# koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji nacin ˇ ce´
# se generirati podaci, a vraca generirani skup podataka u obliku numpy polja pri ´ cemu su prvi i ˇ
# drugi stupac vrijednosti prve odnosno druge ulazne velicine za svaki podatak. Skripta generira ˇ
# 500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.
# 1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
# nacin generiranja podataka. ˇ
# 2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
# obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
# kod. Mijenjate broj K. Što primjecujete? ´
# 3. Mijenjajte nacin de ˇ finiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
# (koristite optimalni broj grupa). Kako komentirate dobivene rezultate?

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

K = 2
kmeans = KMeans(n_clusters=K, init='random', n_init=5, random_state=0)
labels = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K-means grupiranje')
plt.show()

# Analiza dobivenih rješenja

# U ovom zadatku korištena je metoda grupiranja K-srednjih vrijednosti (K-Means) na umjetno generiranim podacima. 
# Funkcija generate_data() omogućuje generiranje različitih oblika skupova podataka kako bi se lakše uočile prednosti i ograničenja algoritma grupiranja.

# Kod prve vrste generiranja podataka (flagc = 1) jasno su vidljive tri odvojene grupe. 
# Takav skup podataka vrlo je pogodan za K-Means jer algoritam najbolje radi kada su grupe približno kružnog oblika i dobro međusobno odvojene. 
# U tom slučaju grupiranje daje vrlo dobre rezultate i većina uzoraka bude ispravno pridružena odgovarajućoj grupi.

# Kod druge vrste podataka (flagc = 2) grupe su izdužene i rotirane zbog linearne transformacije. 
# Iako i dalje postoje tri prirodne grupe, K-Means može imati poteškoće jer pretpostavlja približno sferne klastere. 
# Granice između grupa nisu jednako dobro određene kao u prvom slučaju.

# Kod treće vrste (flagc = 3) postoje četiri grupe različitih veličina i različitih standardnih devijacija. 
# Ovdje se vidi da K-Means slabije radi kada klasteri nisu jednake gustoće i veličine. Veći i raspršeniji klasteri mogu negativno utjecati na položaj centroida.

# Kod četvrte vrste (flagc = 4) podaci imaju oblik koncentričnih krugova, dok kod pete vrste (flagc = 5) imaju oblik dva polumjeseca. 
# Iako se vizualno jasno vide dvije grupe, K-Means ovdje daje loše rezultate jer koristi euklidsku udaljenost i linearne granice između centroida. 
# Takvi nelinearni oblici nisu prikladni za ovaj algoritam.

# Promjenom broja klastera K može se uočiti da premala vrijednost K uzrokuje spajanje različitih grupa u jednu, 
# dok prevelika vrijednost K nepotrebno dijeli prirodne grupe na manje dijelove. 
# Optimalan broj klastera najčešće odgovara stvarnom broju grupa u podacima.

# Kritički osvrt na rezultate

# Rezultati jasno pokazuju da uspješnost K-Means algoritma snažno ovisi o obliku i rasporedu podataka. 
# Kada su klasteri kompaktni, dobro odvojeni i približno kružnog oblika, algoritam daje vrlo dobre rezultate.

# Međutim, kod složenijih oblika poput izduženih klastera, 
# klastera različitih gustoća ili nelinearnih struktura poput krugova i polumjeseca, kvaliteta grupiranja značajno opada. 
# To je posljedica osnovne pretpostavke K-Means algoritma da su klasteri sferni i slične veličine.

# Također, algoritam je osjetljiv na inicijalni odabir centroida. 
# Višestrukim pokretanjem programa mogu se dobiti različiti rezultati, posebno kada se koristi slučajna inicijalizacija (init='random'). 
# To može dovesti do lokalnog minimuma i lošijeg konačnog rješenja.

# Jedan od problema je i potreba unaprijed zadanog broja klastera K. 
# U stvarnim problemima taj broj često nije poznat, pa je potrebno koristiti dodatne metode za njegov odabir.

# Predlaganje mogućih poboljšanja

# Prvo poboljšanje bilo bi korištenje metode Elbow ili Silhouette Score za određivanje optimalnog broja klastera K umjesto ručnog odabira. 
# Time bi rezultat bio objektivniji i pouzdaniji.

# Drugo, umjesto slučajne inicijalizacije centroida (init='random') bolje je koristiti k-means++, 
# koji obično daje stabilnije rezultate i smanjuje rizik lošeg početnog odabira centroida.

# Za podatke složenijih oblika, poput koncentričnih krugova i polumjeseca, 
# prikladnije bi bilo koristiti druge algoritme grupiranja poput DBSCAN ili hijerarhijskog grupiranja (Agglomerative Clustering), 
# jer oni bolje prepoznaju nelinearne strukture.

# Također, mogla bi se analizirati kvaliteta grupiranja pomoću dodatnih metrika poput inertia, 
# silhouette score ili vizualizacije dendrograma kod hijerarhijskog grupiranja.

# Na kraju, korisno bi bilo usporediti rezultate više različitih algoritama grupiranja na istom skupu podataka kako bi se 
# jasnije vidjelo koji algoritam najbolje odgovara određenoj strukturi podataka.