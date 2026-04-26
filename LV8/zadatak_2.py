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

# Analiza dobivenih rješenja

# U ovom zadatku učitan je prethodno spremljeni model neuronske mreže za klasifikaciju rukom pisanih znamenki te je ponovno korišten MNIST skup podataka 
# kako bi se analizirali pogrešno klasificirani primjeri iz testnog skupa.

# Najprije je model uspješno učitan pomoću funkcije load_model(), čime je omogućeno korištenje već istrenirane mreže bez ponovnog procesa treniranja. 
# Nakon toga učitan je testni dio MNIST skupa podataka koji sadrži 10 000 slika rukom pisanih znamenki.

# Ulazni podaci ponovno su skalirani u raspon [0,1] dijeljenjem s 255 te je dodana dodatna dimenzija kako bi oblik podataka odgovarao ulazu modela (28, 28, 1).

# Model zatim računa predikcije za sve slike iz testnog skupa pomoću metode predict(). Budući da izlaz modela predstavlja vjerojatnosti pripadnosti svakoj od 10 klasa, 
# funkcijom argmax() određuje se klasa s najvećom vjerojatnošću, odnosno konačna predviđena oznaka.

# Usporedbom predviđenih oznaka i stvarnih oznaka izdvojeni su svi pogrešno klasificirani primjeri. Funkcija np.where() vraća indekse svih pogrešnih predikcija, 
# čime se može izračunati ukupan broj pogrešaka modela.

# Grafičkim prikazom prvih deset pogrešno klasificiranih slika omogućena je detaljnija analiza ponašanja modela. U naslovu svake slike prikazana je stvarna oznaka i 
# oznaka koju je model predvidio. Time se jasno može vidjeti kod kojih znamenki model najčešće griješi.

# Najčešće pogreške javljaju se kod vizualno sličnih znamenki, primjerice između 3 i 5, 4 i 9, 7 i 9 ili 8 i 3. Kod nekih primjera rukopis je nejasan ili neuobičajen, 
# što dodatno otežava klasifikaciju čak i čovjeku.

# Kritički osvrt na rezultate

# Ovakva analiza pogrešno klasificiranih primjera vrlo je korisna jer sama vrijednost accuracy ne daje dovoljno informacija o tome gdje model zaprvo griješi. 
# Vizualni pregled pogrešaka omogućuje bolje razumijevanje ograničenja modela.

# Rezultati pokazuju da model uglavnom dobro radi, ali ima poteškoća kod znamenki koje imaju sličan oblik ili su loše napisane. 
# To je očekivano jer korištena potpuno povezana mreža ne iskorištava prostorne informacije slike jednako dobro kao konvolucijske mreže.

# Također, prikazano je samo prvih deset pogrešaka, što ne mora biti reprezentativno za sve vrste pogrešnih klasifikacija. 
# Za potpuniju analizu bilo bi korisno pregledati veći broj primjera ili analizirati pogreške po klasama.

# Jedan od nedostataka je što se ne prikazuje sigurnost predikcije (vjerojatnost softmax izlaza). Ponekad model griješi s vrlo malom razlikom između dvije klase, 
# što može pomoći u razumijevanju problema.

# Dodatno, redoslijed prikazanih pogrešaka ovisi o redoslijedu u testnom skupu, a ne o težini pogreške ili sigurnosti modela.

# Predlaganje mogućih poboljšanja

# Jedno od mogućih poboljšanja bilo bi prikazivanje softmax vjerojatnosti uz predviđenu klasu kako bi se vidjelo koliko je model siguran u svoju pogrešnu odluku.

# Također, korisno bi bilo grupirati pogreške prema klasama i analizirati koje se znamenke najčešće međusobno zamjenjuju. 
# To se može dodatno potvrditi pomoću matrice zabune iz prethodnog zadatka.

# Umjesto prikaza samo prvih deset pogrešnih primjera, mogla bi se nasumično birati pogrešno klasificirana slika kako bi analiza bila raznovrsnija i objektivnija.

# Najveće poboljšanje performansi postiglo bi se zamjenom potpuno povezane mreže konvolucijskom neuronskom mrežom (CNN), 
# koja znatno bolje obrađuje slikovne podatke i smanjuje broj ovakvih pogrešaka.

# Moguće je primijeniti i data augmentation tehnike poput malih rotacija, translacija ili promjena debljine linije kako bi model postao robusniji na različite stilove rukopisa.

# Na kraju, korisno bi bilo prikazati i nekoliko primjera gdje je model bio vrlo siguran, ali ipak pogriješio, 
# jer upravo takvi slučajevi često najbolje otkrivaju slabosti modela.