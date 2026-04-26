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

# Analiza dobivenih rješenja

# U ovom zadatku korišten je prethodno istrenirani i spremljeni model neuronske mreže za klasifikaciju rukom pisanih znamenki, 
# a cilj je bio testirati model na potpuno novoj slici spremljenoj izvan MNIST skupa podataka.

# Najprije je model uspješno učitan pomoću funkcije load_model(), čime je omogućena klasifikacija bez ponovnog treniranja mreže. 
# Nakon toga učitana je slika test.png s diska pomoću bibliotke PIL (Image.open()).

# Slika je pretvorena u grayscale način rada (convert("L")) kako bi odgovarala formatu MNIST skupa podataka koji koristi slike u sivim tonovima. 
# Zatim je promijenjena njezina veličina na dimenzije 28 × 28 piksela, što je nužno jer model očekuje upravo takav ulaz.

# Vrijednosti piksela pretvorene su u numpy polje te je izvršena inverzija boja pomoću izraza 255 - img_array. 
# Ovaj korak je važan jer MNIST koristi crnu pozadinu i bijelu znamenku, dok korisnički nacrtane slike često imaju bijelu pozadinu i crni broj.

# Nakon toga provedena je normalizacija dijeljenjem s 255 kako bi vrijednosti bile u rasponu [0,1], jednako kao tijekom treniranja modela. 
# Dodane su dodatne dimenzije kako bi oblik ulaza odgovarao mreži (1, 28, 28, 1).

# Model zatim računa predikciju pomoću metode predict(), a funkcijom argmax() određuje se klasa s najvećom vjerojatnošću. 
# Rezultat klasifikacije ispisuje se u terminal i prikazuje kao naslov slike.

# Kod pravilno nacrtanih i jasno centriranih znamenki model najčešće daje točnu klasifikaciju. Međutim, kod loše nacrtanih, 
# nejasnih ili necentriranih znamenki mogu se pojaviti pogreške jer takve slike odstupaju od uzoraka na kojima je model treniran.

# Kritički osvrt na rezultate

# Rezultati pokazuju da model dobro radi kada ulazna slika dovoljno nalikuje primjerima iz MNIST skupa podataka. 
# To znači da je važno da znamenka bude jasno nacrtana, pravilno centrirana i slične debljine linije kao u originalnom skupu.

# Jedan od glavnih problema je osjetljivost modela na način crtanja znamenke. Ako je broj previše pomaknut, premalen, predebeo ili neuredno nacrtan, 
# model može dati pogrešnu klasifikaciju.

# Također, jednostavna promjena veličine slike na 28 × 28 nije uvijek dovoljna za kvalitetnu pripremu ulaza. 
# Bez dodatnog centriranja i uklanjanja praznog prostora model može dobiti lošiju reprezentaciju znamenke.

# Još jedan nedostatak je što se prikazuje samo konačna predikcija, bez informacija o sigurnosti modela. 
# U nekim slučajevima model može biti nesiguran između dvije slične znamenke, ali to se ne vidi iz ispisa.

# Korištena potpuno povezana mreža dodatno ograničava kvalitetu klasifikacije jer nije optimalna za obradu slikovnih podataka u usporedbi s konvolucijskim mrežama.

# Predlaganje mogućih poboljšanja

# Jedno od najvažnijih poboljšanja bilo bi dodatno predprocesiranje slike prije klasifikacije. To uključuje automatsko centriranje znamenke, 
# uklanjanje viška praznog prostora i podešavanje debljine linije kako bi ulaz što više nalikovao MNIST primjerima.

# Također, korisno bi bilo ispisati i softmax vjerojatnosti za svih 10 klasa kako bi se vidjelo koliko je model siguran u svoju odluku.

# Moguće je implementirati prikaz nekoliko najvjerojatnijih klasa umjesto samo jedne konačne predikcije, što bi bilo posebno korisno kod nejasno nacrtanih znamenki.

# Znatno bolje rezultate dala bi konvolucijska neuronska mreža (CNN), koja bolje prepoznaje lokalne značajke slike i robusnija je na male promjene oblika znamenki.

# Dodatno, model bi se mogao dodatno trenirati na ručno nacrtanim znamenkama izvan MNIST skupa kako bi postao otporniji na različite stilove pisanja.

# Na kraju, korisno bi bilo omogućiti interaktivno crtanje znamenke unutar programa te trenutno prikazivanje predikcije, 
# što bi dodatno olakšalo testiranje modela i analizu ponašanja mreže.