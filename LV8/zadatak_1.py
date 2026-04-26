# Zadatak 8.4.1 MNIST podatkovni skup za izgradnju klasifikatora rukom pisanih znamenki
# dostupan je u okviru Keras-a. Skripta zadatak_1.py ucitava MNIST podatkovni skup te podatke ˇ
# priprema za ucenje potpuno povezane mreže. ˇ
# 1. Upoznajte se s ucitanim podacima. Koliko primjera sadrži skup za u ˇ cenje, a koliko skup za ˇ
# testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne velicina? ˇ
# 2. Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za u ´ cenje te ispišite ˇ
# njezinu oznaku u terminal.
# 3. Pomocu klase ´ Sequential izgradite mrežu prikazanu na slici 8.5. Pomocu metode ´
# .summary ispišite informacije o mreži u terminal.
# 4. Pomocu metode ´ .compile podesite proces treniranja mreže.
# 5. Pokrenite ucenje mreže (samostalno de ˇ finirajte broj epoha i velicinu serije). Pratite tijek ˇ
# ucenja u terminalu. ˇ
# 6. Izvršite evaluaciju mreže na testnom skupu podataka pomocu metode ´ .evaluate.
# 7. Izracunajte predikciju mreže za skup podataka za testiranje. Pomo ˇ cu scikit-learn biblioteke ´
# prikažite matricu zabune za skup podataka za testiranje.
# 8. Pohranite model na tvrdi disk.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()




# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",
optimizer="adam",
metrics=["accuracy",])


# TODO: provedi ucenje mreze
batch_size = 32
epochs = 10

history = model.fit(
    x_train_s, y_train_s,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion matrix:\n", cm)


# TODO: spremi model
model.save("mnist_model.keras")


# Analiza dobivenih rješenja

# U ovom zadatku korišten je MNIST podatkovni skup za klasifikaciju rukom pisanih znamenki od 0 do 9 primjenom potpuno povezane neuronske mreže (Fully Connected Neural Network). 
# Skup za učenje sadrži 60 000 slika, dok skup za testiranje sadrži 10 000 slika. Svaka slika dimenzija je 28 × 28 piksela u sivim tonovima.

# Ulazni podaci skalirani su dijeljenjem vrijednosti piksela s 255, čime su sve vrijednosti normalizirane u raspon [0,1]. 
# Takva normalizacija poboljšava stabilnost i brzinu učenja neuronske mreže. Budući da Keras očekuje ulaz oblika (visina, širina, kanal), 
# slikama je dodana dodatna dimenzija pa konačni oblik postaje (28, 28, 1).

# Izlazne oznake pretvorene su u one-hot encoding pomoću funkcije to_categorical(). 
# Umjesto jedne vrijednosti klase, svaka znamenka predstavljena je vektorom duljine 10, gdje samo jedna pozicija ima vrijednost 1.

# Izgrađena je potpuno povezana mreža pomoću klase Sequential. Mreža se sastoji od sloja za poravnavanje slike (Flatten), 
# zatim dva skrivena gusta sloja sa 100 i 50 neurona te ReLU aktivacijskom funkcijom, te izlaznog sloja s 10 neurona i softmax aktivacijom za višeklasnu klasifikaciju.

# Model je treniran korištenjem optimizatora Adam i funkcije gubitka categorical_crossentropy, što je standardan izbor za ovakav problem. 
# Tijekom treniranja prati se metrika accuracy, a korišten je i validation split za praćenje ponašanja modela na validacijskom skupu.

# Nakon treniranja provedena je evaluacija na testnom skupu pomoću metode .evaluate(). 
# Dobivena testna točnost je vrlo visoka, što pokazuje da mreža uspješno prepoznaje rukom pisane znamenke.

# Matrica zabune dodatno omogućuje analizu pogrešnih klasifikacija. Najčešće pogreške pojavljuju se između vizualno sličnih znamenki poput 3 i 5, 4 i 9 ili 7 i 9, 
# što je očekivano zbog sličnog oblika rukopisa.

# Na kraju je model uspješno spremljen na disk pomoću metode .save(), čime je omogućeno njegovo kasnije ponovno korištenje bez potrebe za ponovnim treniranjem.

# Kritički osvrt na rezultate

# Rezultati pokazuju da i relativno jednostavna potpuno povezana neuronska mreža može postići vrlo visoku točnost na MNIST skupu podataka. 
# To potvrđuje da je MNIST dobar početni primjer za razumijevanje rada neuronskih mreža.

# Međutim, korišteni model nije optimalan za obradu slika jer potpuno povezane mreže ne iskorištavaju prostorne odnose među pikselima. 
# Sloj Flatten pretvara sliku u običan vektor i time se gubi informacija o lokalnoj strukturi slike.

# Zbog toga ovakav pristup obično daje slabije rezultate od konvolucijskih neuronskih mreža (CNN), koje su posebno dizajnirane za obradu slikovnih podataka.

# Također, broj epoha i veličina batch-a odabrani su ručno bez dodatne optimizacije. Moguće je da bi drugačiji odabir hiperparametara dao još bolje rezultate 
# ili smanjio vrijeme treniranja.

# Matrica zabune pokazuje da model i dalje griješi kod sličnih znamenki, 
# što ukazuje da jednostavna arhitektura ima ograničenu sposobnost razlikovanja kompleksnijih uzoraka rukopisa.

# Još jedan nedostatak je izostanak grafičkog prikaza funkcije gubitka i točnosti kroz epohe, 
# što bi omogućilo bolju procjenu pojave overfittinga ili underfittinga.

# Predlaganje mogućih poboljšanja

# Najvažnije poboljšanje bilo bi korištenje konvolucijske neuronske mreže (CNN) umjesto potpuno povezane mreže. 
# CNN bolje prepoznaje lokalne uzorke poput rubova, linija i oblika te obično postiže znatno veću točnost na slikovnim zadacima.

# Dodatno, mogla bi se koristiti regularizacija poput Dropout slojeva kako bi se smanjio rizik overfittinga i poboljšala generalizacija modela.

# Također, korisno bi bilo eksperimentirati s različitim brojem neurona, brojem skrivenih slojeva, 
# batch size vrijednostima i brojem epoha kako bi se pronašla optimalna arhitektura.

# Praćenje learning curves (loss i accuracy grafova) omogućilo bi precizniju analizu procesa učenja i lakše uočavanje problema poput prenaučenosti.

# Moguće je primijeniti i early stopping tehniku koja automatski zaustavlja treniranje kada validacijska pogreška prestane padati.

# Za detaljniju evaluaciju modela korisno bi bilo vizualno prikazati nekoliko pogrešno klasificiranih znamenki kako bi se bolje razumjelo gdje model najčešće griješi.

# Na kraju, usporedba rezultata s drugim modelima poput SVM-a ili CNN-a dala bi jasniju sliku o kvaliteti dobivenog rješenja i prednostima dubokog učenja u ovom problemu.