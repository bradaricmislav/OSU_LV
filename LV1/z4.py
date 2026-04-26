# Zadatak 1.4.4 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
# Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
# datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
# Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.

word_count = {} 

file = open("song.txt")
for line in file:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
file.close()

unique_words = 0
for word in word_count:
    if word_count[word] == 1:
        unique_words += 1
        print(f"Riječi koje se pojavljuju samo jednom: {word}")
print(f"Broj rijeci koje se pojavljuju samo jednom: {unique_words}")
