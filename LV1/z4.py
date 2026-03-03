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
