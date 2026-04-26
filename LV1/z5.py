# Zadatak 1.4.5 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
# [1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
# Primjer dijela datoteke:
# ham Yup next stop.
# ham Ok lar... Joking wif u oni...
# spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
# a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
# prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
# b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?

spam_count = 0
ham_count = 0
spam_words = 0
ham_words = 0
spam_usklicnik = 0

sms_file = open("SMSSpamCollection.txt", encoding="utf-8")
for line in sms_file:
    line = line.rstrip()
    tip, poruka = line.split("\t")

    total_words = len(poruka.split())

    if tip == "spam":
        spam_count += 1
        spam_words += total_words

        if poruka.endswith("!"):
            spam_usklicnik += 1
    elif tip == "ham":
        ham_count += 1
        ham_words += total_words

avg_spam = round(spam_words/spam_count, 2)
avg_ham = round(ham_words/ham_count, 2)

print(f"Prosječan broj riječi u spam porukama: {avg_spam}")
print(f"Prosječan broj riječi u ham porukama: {avg_ham}")
print(f"Broj spam poruka koje završavaju uskličnikom: {spam_usklicnik}")
