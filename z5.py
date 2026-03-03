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
