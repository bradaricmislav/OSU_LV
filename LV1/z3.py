# Zadatak 1.4.3 Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
# sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
# potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
# vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
# (npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku

numbers = []

while True:
    num = input("Unesite broj ili done za prekid unosa: ")
    if num == "Done":
        break
    try:
        num = float(num)
        numbers.append(num)
    except:
        print("Greška, nije broj!")
    
print(f"Uneseno je {len(numbers)} broja/brojeva")
print(f"Srednja vrijednost svih brojeva je {sum(numbers)/len(numbers)}")
print(f"Najmanji broj je {min(numbers)}, a najveći {max(numbers)}")
print(f"Sortirani brojevi: {sorted(numbers)}")
