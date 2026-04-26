# Zadatak 1.4.2 Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
# nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju ¯
# sljedecih uvjeta: ´
# >= 0.9 A
# >= 0.8 B
# >= 0.7 C
# >= 0.6 D
# < 0.6 F
# Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
# Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju ¯ cu poruku.


try:
    num = float(input("Unesi ocjenu između 0 i 1: "))
    if num > 1.0 or num < 0.0:
        print("Ocjena nije u predviđenom rasponu! Ponovite unos")
    elif num < 0.6:
        print('F')
    elif num < 0.7:
        print('D')
    elif num < 0.8:
        print('C')
    elif num < 0.9:
        print('B')
    else:
        print('A')
except:
    print("Unesite broj! ")


