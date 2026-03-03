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


