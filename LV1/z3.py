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
