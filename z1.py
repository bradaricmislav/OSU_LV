def total_euro(hours, pay_rate):
    return hours*pay_rate

hours = input("Radni sati: ")
hours = hours.split(" ")[0]
hours = float(hours)
pay_rate = float(input("eura/h: "))

print(f"Ukupno: {total_euro(hours, pay_rate)} eura")