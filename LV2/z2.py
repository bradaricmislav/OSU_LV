import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("LV1/LV2/data.csv", delimiter=',', skiprows=1)

print("Broj osoba: ", data.shape[0])

height = data[:, 1]
weight = data[:, 2]

plt.scatter(height, weight)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

plt.scatter(height[::50], weight[::50])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase (svaka 50. osoba)")
plt.show()

min_height = np.min(data[:, 1])
max_height = np.max(data[:, 1])
mean_height = np.mean(data[:, 1])

print(f"Najniža osoba visoka je: {min_height}")
print(f"Najvišlja osoba visoka je: {max_height}")
print(f"Prosječna visona svih osoba je: {round(mean_height,2)}")

index_male = (data[:, 0] == 1)
index_female = (data[:, 0] == 0)

male_height = data[index_male, 1]
female_height = data[index_female, 1]

print("\nMuškarci: ")
print(f"Najvišlji: {male_height.max()}, najniži: {male_height.min()}, prosječna visina: {round(male_height.mean(), 2)}")


print("\nŽene: ")
print(f"Najvišlja: {female_height.max()}, najniža: {female_height.min()}, prosječna visina: {round(female_height.mean(), 2)}")