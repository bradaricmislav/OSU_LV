import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))

top = np.hstack((black, white))
bottom = np.hstack((white, black))

img = np.vstack((top, bottom))
plt.imshow(img, cmap='gray')
plt.show()