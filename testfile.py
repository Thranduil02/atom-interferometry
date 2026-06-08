import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
y = x**2

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = x²")
plt.show()
