import matplotlib.pyplot as plt
import numpy as np

freq = np.array([0.3, 0.7, 1.5])
velocity = np.array([0.5, 1.0, 1.6])

func = np.polyfit(freq, velocity, 2)

p = np.poly1d(func)

print(p(0.3))

print(func)