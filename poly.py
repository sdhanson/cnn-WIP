import matplotlib.pyplot as plt
import numpy as np

freq = np.array([0.6, 1.3, 2.7])
velocity = np.array([0.8323, 1.2191, 1.239])

func = np.polyfit(freq, velocity, 2)

print(func)