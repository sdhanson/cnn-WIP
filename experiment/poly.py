import numpy as np

freq = np.array([0.666, 1.06, 2.08])
velocity = np.array([0.58, 1.378, 2.04])

func = np.polyfit(freq, velocity, 2)

print(func)