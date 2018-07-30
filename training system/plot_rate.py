# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:29:29 2018

@author: hansos2
"""


import matplotlib.pyplot as plt

times = []
y = []

peaktimes = []
peaky = []

inputfilename = "rate_test.txt"

with open(inputfilename) as inputfile:
    for inputline in inputfile:
        inputline = inputline.strip()
        parts = inputline.split(",")
        times.append(float(parts[0]))
        y.append(float(parts[3]))

plt.plot(times, y)
plt.ylabel("Acceleration Y")
plt.xlabel("Time.time")
plt.savefig("training_CNN_graph_controller")
plt.show()
plt.close()