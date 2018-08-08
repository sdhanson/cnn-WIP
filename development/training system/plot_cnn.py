# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:19:11 2018

@author: hansos2
"""

import matplotlib.pyplot as plt

times = []
y = []

peaktimes = []
peaky = []

inputfilename = "training_out.txt"
peakfilename = "training_steps_CNN_controller.txt"

with open(inputfilename) as inputfile:
    for inputline in inputfile:
        inputline = inputline.strip()
        parts = inputline.split(",")
        times.append(float(parts[2]))
        y.append(float(parts[4]))


with open(peakfilename) as peakfile:
    for peakline in peakfile:
        peakline = peakline.strip()
        peaktimes.append(float(peakline))
        peaky.append(0)

plt.plot(times, y)
plt.plot(peaktimes,peaky, 'r+')
plt.ylabel("Acceleration Y")
plt.xlabel("Time.time")
plt.savefig("training_CNN_graph")
plt.show()
plt.close()