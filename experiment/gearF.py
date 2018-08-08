# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 08:52:01 2018

@author: vandy
"""
import matplotlib.pyplot as plt

slowAvg = 0
medAvg = 0
fastAvg = 0

slowCount = 0
medCount = 0
fastCount = 0

active = False
count = 0

times = []
y = []

peaktimes = []
peaky = []

inputfilename = "inGearFreq.txt"

with open(inputfilename) as inputfile:
    for inputline in inputfile:
        inputline = inputline.strip()
        parts = inputline.split(";")
        times.append(float(parts[0]))
        y.append(float(parts[3]))
        
        if parts[1] == "False":
            active = False
        elif parts[1] == "True" and not active:
            count += 1
            active = True
        
        if active and count == 1:
            slowAvg += float(parts[13])
            slowCount += 1
        
        elif active and count == 2:
            medAvg += float(parts[13])
            medCount += 1
            
        elif active and count > 2:
            fastAvg += float(parts[13])
            fastCount += 1
            
print(slowAvg / slowCount)
print(medAvg / medCount)
print(fastAvg / fastCount)

plt.plot(times, y)
plt.ylabel("Acceleration Y")
plt.xlabel("Time.time")
plt.savefig("graphGearFreq")
plt.show()
plt.close()