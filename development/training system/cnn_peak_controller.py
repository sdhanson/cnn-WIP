# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:22:46 2018

@author: hansos2
"""

peakfilename = "training_steps_CNN_controller.txt"
filename = "training_steps_CNN.txt"
outputfilename = "training_out_controller.txt"

peaktimes = []
labels = []
i = 0
j = 0
k = 0

with open(peakfilename) as peakfile:
    for peakline in peakfile:
        peakline = peakline.strip()
        peaktimes.append(float(peakline))
        
with open (filename) as file:
    for line in file:
        line = line.strip()
        parts = line.split(",")
        labels.append("Walking")
        if i < len(peaktimes):
            if float(parts[2]) == peaktimes[i]:
                print(parts[2])
                for x in range(j-5, j+1):
                    labels[x] = "Stepping"
                for x in range(j+1, j+5):
                    labels.append("Stepping")
                i += 1
                j += 4 
        j += 1

with open(filename) as finalfile:
    with open(outputfilename, "w+") as outputfile:
        for finalline in finalfile:
            finalline = finalline.strip()
            outputfile.write(finalline + "," + labels[k] + "\n")
            k += 1
