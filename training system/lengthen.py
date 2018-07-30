# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:53:24 2018

@author: hansos2
"""

filename = "training_out_counting_big_all.txt"
outputfilename = "training_out_counting_big_all_20.txt"

labels = []
j = 0
r = 0
z = 0

with open (filename) as file:
    for line in file:
        labels.append("Walking")

with open (filename) as parsefile:
    for parseline in parsefile:
        if j > r:
            line = parseline.strip()
            parts = parseline.split(",")
            if str(parts[1]) == "Stepping":
                for x in range(j-5, j+25):
                    labels[x] = "Stepping"
                r = j+24
        j += 1

        
with open(filename) as finalfile:
    with open(outputfilename, "a+") as outputfile:
        for finalline in finalfile:
            finalline = finalline.strip()
            parts = finalline.split(",")
            templine = parts[0] + ","+ labels[z] + "," + parts[2] + "," + parts[3] + "," + parts[4] + "," + parts[5] + "\n"
            outputfile.write(templine)
            z += 1
