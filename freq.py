# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:15:56 2018

@author: hansos2
"""

slowAvg = 0
medAvg = 0
fastAvg = 0

slowCount = 0
medCount = 0
fastCount = 0

active = False
count = 0

inputfilename = "inGoFreq.txt"

with open(inputfilename) as inputfile:
    for inputline in inputfile:
        inputline = inputline.strip()
        parts = inputline.split(";")
        if parts[1] == "False":
            active = False
        elif not active:
            count += 1
            active = True
        
        if active and count == 1:
            slowAvg += float(parts[13])
            slowCount += 1
        
        elif active and count == 2:
           medAvg += float(parts[13])
           medCount += 1 
           
        elif active and count == 3:
           fastAvg += float(parts[13])
           fastCount += 1 
        
        
        
print(slowAvg/slowCount)
print(medAvg/medCount)
print(fastAvg/fastCount)