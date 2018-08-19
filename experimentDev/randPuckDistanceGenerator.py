# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 18:43:25 2018

@author: rp
"""

import random

random.seed(1)
arrays = []

for i in range(18*7):
    array = [5,5,5,7.5,7.5,7.5,10,10,10]
    for j in range(9):
        array[j] = round(array[j] + random.uniform(-.1, .1),2)
    random.shuffle(array)
    arrays.append(array)
    
print arrays