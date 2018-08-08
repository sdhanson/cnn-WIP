# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:29:50 2018

@author: vandy
"""

from __future__ import print_function
import random

nums = []
x = 0
run = 0

while x < 70:
    num = random.randint(-89, 89)
    
    if num > -45 and num < 45:
        print("no")
    else:
        if ((run + num > 90 and num > 0) or (run + num < -90 and num < 0)):
            print("no")
        else:
            run = run + num
            nums.append(num)
            x += 1
        
for n in nums:
    print(n, end='')
    print(",", end='')