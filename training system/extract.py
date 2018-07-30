# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:15:33 2018

@author: hansos2
"""

filename = "training_out_counting_big_all_30.txt"
outputfilename = "tns.txt"
outputfilename2 = "tws.txt"

with open(filename) as finalfile:
    with open(outputfilename, "a+") as outputfile:
        with open(outputfilename2, "a+") as outputfile2:
            for finalline in finalfile:
                finalline = finalline.strip()
                parts = finalline.split(",")
                templine = parts[0] + ","+ parts[1] + "," + parts[2] + "," + parts[3] + "," + parts[4] + "," + parts[5] + "\n"
                if parts[1] != "Stepping":
                    outputfile.write(templine)
                else:
                    outputfile2.write(templine)
