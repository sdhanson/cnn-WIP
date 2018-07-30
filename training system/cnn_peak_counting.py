# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:51:31 2018

@author: hansos2
"""
import matplotlib.pyplot as plt

#peakfilename = "training_steps_CNN_controller.txt"
filename = "training_steps_CNN_big.txt"
outputfilename = "training_out_counting_big_test.txt"

values = []
count = 0
i = 0

#NEED TO DO THE 56000 AS BEGINNING ONE
BATCH = 10
BEGIN = 110000
END = BEGIN + BATCH
#with open(peakfilename) as peakfile:
#    for peakline in peakfile:
#        count += 1
        
with open (filename) as file:
    for line in file:
        if i >= BEGIN:
            line = line.strip()
            parts = line.split(",")
            if parts[1] == "1":
                temp = []
                temp.append(float(parts[2]))
                temp.append(float(parts[4]))
                values.append(temp)
                i += 1
        else:
            i += 1
        if i == END:
            break

times = []
y = []



for val in values:
    times.append(val[0])
    y.append(val[1])


plt.plot(times, y)


def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
   pivotvalue = alist[first][1]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and alist[leftmark][1] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark][1] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark][1]
           alist[leftmark][1] = alist[rightmark][1]
           alist[rightmark][1] = temp
           temp2 = alist[leftmark][0]
           alist[leftmark][0] = alist[rightmark][0]
           alist[rightmark][0] = temp2
#have to do the swapping below and above so it swaps the whoooooole thing the time and the other
   temp = alist[first][1]
   alist[first][1] = alist[rightmark][1]
   alist[rightmark][1] = temp
   temp2 = alist[first][0]
   alist[first][0] = alist[rightmark][0]
   alist[rightmark][0] = temp2


   return rightmark

quickSort(values)

peaks = []
print(len(values))
peaks.append(values[len(values)-1])

x = 0
count = 0

STEPS = 50
WINDOW = 0.2

while x < STEPS and count < len(values):
    skip = False
    for peak in peaks:
        if values[len(values)-1 - count][0] >= peak[0] - WINDOW and values[len(values)-1 - count][0] <= peak[0] + WINDOW:
            skip = True
    if not skip:
        peaks.append(values[len(values)-1 - count])
        x += 1
    count += 1
   

peaktimes = []
peaky = []

for peak in peaks:
    peaktimes.append(peak[0])     
    peaky.append(peak[1])

plt.plot(peaktimes,peaky, 'r+')
plt.ylabel("Acceleration Y")
plt.xlabel("Time.time")
plt.savefig("training_CNN_graph_controller")
plt.show()
plt.close()


labels = []
i = 0
j = 0
k = 0
r = 0

peaktimes.sort()
BASE_ACTIVITY = "Walking"
        
with open (filename) as file:
    check = 0
    for line in file:
        if check >= BEGIN:
            labels.append(BASE_ACTIVITY)
            if check == END:
                break
        check += 1

print(len(labels))
with open (filename) as file:
    check = 0
    for line in file:
        if check >= BEGIN:
            if j > r:
                line = line.strip()
                parts = line.split(",")
                if i < len(peaktimes):
                    if float(parts[2]) == peaktimes[i]:
                        for x in range(j-4, j+6):
                            labels[x] = "Stepping"
                        i += 1
                        r = j+5
            j += 1
            if check == END:
                break
        check += 1
#
with open(filename) as finalfile:
    with open(outputfilename, "a+") as outputfile:
        check = 0
        for finalline in finalfile:
            if check >= BEGIN:
                finalline = finalline.strip()
                parts = finalline.split(",")
                templine = parts[0] + "," + labels[k] + "," + parts[2] + "," + parts[3] + "," + parts[4] + "," + parts[5] + "\n"
                outputfile.write(templine)
                k += 1
            check += 1
            if check == END:
                break

