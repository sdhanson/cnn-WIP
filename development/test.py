# loop through files in folder.
# append data to csv file.
# write header only once 
0.848524113047 ,1.58163582291 ,1.58163582291, 1.52954069426, 6.57268546133, 7.38546518379, 7.38546518379, 1.61475645265, 3.01760083498, 3.01760083498, 5.76850277868, 5.01018299524, 5.01018299524, 1.41501550477, 0.754068041821, 0.754068041821, 0.753660838328, 1.96509359661, 1.96509359661, 2.82629036879, 1.45099468309, 0.998067540133, 0.998067540133, 1.57042701653, 1.57042701653, 1.00493524959, 0.558523831648, 0.655353796242, 0.655353796242, 0.661641941911, 0.856198374392, 0.856198374392, 0.808974031543, 0.831827190537, 0.831827190537, 0.969459043919, 0.880326451026, 0.456902047508, 0.456902047508, 0.241778039897, 0.241778039897, 0.262912228107, 0.290562157743, 0.72008842357, 0.72008842357, 1.39548726346, 1.97309391695, 2.55865355988, 2.55865355988, 0.15004537437, 4.68267434294, 4.68267434294, 4.18507900813, 4.59276048465, 4.59276048465, 2.72919885501, 0.912506716504, 0.912506716504, 3.15768301092, 3.57336440789, 1.64285908664, 1.64285908664, 1.14295237295, 1.27661474513, 1.27661474513, 1.09487989065, 1.40244829748, 1.40244829748, 1.1800918669, 0.747701214832, 0.747701214832, 1.23024453871, 1.29924596256, 0.685612138565, 0.685612138565, 0.442574242439, 0.630389361129, 0.630389361129, 0.521737298604, 0.443005753053, 0.443005753053, 0.425106586618, 0.575294303207, 0.575294303207, 0.574949036257, 0.510081077671, 0.510081077671, 0.468602836395, 0.486943486673, 0.567981800018
import csv
import math
import re


def main():

    #filename = input("enter filename: ") 
    filename = 'GO_1_raw'
    foldername = 'GO_v1.1'
    csvname = filename + '.csv'

    header_temp = ['User', 'Activity', 'TimeStamp', 'x_acceleration', 'y_acceleration', 'z_acceleration']

    data = []
    
    parse_text(foldername, filename, data)
    write_csv(csvname, data)
    print("finish") 


# Parse the text file. Skip the header. Ignore all data after the first integer value.
# If a line contains whitespace or alpha characters, ignore it. 
def parse_text(foldername, filename, data):
    current_line = 1; 
    num_lines = sum(1 for line in open(foldername+'/'+filename + '.txt'))
    inputfile = open(foldername+'/'+filename + '.txt') 
    
    for line in inputfile:

        #remove ;\n and sep. data by , 
        line = line[:-2] 
        temp = line.split(",")

        # error check - ensure appropriate number of inputs        
        if (len(temp) != 6):
            continue

        del temp[0] #remove user (id#)
        #del temp[1] #remove time stamp 
        
        #is label string or number?
        label= select_label(temp[0])
        if (label == 6): # undesired label
            continue
        label= temp[0]

        # select specific data
               
        #if (label != 0 and label != 1 and label != 4 and label != 5):
        
        
        #if (label != 0 and label != 5): 
        #    continue;
        #elif (label == 1): #jogging = 2
        #    label = 2; 
        #elif (label == 0): #walking = 1
        #    label = 1;     
        #elif (label == 5 or label == 4): #standing/still = 0
        #    label = 0;  

        # calculate vector magnitude Euclidean (L2) norm
        vector_mag = euclidean_norm(float(temp[2]), float(temp[3]), float(temp[4]))

        #is data vector magnitude or triaxial?
        new_data = [str(label),  temp[1], str(vector_mag)]
        #new_data = [str(vector_mag), temp[1], label] 
        #new_data = [ temp[1],  temp[2],  temp[3], label]
     

        if (current_line%10000 == 0):
            print( "Progress: " + str(100.0*(float(current_line)/num_lines)) +"%" )
        current_line +=1; 

        # append data
        data.append(new_data) 

        
    inputfile.close()
    return

#Walking: 424,400 (38.6%)
#Jogging: 342,177 (31.2%)
#Upstairs: 122,869 (11.2%)
#Downstairs: 100,427 (9.1%)
#Sitting: 59,939 (5.5%)
#Standing: 48,395 (4.4%)

def select_label(label):
    if (label == "Walking"): 
        return 0
    elif (label == "Jogging"): 
        return 1
    elif (label == "Upstairs"): 
        return 6 #2
    elif (label == "Downstairs"): 
        return 6 #3
    elif (label == "Sitting"): 
        return 6 #4
    elif (label == "Standing"): 
        return 5
    else:
        return 6 



def euclidean_norm(x, y, z): 
    return math.sqrt(x**2 + y**2 + z**2)


        
# Write data to the output file (CSV)
def write_csv(csvname, data):
    print("Write to CSV")
    current_line = 0; 
    num_lines = sum(1 for line in open('GO_v1.1'+'/'+'GO_1_raw' + '.txt'))    
    
    with open(csvname,'a') as myfile:
        myfile = csv.writer(myfile, delimiter = ',', lineterminator = '\n')
        #myfile.writerow([x for x in data])
        for i in data:
            if (i <= 1):
                continue
            
            myfile.writerow(i); 
            if (current_line%10000 == 0):
                print( "Progress: " + str(100.0*(float(current_line)/num_lines)) +"%" )         
            current_line +=1; 
    return



main()