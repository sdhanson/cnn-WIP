# loop through files in folder.
# append data to csv file.
# write header only once 

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
        # label= temp[0]

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
        new_data = [label,  temp[1], str(vector_mag)]
        #new_data = [str(vector_mag), temp[1], label] 
        #new_data = [ temp[1],  temp[2],  temp[3], label]
     

        if (current_line%10000 == 0):
            print( "Progress: " + str(100.0*(float(current_line)/num_lines)) +"%" )
        current_line +=1; 

        # append data
        data.append(new_data) 

        
    inputfile.close()
    return


#Standing
#Walking
#Stepping

def select_label(label):
    #walking = 1 standing = 0
    if (label == "0"): 
        return 0
    elif (label == "1"): 
        return 1
    # elif (label == "Stepping"): 
    #     return 2
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