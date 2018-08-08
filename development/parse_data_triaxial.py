
# loop through files in folder.
# append data to csv file.
# write header only once 

import csv



def main():

    #filename = input("enter filename: ") 
    filename = 'GO_1_raw_GEAR'
    foldername = 'GO_v1.1'
    csvname = filename + '.csv'

    #header_temp = ['User', 'Activity', 'TimeStamp', 'x_acceleration', 'y_acceleration', 'z_acceleration']

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



        if (len(temp) != 6):
            continue

        del temp[0] #remove user (id#)
        #del temp[1] #remove time stamp 
        
        #is label string or number?
        label = select_label(temp[0])
        if (label == 6): # undesired label
            continue

        ## OR Use triaxial information 
        new_data = [ label, temp[1],  temp[2],  temp[3], temp[4]];
        #    header_temp = ['User', 'Activity', 'TimeStamp', 'x_acceleration', 'y_acceleration', 'z_acceleration']

     

        if (current_line%10000 == 0):
            print( "Progress: " + str(100.0*(float(current_line)/num_lines)) +"%" )
        current_line +=1; 

        # append data
        data.append(new_data) 

        
    inputfile.close()
    return

def select_label(label):
    #walking = 1 standing = 0
    if (label == "0"): 
        return 0
    elif (label == "1"): 
        return 1
    elif(label == "Walking"):
        return 1
    elif(label == "Standing"):
        return 0
    elif(label == "Looking"):
        return 2
    else:
        return 6 

        
# Write data to the output file (CSV)
def write_csv(csvname, data):
    print("Write to CSV")
    current_line = 0; 
    num_lines = sum(1 for line in open('GO_v1.1'+'/'+'GO_1_raw_GEAR' + '.txt'))    
    
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



