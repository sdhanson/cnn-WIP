
# loop through files in folder.
# append data to csv file.
# write header only once 

import csv



def main():

    #filename = input("enter filename: ") 
    filename = 'WISDM_ar_v1.1_raw'
    foldername = 'WISDM_v1.1'
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

        ## OR Use triaxial information 
        new_data = [ temp[0], temp[1],  temp[2],  temp[3], temp[4]. temp[5]];
        #    header_temp = ['User', 'Activity', 'TimeStamp', 'x_acceleration', 'y_acceleration', 'z_acceleration']

     

        if (current_line%10000 == 0):
            print( "Progress: " + str(100.0*(float(current_line)/num_lines)) +"%" )
        current_line +=1; 

        # append data
        data.append(new_data) 

        
    inputfile.close()
    return



        
# Write data to the output file (CSV)
def write_csv(csvname, data):
    print("Write to CSV")
    current_line = 0; 
    num_lines = sum(1 for line in open('WISDM_v1.1'+'/'+'WISDM_ar_v1.1_raw' + '.txt'))    
    
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



