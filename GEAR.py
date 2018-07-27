filename = "training_gear.txt"
outputfilename = "training_gear_organized.txt"

with open(filename) as file:
	with open(outputfilename, "a+") as outputfile:
		for line in file:
			line = line.strip()
			parts = line.split(",")
			templine = parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + "," + parts[4] + "," + parts[5] + "\n"
			outputfile.write(templine)