import csv
import json

def cleaner(csvFilePath, jsonOutputFilePath):
    outputData = []
    with open(csvFilePath) as csvFile:
        csvReader = csv.reader(csvFile)
        i = 0
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        for row in csvReader:
            jsonRow = {}
            if(row[1].isnumeric() == False):
                continue               
            jsonRow["pattern"] = row[0]
            if(int(row[1]) == 1): 
                jsonRow["type"] = "sqli"
            else:
                jsonRow["type"] = "valid"
                
            outputData.append(jsonRow)
            i+=1

    
    with open(jsonOutputFilePath, 'w') as jsonFile:
        jsonFile.write(json.dumps(outputData, indent=None))


filepath = './SQLiV3.csv'
jsonFilePath = './SQLiV3.json'
cleaner(filepath, jsonFilePath)
