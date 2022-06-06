import csv
import json

def cleaner(csv_file_path, json_output_filepath):
    output_data = []
    with open(csv_file_path) as csvFile:
        csv_reader = csv.reader(csvFile)
        i = 0
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        for row in csv_reader:
            json_row = {}
            if(row[1].isnumeric() == False):
                continue               
            json_row["pattern"] = row[0]
            if(int(row[1]) == 1): 
                json_row["type"] = "sqli"
            else:
                json_row["type"] = "valid"
                
            output_data.append(json_row)
            i+=1

    
    # with open(json_output_filepath, 'w') as jsonFile:
    #     jsonFile.write(json.dumps(output_data, indent=None))
    with open(json_output_filepath, 'w') as jsonFile:
        jsonFile.write(json.dumps(output_data, indent = 4))

input_filepath = './SQLiV3.csv'
json_filepath = './SQLiV3_test.json'
cleaner(input_filepath, json_filepath)
