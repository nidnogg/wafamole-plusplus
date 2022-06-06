import sqlparse
import json

def generator(input_filepath, json_output_filepath, type):
    """Works with dataset files in same folder.
    """
    output_data = []
    raw_queries = open(input_filepath, 'r').read() 
    queries = sqlparse.split(raw_queries)
    for row in queries:
        json_row = {}
        json_row["pattern"] = row
        if(type == "sqli"): 
            json_row["type"] = "sqli"
        elif(type == "valid"):
            json_row["type"] = "valid"
            
        output_data.append(json_row)

    # with open(json_output_filepath, 'w') as jsonFile:
    #     jsonFile.write(json.dumps(output_data, indent=None))
    with open(json_output_filepath, 'w') as jsonFile:
        jsonFile.write(json.dumps(output_data, indent = 4))

input_attacks_filepath = './attacks.sql'
json_attacks_filepath = './wafamole_dataset_attacks.json'
input_sane_filepath = './sane.sql'
json_sane_filepath = './wafamole_dataset_sane.json'

generator(input_attacks_filepath, json_attacks_filepath, "sqli")
generator(input_sane_filepath, json_sane_filepath, "valid")

