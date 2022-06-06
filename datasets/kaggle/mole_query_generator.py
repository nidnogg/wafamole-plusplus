import subprocess
import json

f = open('SQLiV3.json')
queries = json.load(f)

i = 0
for query in queries:
    # print(query['pattern'] + '\n \n')
    bash_cmd = ["wafamole", "evade", "--model-type", "svc", "../wafamole/models/custom/svc/test_svc_classifier_no_mole.dump", query['pattern']]
    # print("command to run - {}".format(bashCmd))
    process = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    i+=1
    if(i == 7000): break