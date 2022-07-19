import subprocess
import time

total_runs = 500
file_name = './token_linear_svm.txt'
# model_location = "../../wafamole/models/custom/svc/test_ada_classifier.dump"
# model_location = "../../wafamole/models/custom/example_models/waf-brain.h5"
model_location="../../wafamole/models/custom/example_models/lin_svm_trained.dump"
# model_type = "svc"
# model_type = "waf-brain"
model_type = "token"
i = 0
with open(file_name, 'a') as outfile:
    while i < total_runs:
        bash_cmd = ["wafamole", "evade", "--model-type", model_type, model_location, "admin' OR 1=1#"]
        start_time = time.perf_counter()
        process = subprocess.run(bash_cmd, stdout=outfile)
        end_time = time.perf_counter()
        print("{:.8} seconds".format((end_time - start_time)), file=outfile)
        i+=1


