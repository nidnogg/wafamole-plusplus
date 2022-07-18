import os
import pandas as pd
import re

dirname = os.getcwd()

for files in os.scandir(dirname):
    if(files.path.endswith('.txt')):
        input_filename = files.name
        output_filename = './excel/' + input_filename + '_excel'
        runtimes = []
        results_with_runtime = []

        with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
            for line in input_file:
                if('seconds' in line):
                    # print(line)
                    runtime = line.replace(' seconds', '')
                    runtimes.append(runtime)
                    continue
                if('max_iter' in line):
                    continue
                if('e-' in line):
                    sci_notation_num = line.split()[0]
                    float_converted_num = format(float(sci_notation_num), '.6f')
                    results_with_runtime.append([float_converted_num, line.split()[1]])
                else:
                    results_with_runtime.append(line.split())
            
            i = 0
            for runtime in runtimes:
                results_with_runtime[i].append(runtime.replace('\n', ''))
                prob = results_with_runtime[i][0]
                rounds = results_with_runtime[i][1]
                time = results_with_runtime[i][2]
                print('{} {} {}'.format(prob, rounds, time), file=output_file)
                i += 1
    else:
        continue

excel_dirname = dirname + '/excel'
for files in os.scandir(excel_dirname):
    try:    
        input_filename = './excel/' + files.name
        output_filename = input_filename.replace('.txt_excel', '.xlsx')
        data = pd.read_csv(input_filename, sep=" ", header=None, encoding='utf-8')
        data.columns = ["Probability", "Rounds left", "Runtime"]
        data.to_excel(output_filename, 'Main data', index=False)
    except Exception as inst:
        pass

for files in os.scandir(excel_dirname):
    if(files.path.endswith('.txt_excel')):
        os.remove(files.path)