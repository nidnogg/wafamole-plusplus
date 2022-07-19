import os
import seaborn as sns
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

dirname = os.getcwd()

for files in os.scandir(dirname):
    if(files.path.endswith('.txt')):
        input_filename = files.name
        runtimes = []
        results_with_runtime = []

        with open(input_filename, 'r') as input_file:
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
            probs = []
            rounds_total = []
            runtimes_clean = []
            for runtime in runtimes:
                results_with_runtime[i].append(runtime.replace('\n', ''))
                prob = results_with_runtime[i][0]
                probs.append(prob)
                rounds = results_with_runtime[i][1]
                rounds_total.append(rounds)
                time = results_with_runtime[i][2]
                runtimes_clean.append(time)
                print('{} {} {}'.format(prob, rounds, time))
                i += 1
            # data = {
            #     "probabilities": probs,
            #     "rounds_left": rounds_total,
            #     "runtimes": runtimes_clean
            #     }
            data = {
                "runtimes": runtimes_clean
                }
            df = DataFrame(data)
            bins = np.arange(min(df['runtimes']), max(df['runtimes'])+0.2, step=0.3)
            ax = sns.distplot(df['runtimes'], 
                color='red', kde=False, bins=bins, label='New')
            plt.savefig('runtimes.png')
            
    else:
        continue

# d = {"country": ['UK','US','US','UK','PRC'],
#        "age": [32, 37, 17, 34, 29],
#        "new_user": [1, 0, 0, 0,1]}

# df = DataFrame(d)
# bins = range(0, 100, 10)
# ax = sns.distplot(df.age[df.new_user==1],
#               color='red', kde=False, bins=bins, label='New')
# sns.distplot(df.age[df.new_user==0],
#          ax=ax,  # Overplots on first plot
#          color='blue', kde=False, bins=bins, label='Existing')
# legend()
# show()
# dirname = os.getcwd()
