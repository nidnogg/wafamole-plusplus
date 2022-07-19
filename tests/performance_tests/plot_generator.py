import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dirname = os.getcwd()


max_runtimes = []
min_runtimes = []
for files in os.scandir(dirname):
    if(files.path.endswith('.txt')):
        input_filename = files.name
        runtimes = []
        results_with_runtime = []

        with open(input_filename, 'r') as input_file:
            probs = []
            rounds_total = []
            runtimes_clean = []
            max_iters = 0
            for line in input_file:
                if('seconds' in line):
                    # print(line)
                    runtime = line.replace(' seconds', '')
                    runtimes.append(runtime)
                    continue
                if('max_iter' in line):
                    max_iters += 1
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
                probs.append(prob)
                rounds = results_with_runtime[i][1]
                rounds_total.append(rounds)
                time = results_with_runtime[i][2]
                runtimes_clean.append(time)
                # print('{} {} {}'.format(prob, rounds, time))
                i += 1
            # data = {
            #     "probabilities": probs,
            #     "rounds_left": rounds_total,
            #     "runtimes": runtimes_clean
            #     }
            data = {
                "runtimes": pd.to_numeric(runtimes_clean)
                }
            round_data = {
                "rounds": pd.to_numeric(rounds_total)
            }
            df = pd.DataFrame(data)
            rounds_df = pd.DataFrame(round_data)
            print(input_filename)
            max_runtime = df['runtimes'].max()
            min_runtime = df['runtimes'].min()
            mean_runtime = df['runtimes'].mean()
            mean_rounds = rounds_df['rounds'].mean()
            print(mean_rounds)
            trimmed_filename = input_filename.replace('.txt', '')
            max_runtimes.append({trimmed_filename: max_runtime})
            min_runtimes.append({trimmed_filename: min_runtime})

            # print(max_iters)
            if(max_runtime <= 6):
                bins = np.arange(min_runtime, max_runtime + 0.2, step=0.3)
                ax = sns.displot(df['runtimes'], 
                    color='orange', kde=False, bins=bins)
                ax.set(title=trimmed_filename, xlabel='Runtimes in seconds', ylabel='Amount of runtimes')
                plt.savefig('./plots/{}.png'.format(trimmed_filename), bbox_inches='tight')
            else:
                bins = np.arange(min_runtime, max_runtime + 3, step=6.0)
                ax = sns.displot(df['runtimes'], 
                    color='orange', kde=False, bins=bins, label='Runtimes')
                ax.set(title=trimmed_filename, xlabel='Runtimes in seconds', ylabel='Amount of runtimes')
                plt.savefig('./plots/{}.png'.format(trimmed_filename), bbox_inches='tight')

            
    else:
        continue

# bins = np.arange(min_runtime, max_runtime + 0.2, step=0.3)
#                 ax = sns.displot(df['runtimes'], 
#                     color='orange', kde=False, bins=bins)
#                 ax.set(xlabel='Runtimes in seconds', ylabel='Amount of runtimes')
#                 plt.savefig('./plots/{}.png'.format(trimmed_filename))


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
