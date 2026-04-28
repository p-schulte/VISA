import subprocess
import sys
import os
import time
from tqdm import *

if len(sys.argv) < 2:
    print("No benchmark file given.")
    sys.exit(1)

processes = 8
if len(sys.argv) >= 3:
    try:
        processes = int(sys.argv[2])
    except ValueError:
        print("The second argument should be the process count.")
        processes = 8

params_file_path = sys.argv[1]

try:
    with open(params_file_path, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"The benchmark file '{params_file_path}' could not be accessed.")
    sys.exit(1)

benchmark_name = os.path.splitext(os.path.basename(params_file_path))[0]
runs = 0
sruns = 0
stats_table_out = ''

bars = []
total_number_lines = 0
total_number_runs = 0

dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(dir_path+"/output/"):
    os.makedirs(dir_path+"/output/")

if not os.path.exists(dir_path+"/output/tables/"):
    os.makedirs(dir_path+"/output/tables/")

if not os.path.exists(dir_path+"/output/debug/"):
    os.makedirs(dir_path+"/output/debug/")


for line_num,line in enumerate(lines):
    if len(line) >= 1:
        
        parts = line.strip().split()

        try:
            num_runs = int(parts[0])
            params = parts[1:]
        except ValueError:
            print("The first item of each line must be a natural number stating the number of testruns.")
            continue

        bars.append(tqdm(total=num_runs, position=total_number_lines))
        total_number_lines += 1
        total_number_runs += num_runs

bars.append(tqdm(total=total_number_runs, position=total_number_lines))

iteration_index = -1
max_seen_combis = 0
for line_num,line in enumerate(lines):
    if len(line) < 1:
        continue
    parts = line.strip().split()

    try:
        num_runs = int(parts[0])
        params = parts[1:]
    except ValueError:
        print("The first item of each line must be a natural number stating the number of testruns.")
        continue

    iteration_index += 1 

    max_time = 0
    srun = 0
    avg_size = 0
    all_survived_combis = 0
    #print('\n {}\n'.format(params))
    size, seen_combis, survived_combis, num_traces = None, None, None, None

    for run in range(num_runs):
        runs +=1
        output_file = '{}_{}_{:02d}'.format(benchmark_name,line_num,run)
        time_start = time.time()
        result = subprocess.run(['python', 'graph_seperator.py'] + params + ['-o', output_file,'-p', str(processes)],
            capture_output = True, text = True)
        time_end = time.time()
        max_time = max(max_time,time_end-time_start)
        out_path = 'output/debug/{}_stdout.txt'.format(output_file)
        out_path_1 = 'output/debug/{}_stdout_err.txt'.format(output_file)
        
        out_path_1 = 'output/debug/{}_stdout_err1.txt'.format(output_file)
        out_path_2 = 'output/debug/{}_stdout_err2.txt'.format(output_file)

        for param_position, cur_param in enumerate(params):
            if cur_param == "-ln":
                num_traces = params[(param_position+1)]
                break

        res_string = result.stdout

        if result.stderr is not None:
            with open(out_path_1, "w") as text_file:
                text_file.write(result.stderr)
            with open(out_path_2, "w") as text_file:
                text_file.write(result.stdout)    

        res_string = result.stdout
        res = res_string.split('??????????')
        
        res_thing = res[1]
        learning_values = res_thing.split(',')
        size, seen_combis, survived_combis = int(learning_values[0]), int(learning_values[1]), int(learning_values[2])

        max_seen_combis = max(max_seen_combis, seen_combis)
        all_survived_combis += survived_combis

        avg_size += size
        with open(out_path, "w") as text_file:
            text_file.write(res[0])

        bars[iteration_index].update(1)
        bars[-1].update(1)

        if result.returncode == 0:
            sruns +=1
            srun +=1
    if num_traces is None:
        stats_table_out += '&${:3.1f}$&${:6} $&${:5.0f}\seconds$&${:3.0f}\%$'.format(all_survived_combis/num_runs,avg_size//num_runs,max_time, srun*100/num_runs)
    else:
        stats_table_out += '&${:3.1f}$&${:2} \\times {:6} $&${:5.0f}\seconds$&${:3.0f}\%$'.format(all_survived_combis/num_runs,num_traces,avg_size//num_runs,max_time, srun*100/num_runs)
stats_table_out = '&${:5}$   {}'.format(max_seen_combis, stats_table_out)
stats_table_out=stats_table_out+'\n'
tqdm.write(stats_table_out)
# print(stats_table_out)
out_path = 'output/tables/{}_table.txt'.format(benchmark_name)
with open(out_path, "w") as text_file:
    text_file.write(stats_table_out)
tqdm.write('({}/{}) runs successful'.format(sruns,runs))