import os 
import re
import numpy as np
import statistics
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--group', type=int, default=False,)
    
args = parser.parse_args()
runs = {}
for run in os.listdir("../logs"):
    values = []
    splits = ["a1","a2","a3", "b1", "b2", "c1","c2"]    
    if run.startswith("reascan"):
        seed =  run.split("_")[-1]
        for file_path in [f"../logs/{run}/log_train.txt",f"../logs/{run}/log_test_comp.txt"]:
            try:
                with open(file_path,"r") as file:
                    res_mode = False
                    index = -1
                    for line in file:
                        if res_mode and "--------------" in line:
                            res = re.findall(r"[0-9]\.+[0-9]+",line)
                            if res:
                                acc = float(res[0])
                                values.append(round(100*acc,2))
                                res_mode = False
                        if "Test Data Path" in line:
                            res_mode = True
                            index += 1
                if args.group:
                    if str(seed).isnumeric():# and len(seed) == 4:
                        run = run.strip("_"+seed)
                        while run[-1].isnumeric():
                            run = run[:-1]
                # if res_mode:
                    # print(run)
                # if len(values) == 0:
                #     print(run, "not finished")
                # print(run, dict(zip(splits,values)), sum(values)/len(splits))
                # run = run.replace("","")

                if "default_12_6_enc_dec_layers_interleave_co_self_" in run:
                    run = run.replace("default_12_6_enc_dec_layers_interleave_co_self_","")
                if len(values) == 8:
                    values = values[1:]
                    if runs.get(run) is None:
                        runs[run] = []        
                    runs[run].append(values)
                    # print("{:<50}".format(run)," & ".join(map(lambda x:"{:<6}".format(x),values)),"&", round(sum(values)/len(values),2))
                    # print("{:<50}".format(run),seed,",".join(map(lambda x:"{:<6}".format(x),values)), round(sum(values)/len(values),2))
                else:
                    raise Exception()
            except Exception as e:
                continue
            break

new_data = []
for run,run_values in runs.items():
    try:
        total_runs = len(run_values)
        temp_values = []
        new_avgs = []
        means = []
        for index, split in enumerate(splits):
            runs_splits_values = [run_values[i][index] for i in range(total_runs)]
            if total_runs > 1:
                m = statistics.mean(runs_splits_values)
                v = statistics.stdev(runs_splits_values)
            else:
                m = runs_splits_values[0]
                v = 0
            temp_values.append(f"{round(m,2):.2f}+-{round(v,2):.2f}")
        
        new_avgs.append(m)
        avg_v = 0

        rep = "{:<55}".format(run) + f" {total_runs} " + " & ".join(map(lambda x:"{:<14}".format(x),temp_values))

        means = []
        for run_value in run_values:
            temp_m = statistics.mean(run_value[:])
            means.append(temp_m)
        if len(run_values) > 1:
            m = statistics.mean(means)
            v = statistics.stdev(means)
        else:
            m = means[0]
            v = 0
        # print(f"{round(m,2):.2f}+-{round(v,2):.2f}")
        new_data.append((rep,m,v))
    except:
        print("#######")
        pass

for (rep,m,v) in sorted(new_data,key= lambda x:x[1],reverse=True):
    print(rep,f"{round(m,2):.2f}+-{round(v,2):.2f}")