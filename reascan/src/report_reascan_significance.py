import os 
import re
import numpy as np
import statistics
import argparse
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()

parser.add_argument('--a', type=str, default=False,required=True)
parser.add_argument('--b', type=str, default=False,required=True)
parser.add_argument('--mode', type=str, default="single",choices=["single","avg"])
    
args = parser.parse_args()
runs = {}
splits = ["a1","a2","a3", "b1", "b2", "c1","c2"]    

for run in os.listdir("../logs"):
    values = []
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
            seed =  run.split("_")[-1]
            if str(seed).isnumeric():# and len(seed) == 4:
                run = run.strip("_"+seed)
                while run[-1].isnumeric():
                    run = run[:-1]
            if len(values) == len(splits) + 1:
                values = values[1:]
                if runs.get(run) is None:
                    runs[run] = []        
                runs[run].append(values)
            else:
                raise Exception("File corrupted")
        except Exception as e:
            continue
        break

if args.mode == "single":
    for split_index,split in enumerate(splits):    
        split_values_a = [runs[args.a][i][split_index] for i in range(len(runs[args.a]))]
        split_values_b = [runs[args.b][i][split_index] for i in range(len(runs[args.a]))]
        t_stat, p_val = ttest_ind(split_values_a, split_values_b)
        alpha = 0.05
        if p_val < alpha:
            print(split, "Sinigicant",p_val)
        else:
            print(split, "not Sinigicant",p_val)
elif args.mode == "avg":
    split_values_a = ([statistics.mean(i) for i in runs[args.a]])
    split_values_b = ([statistics.mean(i) for i in runs[args.b]])
    t_stat, p_val = ttest_ind(split_values_a, split_values_b)
    alpha = 0.05
    if p_val < alpha:
        print("Sinigicant",p_val)
    else:
        print("not Sinigicant",p_val)
