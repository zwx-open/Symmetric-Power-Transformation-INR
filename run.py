import os
from manager import ParamManager
import subprocess
import numpy as np
import math
from config import PAMRAM_SET 


def run_single(exp_num):
    #### RUNING A DEM
    pm = ParamManager(idx=0, exp_num=exp_num)
    pm.p.up_folder_name = "debug"
    pm.p.num_epochs = 500
    pm.p.log_epoch = 50
    # test dataset
    # pm.p.input_path = "/home/wxzhang/projects/coding4paper/data"
    # pm.p.multi_data = None

    # pm.p.transform = "sym_power"
    pm.p.signal_type = "series"
    pm.p.hidden_layers = 1
    pm.p.hidden_features = 64
    use_cuda = 0
    cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
    print(f"Running: {cmd_str}")
    os.system(cmd_str)


def run_subprocess(param_sets, gpu_list, exp_num):
    processes = []

    assert len(param_sets) == len(gpu_list)

    _len = min(len(param_sets), len(gpu_list))
    param_sets = param_sets[:_len]
    gpu_list = gpu_list[:_len]

    for param_set, use_cuda in zip(param_sets, gpu_list):
        pm = ParamManager(param_set=param_set, exp_num=exp_num)
        cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
        process = subprocess.Popen(cmd_str, shell=True)
        print(f"PID: {process.pid}")
        processes.append(process)

    for process in processes:
        process.wait()


def run_tasks(exp_num, param_sets, gpu_list):

    gpus = len(gpu_list)
    rounds = math.ceil(len(param_sets) / gpus)
    print("rounds: ", rounds)

    for i in range(rounds):
        cur_param_sets = param_sets[i * gpus : min(len(param_sets), (i + 1) * gpus)]
        cur_len = len(param_sets)
        gpu_list = gpu_list[:cur_len]
        run_subprocess(cur_param_sets, gpu_list, exp_num)


if __name__ == "__main__":

    exp = "001"
    param_sets = PAMRAM_SET[exp]
    gpu_list = [i for i in range(len(param_sets))]
    
    run_tasks(exp, param_sets, gpu_list)
