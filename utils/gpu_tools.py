""" GPU tools. """
import os
import torch
import time

def check_memory(cuda_device):
    """ Check the total memory and occupied memory for GPU """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_memory(cuda_device):
    total, used = check_memory(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.90)
    print('Total memory: ' + str(total) + ', used memory: ' + str(used))
    block_mem = max_mem - used
    if block_mem > 0:
        x = torch.cuda.FloatTensor(256, 1024, block_mem)
        del x

def set_gpu(x):
    """ Set up which GPU we use for this process """
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('Using gpu:', x)
    

