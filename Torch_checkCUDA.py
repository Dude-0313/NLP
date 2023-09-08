# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:23:17 2023
PyTorch CUDA Check
@author: kulje
"""
import torch

def checkCUDA():
    print("CUDA available",torch.cuda.is_available())
    print("CUDA Version :", torch.version.cuda)
    print("Current available index :", torch.cuda.current_device())
    print("Device name : ",torch.cuda.get_device_name(torch.cuda.current_device()))
    return

checkCUDA()