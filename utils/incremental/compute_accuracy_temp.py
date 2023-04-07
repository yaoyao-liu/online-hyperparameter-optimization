""" The functions that compute the accuracies """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils.misc import *
from utils.imagenet.utils_dataset import merge_images_labels
from utils.process_fp import process_inputs_fp

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def compute_accuracy_temp(the_args, b1_model, the_intensity_gamma, class_means, \
    X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, is_start_iteration=False, \
    fast_fc=None, scale=None, print_info=True, device=None, cifar=True, imagenet=False, \
    valdir=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b1_model.eval()
    fast_fc = 0.0
    correct = 0
    correct_icarl = 0
    correct_icarl_cosine = 0
    correct_icarl_cosine2 = 0
    correct_ncm = 0
    correct_maml = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = b1_model(inputs)
            
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = np.squeeze(b1_model.forward_feature(inputs))
            try:
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature.cpu(), 'sqeuclidean')
            except:
                import pdb
                pdb.set_trace()
            #import pdb
            #pdb.set_trace()
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()

    if print_info:
        print("  Current accuracy (FC)         :\t\t{:.2f} %".format(100.*correct/total)) 
        print("  Current accuracy (Proto)      :\t\t{:.2f} %".format(100.*correct_icarl/total))
    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    if the_intensity_gamma==1:
        return cnn_acc
    else:
        return icarl_acc
