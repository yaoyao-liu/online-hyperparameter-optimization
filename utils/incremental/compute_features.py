""" The functions that compute the features """
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
from utils.process_fp import process_inputs_fp

def compute_features(the_args, tg_model,  \
    is_start_iteration, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)

            the_feature = tg_model.forward_feature(inputs)

            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(the_feature.cpu())

            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
