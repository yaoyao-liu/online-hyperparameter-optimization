""" Using the aggregation weights to compute the feature maps from two branches """
import torch
import torch.nn as nn
from utils.misc import *

def process_inputs_fp(the_args, b1_model, inputs, feature_mode=False):
    if feature_mode:
        tg_feature_model = nn.Sequential(*list(b1_model.children())[:-1])
        fp_final = tg_feature_model(inputs)
        return fp_final
    else:
        outputs = b1_model(inputs)
        return outputs
