""" Training code for LUCIR """
import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
import torch.nn.functional as F

cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def incremental_train_and_eval(the_args, epochs, b1_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, \
    start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list, the_lambda, dist, \
    K, lw_mr, intensity_alpha=0.5, intensity_beta=0.5, intensity_gamma=1.0, T=None, beta=None, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features

    # Get the features from the current and the reference model
    handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
    handle_cur_features = b1_model.fc.register_forward_hook(get_cur_features)
    handle_old_scores_bs = b1_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
    handle_new_scores_bs = b1_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    for epoch in range(epochs):
        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss_CE = 0
        train_loss_feature_KD = 0
        train_loss_normal_KD = 0
        # Set the counters to zeros
        correct = 0
        total = 0
    
        # Learning rate decay
        tg_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if the_args.debug_mode:
                if batch_idx>=1:
                    break
                    
            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()

            # Forward the samples in the deep networks
            outputs = process_inputs_fp(the_args, b1_model, inputs)
    
            # Loss 1: feature-level distillation loss
            #if iteration == start_iteration+1:
            ref_outputs = ref_model(inputs)
            loss_feature_KD = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * the_lambda

            loss_normal_KD = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes

            # Loss 2: classification loss
            loss_CE = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            # Loss 3: margin ranking loss
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==outputs.size())
            gt_index = torch.zeros(outputs_bs.size()).to(device)
            gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)
            max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
            hard_index = targets.lt(num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)
            if hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss_MR = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
            else:
                loss_MR = torch.zeros(1).to(device)

            # Sum up all looses
            #loss = lossCE + the_args.loss_feature_KD_weight*loss_feature_KD + the_args.loss_normal_KD_weight*loss_normal_KD + the_args.loss_MR_weight*loss_MR
            loss = loss_CE + intensity_alpha*loss_feature_KD + intensity_beta*loss_normal_KD + intensity_gamma*loss_MR

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss_CE += loss_CE.item()
            train_loss_feature_KD += loss_feature_KD.item()
            train_loss_normal_KD += loss_normal_KD.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train_loss_CE: {:.4f}, train_loss_feature_KD: {:.4f}, train_loss_normal_KD: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss_CE/(batch_idx+1), train_loss_feature_KD/(batch_idx+1), train_loss_normal_KD/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))

        # Running the test for this epoch
        b1_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = process_inputs_fp(the_args, b1_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register forward hook")
    handle_ref_features.remove()
    handle_cur_features.remove()
    handle_old_scores_bs.remove()
    handle_new_scores_bs.remove()
    return b1_model