""" Class-incremental learning base trainer. """
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
import numpy as np
import time
import wandb
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_resnet as modified_resnet
import models.modified_resnetmtl as modified_resnetmtl
import models.modified_vit as modified_vit
import models.modified_efficientnet as modified_efficientnet
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_features import compute_features
from utils.incremental.compute_accuracy import compute_accuracy
from utils.incremental.compute_accuracy_temp import compute_accuracy_temp
from utils.misc import process_mnemonics
from trainer.mixed import incremental_train_and_eval as incremental_train_and_eval_mixed
import warnings
warnings.filterwarnings('ignore')

class BaseTrainer(object):
    """The class that contains the code for base trainer class."""
    def __init__(self, the_args):
        """The function to initialize this class.
        Args:
          the_args: all inputted parameter.
        """
        self.args = the_args
        self.set_save_path()
        self.set_cuda_device()
        self.set_dataset_variables()
        if self.args.debug_mode:
            self.args.epochs=1


    def set_save_path(self):
        """The function to set the saving path."""
        if self.args.using_msr_server:
            self.log_dir = '/mnt/output/logs/'
        else:
            self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_label_all = self.args.dataset + \
            '_nfg' + str(self.args.nb_cl_fg) + \
            '_ncls' + str(self.args.nb_cl) + \
            '_nproto' + str(self.args.nb_protos) + \
            '_' + self.args.baseline + \
            '_' + self.args.net_arch + \
            '_' + self.args.branch_type + \
            '_featureKDw' + str(self.args.loss_feature_KD_weight) + \
            '_normalKDw' + str(1.0-self.args.loss_feature_KD_weight) + \
            '_MRw' + str(self.args.loss_MR_weight) + \
            '_epoch' + str(self.args.epochs) + \
            '_nps' + str(self.args.num_phase_search)        

        if self.args.cb_finetune:
            self.save_label_all += '_cbf' + str(self.args.ft_flag)

        if self.args.dynamic_budget:
            self.save_label_all += '_dynamic'
        else:
            self.save_label_all += '_fixed'  

        self.save_label_all += '_' + str(self.args.ckpt_label)

        if self.args.using_msr_server:
            self.train_writer = SummaryWriter('/mnt/output/runs_new/'+self.save_label_all)
        else:
            self.train_writer = SummaryWriter('./runs/'+self.save_label_all)

        self.save_path = self.log_dir + self.save_label_all
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path) 


    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       

    def set_dataset_variables(self):
        """The function to set the dataset parameters."""
        if self.args.dataset == 'cifar100':
            # Set CIFAR-100
            # Set the pre-processing steps for training set
            if self.args.net_arch=='resnet32':
                self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
                # Set the pre-processing steps for test set
                self.transform_test = transforms.Compose([transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            elif self.args.net_arch=='std_resnet18' or 'std_resnet34' or 'std_resnet50' or 'std_resnet101' or 'efficientnet_b0' or 'efficientnet_b1' or 'efficientnet_b2' or 'efficientnet_b3' or 'efficientnet_b4' or 'efficientnet_b5' or 'efficientnet_b6' or 'efficientnet_b7':
                self.transform_train = transforms.Compose([transforms.Resize(256), \
                    transforms.RandomCrop(224, padding=4), \
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
                # Set the pre-processing steps for test set
                self.transform_test = transforms.Compose([transforms.Resize(256), \
                    transforms.CenterCrop(224), \
                    transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            elif self.args.net_arch=='vit':
                self.transform_train = transforms.Compose([transforms.Resize(256), \
                    transforms.RandomCrop(256, padding=4), \
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
                # Set the pre-processing steps for test set
                self.transform_test = transforms.Compose([transforms.Resize(256), \
                    transforms.CenterCrop(256), \
                    transforms.ToTensor(), \
                    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),])
            else:
                raise ValueError('Please set the correct network architecture.')
            # Initial the dataloader
            self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
            self.largetrainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
            self.smalltestset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_test)
            self.balancedset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=self.transform_train)
            # Set the network architecture
            if self.args.net_arch=='resnet32':
                self.network = modified_resnet_cifar.resnet32
                self.network_mtl = modified_resnetmtl_cifar.resnetmtl32
            elif self.args.net_arch=='std_resnet18':
                self.network = modified_resnet.resnet18
                self.network_mtl = modified_resnetmtl.resnetmtl18
            elif self.args.net_arch=='std_resnet34':
                self.network = modified_resnet.resnet34
                self.network_mtl = modified_resnetmtl.resnetmtl34
            elif self.args.net_arch=='std_resnet50':
                self.network = modified_resnet.resnet50
                self.network_mtl = modified_resnetmtl.resnetmtl50
            elif self.args.net_arch=='std_resnet101':
                self.network = modified_resnet.resnet101
                self.network_mtl = modified_resnetmtl.resnetmtl101     
            elif self.args.net_arch=='vit': 
                self.network = modified_vit.vit_cifar
                self.network_mtl = self.network       
            elif self.args.net_arch=='efficientnet_b0': 
                self.network = modified_efficientnet.efficientnet_b0
                self.network_mtl = self.network            
            elif self.args.net_arch=='efficientnet_b1': 
                self.network = modified_efficientnet.efficientnet_b1
                self.network_mtl = self.network     
            elif self.args.net_arch=='efficientnet_b2': 
                self.network = modified_efficientnet.efficientnet_b2
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b3': 
                self.network = modified_efficientnet.efficientnet_b3
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b4': 
                self.network = modified_efficientnet.efficientnet_b4
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b5': 
                self.network = modified_efficientnet.efficientnet_b5
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b6': 
                self.network = modified_efficientnet.efficientnet_b6
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b7': 
                self.network = modified_efficientnet.efficientnet_b7
                self.network_mtl = self.network   
            else:
                raise ValueError('Please set the correct network architecture.')
            # Set the learning rate decay parameters
            self.lr_strat = [int(self.args.epochs*0.5), int(self.args.epochs*0.75)]
            # Set the dictionary size
            self.dictionary_size = 500
            
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            if self.args.imgnet_split == 'icarl':
                # Set imagenet-subset and imagenet
                # Set the data directories
                if self.args.dataset == 'imagenet_sub':
                    traindir = os.path.join('data/seed_1993_subset_100_imagenet/data', 'train')
                    valdir = os.path.join('data/seed_1993_subset_100_imagenet/data', 'val') 
                else:            
                    traindir = os.path.join('data/imagenet/data', 'train')
                    valdir = os.path.join('data/imagenet/data', 'val') 
                # Set the dataloaders
                if self.args.net_arch=='std_resnet18' or 'std_resnet34' or 'std_resnet50' or 'std_resnet101' or 'efficientnet_b0' or 'efficientnet_b1' or 'efficientnet_b2' or 'efficientnet_b3' or 'efficientnet_b4' or 'efficientnet_b5' or 'efficientnet_b6' or 'efficientnet_b7':
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    self.trainset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), \
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
                    self.largetrainset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), \
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
                    self.testset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256), \
                        transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))
                    self.smalltestset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256), \
                        transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))
                    self.evalset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256), \
                        transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))
                    self.balancedset =  datasets.ImageFolder(traindir, transforms.Compose([transforms.Resize(256), \
                        transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))
                elif self.args.net_arch=='vit':
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    self.trainset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), \
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
                    self.largetrainset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), \
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
                    self.testset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(270), \
                        transforms.CenterCrop(256), transforms.ToTensor(), normalize,]))
                    self.smalltestset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(270), \
                        transforms.CenterCrop(256), transforms.ToTensor(), normalize,]))
                    self.evalset =  datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(270), \
                        transforms.CenterCrop(256), transforms.ToTensor(), normalize,]))
                    self.balancedset =  datasets.ImageFolder(traindir, transforms.Compose([transforms.Resize(270), \
                        transforms.CenterCrop(256), transforms.ToTensor(), normalize,]))
                else:
                    raise ValueError('Please set the correct network architecture.')
            elif self.args.imgnet_split == 'podnet':
                if self.args.dataset == 'imagenet_sub':
                    traindir = os.path.join('data/seed_1993_subset_100_imagenet/data', 'train')
                    valdir = os.path.join('data/seed_1993_subset_100_imagenet/data', 'val') 
                else:            
                    traindir = os.path.join('data/imagenet/data', 'train')
                    valdir = os.path.join('data/imagenet/data', 'val') 

                train_transforms = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=63 / 255)
                ]
                test_transforms = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
                common_transforms = [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]

                train_trsf = transforms.Compose([*train_transforms, *common_transforms])
                test_trsf = transforms.Compose([*test_transforms, *common_transforms])

                if self.args.net_arch=='std_resnet18' or 'std_resnet34' or 'std_resnet50' or 'std_resnet101' or 'efficientnet_b0' or 'efficientnet_b1' or 'efficientnet_b2' or 'efficientnet_b3' or 'efficientnet_b4' or 'efficientnet_b5' or 'efficientnet_b6' or 'efficientnet_b7':
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    self.trainset = datasets.ImageFolder(traindir, train_trsf)
                    self.largetrainset = datasets.ImageFolder(traindir, train_trsf)
                    self.testset =  datasets.ImageFolder(valdir, test_trsf)
                    self.smalltestset =  datasets.ImageFolder(valdir, test_trsf)
                    self.evalset =  datasets.ImageFolder(valdir, test_trsf)
                    self.balancedset =  datasets.ImageFolder(traindir, traindir)
                else:
                    raise ValueError('Please set the correct network architecture.')
            else:
                raise ValueError('Please set the correct imagenet split.')
            # Set the network architecture
            if self.args.net_arch=='std_resnet18':
                self.network = modified_resnet.resnet18
                self.network_mtl = modified_resnetmtl.resnetmtl18
            elif self.args.net_arch=='std_resnet34':
                self.network = modified_resnet.resnet34
                self.network_mtl = modified_resnetmtl.resnetmtl34
            elif self.args.net_arch=='std_resnet50':
                self.network = modified_resnet.resnet50
                self.network_mtl = modified_resnetmtl.resnetmtl50
            elif self.args.net_arch=='std_resnet101':
                self.network = modified_resnet.resnet101
                self.network_mtl = modified_resnetmtl.resnetmtl101     
            elif self.args.net_arch=='vit': 
                self.network = modified_vit.vit_cifar
                self.network_mtl = self.network       
            elif self.args.net_arch=='efficientnet_b0': 
                self.network = modified_efficientnet.efficientnet_b0
                self.network_mtl = self.network            
            elif self.args.net_arch=='efficientnet_b1': 
                self.network = modified_efficientnet.efficientnet_b1
                self.network_mtl = self.network     
            elif self.args.net_arch=='efficientnet_b2': 
                self.network = modified_efficientnet.efficientnet_b2
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b3': 
                self.network = modified_efficientnet.efficientnet_b3
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b4': 
                self.network = modified_efficientnet.efficientnet_b4
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b5': 
                self.network = modified_efficientnet.efficientnet_b5
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b6': 
                self.network = modified_efficientnet.efficientnet_b6
                self.network_mtl = self.network   
            elif self.args.net_arch=='efficientnet_b7': 
                self.network = modified_efficientnet.efficientnet_b7
                self.network_mtl = self.network   
            else:
                raise ValueError('Please set the correct backbone.')
            # Set the learning rate decay parameters
            self.lr_strat = [int(self.args.epochs*0.333), int(self.args.epochs*0.667)]
            # Set the dictionary size
            self.dictionary_size = 1500
        else:
            raise ValueError('Please set the correct dataset.')

    def get_podnet_imgnet(self, train=True, imagenet_size=100):

        split = "train" if train else "val"
        podnet_imgnet_base_dir = 'data/podnet-imgnet'

        print("Loading metadata of ImageNet_{} ({} split).".format(imagenet_size, split))
        metadata_path = os.path.join(podnet_imgnet_base_dir, "{}_{}.txt".format(split, imagenet_size)
        )

        data_all, targets_all = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")
                data_all.append(os.path.join(podnet_imgnet_base_dir, path))
                data_all.append(int(target))

        data_all = np.array(data_all)

        return data_all, targets_all

    def map_labels(self, order_list, Y_set):
        """The function to map the labels according to the class order list.
        Args:
          order_list: the class order list.
          Y_set: the target labels before mapping
        Return:
          map_Y: the mapped target labels
        """
        map_Y = []
        for idx in Y_set:
            map_Y.append(order_list.index(idx))
        map_Y = np.array(map_Y)
        return map_Y

    def set_dataset(self):
        """The function to set the datasets.
        Returns:
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
          X_valid_total: an array that contains all validation samples
          Y_valid_total: an array that contains all validation labels 
        """
        if self.args.dataset == 'cifar100':
            X_train_total = np.array(self.trainset.data)
            Y_train_total = np.array(self.trainset.targets)
            X_valid_total = np.array(self.testset.data)
            Y_valid_total = np.array(self.testset.targets)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            X_train_total, Y_train_total = split_images_labels(self.trainset.imgs)
            X_valid_total, Y_valid_total = split_images_labels(self.testset.imgs)
        else:
            raise ValueError('Please set the correct dataset.')

        return X_train_total, Y_train_total, X_valid_total, Y_valid_total    

    def init_class_order(self):
        """The function to initialize the class order.
        Returns:
          order: an array for the class order
          order_list: a list for the class order
        """
        # Set the random seed according to the config
        np.random.seed(self.args.random_seed)
        # Set the name for the class order file
        order_name = osp.join(self.save_path, "seed_{}_{}_order.pkl".format(self.args.random_seed, self.args.dataset))
        # Print the name for the class order file
        print("Order name:{}".format(order_name))
        
        if osp.exists(order_name):
            # If we have already generated the class order file, load it
            print("Loading the saved class order")
            order = utils.misc.unpickle(order_name)
        else:
            # If we don't have the class order file, generate a new one
            print("Generating a new class order")
            order = np.arange(self.args.num_classes)
            np.random.shuffle(order)
            utils.misc.savepickle(order, order_name)
        # Transfer the array to a list
        order_list = list(order)
        # Print the class order
        print(order_list)
        return order, order_list

    def init_prototypes(self, dictionary_size, order, X_train_total, Y_train_total):
        """The function to intialize the prototypes.
           Please note that the prototypes here contains all training samples.
           alpha_dr_herding contains the indexes for the selected exemplars
        Args:
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
          order: the class order
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
        Returns:
          alpha_dr_herding: an empty array to store the indexes for the exemplars
          prototypes: an array contains all training samples for all phases
        """
        # Set an empty to store the indexes for the selected exemplars
        alpha_dr_herding  = np.zeros((int(self.args.num_classes/self.args.nb_cl), dictionary_size, self.args.nb_cl), np.float32)
        if self.args.dataset == 'cifar100':
            # CIFAR-100, directly load the tensors for the training samples
            prototypes = np.zeros((self.args.num_classes, dictionary_size, X_train_total.shape[1], X_train_total.shape[2], X_train_total.shape[3]))
            for orde in range(self.args.num_classes):
                prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # ImageNet, save the paths for the training samples if an array
            prototypes = [[] for i in range(self.args.num_classes)]
            for orde in range(self.args.num_classes):
                prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
            prototypes = np.array(prototypes)
        else:
            raise ValueError('Please set correct dataset.')
        return alpha_dr_herding, prototypes

    def init_current_phase_model(self, iteration, start_iter, b1_model):
        """The function to intialize the models for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          b1_model: the 1st branch model from last phase
        Returns:
          b1_model: the 1st branch model from the current phase
          ref_model: the 1st branch model from last phase (frozen, not trainable)
          the_lambda_mult, cur_the_lambda: the_lambda-related parameters for the current phase
          last_iter: the iteration index for last phase
        """
        if iteration == start_iter:
            # The 0th phase
            # Set the index for last phase to 0
            last_iter = 0
            # For the 0th phase, use the conventional ResNet
            b1_model = self.network(num_classes=self.args.nb_cl_fg)
            if self.args.multiple_gpu:
                b1_model = nn.DataParallel(b1_model)
                b1_model = b1_model.cuda()
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features = b1_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            # The 2nd branch and the reference model are not used, set them to None
            ref_model = None
            the_lambda_mult = None
        elif iteration == start_iter+1:
            # The 1st phase
            # Update the index for last phase
            last_iter = iteration
            # Copy and freeze the 1st branch model
            ref_model = copy.deepcopy(b1_model)
            # Set the 1st branch for the 1st phase
            if self.args.branch_type == 'ss':
                b1_model = self.network_mtl(num_classes=self.args.nb_cl_fg)
            else:
                b1_model = self.network(num_classes=self.args.nb_cl_fg)
            # Load the model parameters trained last phase to the current phase model
            ref_dict = ref_model.state_dict()
            tg_dict = b1_model.state_dict()
            tg_dict.update(ref_dict)
            b1_model.load_state_dict(tg_dict)
            if self.args.multiple_gpu:
                b1_model = nn.DataParallel(b1_model)
                b1_model = b1_model.cuda()
            b1_model.to(self.device)
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features = b1_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
            # Set the final FC layer for classification
            new_fc.fc1.weight.data = b1_model.fc.weight.data
            new_fc.sigma.data = b1_model.fc.sigma.data
            b1_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = out_features*1.0 / self.args.nb_cl
        else:
            # The i-th phase, i>=2
            # Update the index for last phase
            last_iter = iteration
            # Copy and freeze the 1st branch model
            ref_model = copy.deepcopy(b1_model)
            # Get the information about the input and output features from the network
            in_features = b1_model.fc.in_features
            out_features1 = b1_model.fc.fc1.out_features
            out_features2 = b1_model.fc.fc2.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features1+out_features2)
            # Set the final FC layer for classification
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = b1_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = b1_model.fc.fc2.weight.data
            new_fc.sigma.data = b1_model.fc.sigma.data
            b1_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = (out_features1+out_features2)*1.0 / (self.args.nb_cl)

        # Update the current lambda value for the current phase
        if iteration > start_iter:
            cur_the_lambda = self.args.the_lambda * math.sqrt(the_lambda_mult)
        else:
            cur_the_lambda = self.args.the_lambda
        return b1_model, ref_model, the_lambda_mult, cur_the_lambda, last_iter

    def init_current_phase_dataset(self, iteration, start_iter, last_iter, order, order_list, \
        X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
        X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
        X_protoset_cumuls, Y_protoset_cumuls):
        """The function to intialize the dataset for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          last_iter: the iteration index for last phase
          order: the array for the class order
          order_list: the list for the class order
          X_train_total: the array that contains all training samples
          Y_train_total: the array that contains all training labels 
          X_valid_total: then array that contains all validation samples
          Y_valid_total: the array that contains all validation labels 
          X_train_cumuls: the array that contains old training samples
          Y_train_cumuls: the array that contains old training labels 
          X_valid_cumuls: the array that contains old validation samples
          Y_valid_cumuls: the array that contains old validation labels 
          X_protoset_cumuls: the array that contains old exemplar samples
          Y_protoset_cumuls: the array that contains old exemplar labels
        Returns:
          indices_train_10: the indexes of new-class samples
          X_train_cumuls: an array that contains old training samples, updated
          Y_train_cumuls: an array that contains old training labels, updated 
          X_valid_cumuls: an array that contains old validation samples, updated
          Y_valid_cumuls: an array that contains old validation labels, updated
          X_protoset_cumuls: an array that contains old exemplar samples, updated
          Y_protoset_cumuls: an array that contains old exemplar labels, updated
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
          X_valid_ori: an array that contains the 0th-phase validation samples, updated
          Y_valid_ori: an array that contains the 0th-phase validation labels, updated
          X_protoset: an array that contains the exemplar samples
          Y_protoset: an array that contains the exemplar labels
        """

        # Get the indexes of new-class samples (including training and test)

        indices_train_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_train_total])
        indices_test_10 = np.array([i in order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)] for i in Y_valid_total])

        indices_train_10_large = []
        indices_train_10_small = []
        cls_list = list(order[range(last_iter*self.args.nb_cl,(iteration+1)*self.args.nb_cl)])
        cls_counter_array = np.zeros(len(cls_list))
        for the_item_idx in range(len(indices_train_10)):
            the_item = indices_train_10[the_item_idx]
            the_class_label = Y_train_total[the_item_idx]
            if the_item==True:
                the_idx_in_cls_counter_array = cls_list.index(the_class_label)
                if cls_counter_array[the_idx_in_cls_counter_array]<int(self.args.nb_protos/2):
                    cls_counter_array[the_idx_in_cls_counter_array] += 1
                    indices_train_10_small.append(True)
                    indices_train_10_large.append(False)
                else:
                    indices_train_10_small.append(False)
                    indices_train_10_large.append(True)    
            else:
                indices_train_10_small.append(False)
                indices_train_10_large.append(False)                                    

        indices_train_old = np.array([i in order[range(0,(iteration)*self.args.nb_cl)] for i in Y_train_total])
        indices_test_old = np.array([i in order[range(0,(iteration)*self.args.nb_cl)] for i in Y_valid_total])
                
        # Get the samples according to the indexes
        X_train = X_train_total[indices_train_10]
        X_train_small = X_train_total[indices_train_10_small]
        X_train_large = X_train_total[indices_train_10_large]
        X_valid = X_valid_total[indices_test_10]

        X_train_new = X_train_total[indices_train_10]
        X_valid_new = X_valid_total[indices_test_10]
        Y_valid_new = Y_valid_total[indices_test_10]

        X_train_old = X_train_total[indices_train_old]
        X_valid_old = X_valid_total[indices_test_old]
        Y_valid_old = Y_valid_total[indices_test_old]

        # Add the new-class samples to the cumulative X array
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)
        X_train_cumul = np.concatenate(X_train_cumuls)

        # Get the labels according to the indexes, and add them to the cumulative Y array
        Y_train = Y_train_total[indices_train_10]
        Y_train_small = Y_train_total[indices_train_10_small]
        Y_train_large = Y_train_total[indices_train_10_large]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        Y_train_cumul = np.concatenate(Y_train_cumuls)

        if iteration == start_iter:
            # Save the 0th-phase validation samples and labels 
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            for this_idx in range(len(X_protoset_cumuls)):
                if this_idx==0:
                    X_protoset_1 = X_protoset_cumuls[this_idx][0:int(0.5*len(X_protoset_cumuls[this_idx]))]
                    Y_protoset_1 = Y_protoset_cumuls[this_idx][0:int(0.5*len(Y_protoset_cumuls[this_idx]))]
                    X_protoset_2 = X_protoset_cumuls[this_idx][int(0.5*len(X_protoset_cumuls[this_idx])):]
                    Y_protoset_2 = Y_protoset_cumuls[this_idx][int(0.5*len(Y_protoset_cumuls[this_idx])):]
                else:
                    X_protoset_1 = np.concatenate((X_protoset_1, X_protoset_cumuls[this_idx][0:int(0.5*len(X_protoset_cumuls[this_idx]))]),axis=0)
                    Y_protoset_1 = np.concatenate((Y_protoset_1, Y_protoset_cumuls[this_idx][0:int(0.5*len(Y_protoset_cumuls[this_idx]))]))
                    X_protoset_2 = np.concatenate((X_protoset_2, X_protoset_cumuls[this_idx][int(0.5*len(X_protoset_cumuls[this_idx])):]))
                    Y_protoset_2 = np.concatenate((Y_protoset_2, Y_protoset_cumuls[this_idx][int(0.5*len(Y_protoset_cumuls[this_idx])):]))                    
            # Update the exemplar set
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            X_train_small = np.concatenate((X_train_small,X_protoset_1),axis=0)
            Y_train_small = np.concatenate((Y_train_small,Y_protoset_1))

            # Create the training samples/labels for the current phase training
            X_train = np.concatenate((X_train,X_protoset),axis=0)
            Y_train = np.concatenate((Y_train,Y_protoset))
            X_train_large = np.concatenate((X_train_large,X_protoset_2),axis=0)
            Y_train_large = np.concatenate((Y_train_large,Y_protoset_2))

            #import pdb
            #pdb.set_trace()

        # Generate the mapped labels, according the order list
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_train_small = np.array([order_list.index(i) for i in Y_train_small])
        map_Y_train_large = np.array([order_list.index(i) for i in Y_train_large])
        map_Y_train_cumul = np.array([order_list.index(i) for i in Y_train_cumul])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
    
        # Return different variables for different phases
        if iteration == start_iter:
            return indices_train_10, X_valid_cumul, X_train_cumul, map_Y_train_cumul, Y_valid_cumul, Y_train_cumul, X_train_cumuls, Y_valid_cumuls, \
                X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, \
                map_Y_valid_cumul, X_valid_ori, Y_valid_ori, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old
        else:
            return indices_train_10, X_valid_cumul, X_train_cumul, map_Y_train_cumul, Y_valid_cumul, Y_train_cumul, X_train_cumuls, Y_valid_cumuls, \
                X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, \
                map_Y_valid_cumul, X_protoset, Y_protoset, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old

    def imprint_weights(self, b1_model, iteration, is_start_iteration, X_train, map_Y_train, dictionary_size):
        """The function to imprint FC classifier's weights 
        Args:
          b1_model: the 1st branch model from last phase 
          iteration: the iteration index 
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
        Returns:
          b1_model: the 1st branch model from the current phase, the FC classifier is updated
        """
        if self.args.dataset == 'cifar100':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = b1_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = b1_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                # Set a temporary dataloader for the current class
                self.evalset.data = X_train[cls_indices].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the feature maps using the current model
                cls_features = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps 
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            b1_model.to(self.device)
            b1_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = b1_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = b1_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])<=dictionary_size)
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=2)
                num_samples = len(X_train[cls_indices])
                # Compute the feature maps using the current model
                cls_features = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            b1_model.to(self.device)
            b1_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        else:
            raise ValueError('Please set correct dataset.')
        return b1_model

    def update_train_and_valid_loader(self, X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, X_valid_cumul, map_Y_valid_cumul, iteration, X_train_cumul, map_Y_train_cumul, start_iter):
        """The function to update the dataloaders
        Args:
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          X_valid_cumuls: an array that contains old validation samples
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
          iteration: the iteration index 
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
        Returns:
          trainloader: the training dataloader
          testloader: the test dataloader
        """
        print('Setting the dataloaders ...')
        X_train_cumul_legnth = len(X_train_cumul)
        X_train_legnth = len(X_train)
        X_train_old = X_train_cumul[0:X_train_cumul_legnth-X_train_legnth]
        map_Y_train_old = map_Y_train_cumul[0:X_train_cumul_legnth-X_train_legnth]
        if self.args.dataset == 'cifar100':
            # Set the training dataloader
            self.trainset.data = X_train.astype('uint8')
            self.trainset.targets = map_Y_train
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers)

            self.largetrainset.data = X_train_large.astype('uint8')
            self.largetrainset.targets = map_Y_train_large
            largetrainloader = torch.utils.data.DataLoader(self.largetrainset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers)

            # Set the test dataloader
            self.testset.data = X_valid_cumul.astype('uint8')
            self.testset.targets = map_Y_valid_cumul
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.smalltestset.data = X_train_small.astype('uint8')
            self.smalltestset.targets = map_Y_train_small
            smalltestloader = torch.utils.data.DataLoader(self.smalltestset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers)

        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Set the training dataloader
            current_train_imgs = merge_images_labels(X_train, map_Y_train)
            self.trainset.imgs = self.trainset.samples = current_train_imgs
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

            current_train_imgs = merge_images_labels(X_train_large, map_Y_train_large)
            self.largetrainset.imgs = self.largetrainset.samples = current_train_imgs
            largetrainloader = torch.utils.data.DataLoader(self.largetrainset, batch_size=self.args.train_batch_size,
                shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

            # Set the test dataloader
            current_test_imgs = merge_images_labels(X_train_small, map_Y_train_small)
            self.smalltestset.imgs = self.smalltestset.samples = current_test_imgs
            smalltestloader = torch.utils.data.DataLoader(self.smalltestset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)

            current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.testset.imgs = self.testset.samples = current_test_imgs
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set correct dataset.')
        return trainloader, largetrainloader, testloader, smalltestloader

    def set_optimizer(self, iteration, start_iter, b1_model, ref_model):
        """The function to set the optimizers for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          b1_model: the 1st branch model from the current phase
        Returns:
          tg_optimizer: the optimizer for b1_model
          tg_lr_scheduler: the learning rate decay scheduler for b1_model
        """
        if iteration > start_iter: 
            # The i-th phase (i>=2)
            
            # Transfer the forzen reference models to GPU
            if ref_model is not None:                  
                ref_model = ref_model.to(self.device)

            # Freeze the FC weights for old classes, get the parameters for the 1st branch
            ignored_params = list(map(id, b1_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, b1_model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)

            if self.args.branch_type == 'fixed':
                # If the 1st branch is fixed, set the learning rate to zero
                branch1_lr = 0.0
                branch1_weight_decay = 0
            else:
                # If the 1st branch is not fixed, using the learning rate in the config
                branch1_lr = self.args.base_lr2
                branch1_weight_decay = self.args.custom_weight_decay            

            # Combine the parameters and the learning rates
            tg_params_new =[{'params': base_params, 'lr': branch1_lr, 'weight_decay': branch1_weight_decay}, {'params': b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

            # Transfer the 1st branch model to the GPU
            b1_model = b1_model.to(self.device)
            
            # Set the optimizer for b1_model
            tg_optimizer = optim.SGD(tg_params_new, lr=self.args.base_lr2, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
         
        else:
            # The 0th phase
            # For the 0th phase, we train conventional CNNs, so we don't need to update the aggregation weights
            tg_params = b1_model.parameters()
            b1_model = b1_model.to(self.device)
            tg_optimizer = optim.SGD(tg_params, lr=self.args.base_lr1, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)

        # Set the learning rate decay scheduler 
        if self.args.imgnet_split == 'podnet':
            tg_lr_scheduler = lr_scheduler.CosineAnnealingLR(tg_optimizer, self.args.epochs)
        elif self.args.imgnet_split == 'icarl':
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat, gamma=self.args.lr_factor)   
        else:
            raise ValueError('Please set correct split.')

        return tg_optimizer, tg_lr_scheduler

    def compute_acc_temp(self, the_intensity_gamma, temp_class_means, order, order_list, b1_model, X_protoset_cumuls, Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration):

        current_means = temp_class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]
        pin_memory = False
        # Get mapped labels for the current-phase data, according the the order list
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        # Set a temporary dataloader for the current-phase data
        print('Computing cumulative accuracy...')
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_cumul.astype('uint8')
            self.evalset.targets = map_Y_valid_cumul
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':  
            current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.evalset.imgs = self.evalset.samples = current_eval_set
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)    
        # Compute the accuracies for the current-phase data    
        cumul_acc = compute_accuracy_temp(self.args, b1_model, the_intensity_gamma, current_means, \
            X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, \
            is_start_iteration=is_start_iteration)         

        return cumul_acc


    def compute_acc(self, intensity_gamma, class_means, order, order_list, b1_model, X_train_small, map_Y_train_small, X_protoset_cumuls, Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old, before_cbf=True):
        """The function to compute the accuracy
        Args:
          class_means: the mean values for each class
          order: the array for the class order
          order_list: the list for the class order
          b1_model: the 1st branch model from the current phase
          X_protoset_cumuls: the array that contains old exemplar samples
          Y_protoset_cumuls: the array that contains old exemplar labels
          X_valid_ori: the array that contains the 0th-phase validation samples, updated
          Y_valid_ori: the array that contains the 0th-phase validation labels, updated
          X_valid_cumuls: the array that contains old validation samples
          Y_valid_cumuls: the array that contains old validation labels 
          iteration: the iteration index
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          top1_acc_list_ori: the list to store the results for the 0th classes
          top1_acc_list_cumul: the list to store the results for the current phase
        Returns:
          top1_acc_list_ori: the list to store the results for the 0th classes, updated
          top1_acc_list_cumul: the list to store the results for the current phase, updated
        """

        # Get the class mean values for all seen classes
        current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]

        # Get mapped labels for the 0-th phase data, according the the order list
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        map_Y_valid_new = np.array([order_list.index(i) for i in Y_valid_new])
        map_Y_valid_old = np.array([order_list.index(i) for i in Y_valid_old])
        print('Computing accuracy on the 0-th phase classes...')
        # Set a temporary dataloader for the 0-th phase data
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_ori.astype('uint8')
            self.evalset.targets = map_Y_valid_ori
            pin_memory = False
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
            current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
            self.evalset.imgs = self.evalset.samples = current_eval_set
            pin_memory = True
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)
        # Compute the accuracies for the 0-th phase test data
        ori_acc, fast_fc = compute_accuracy(self.args, intensity_gamma, b1_model, \
            current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, \
            order_list, is_start_iteration=is_start_iteration)
        # Add the results to the array, which stores all previous results
        top1_acc_list_ori[iteration, :, 0] = np.array(ori_acc).T
        # Write the results to tensorboard
        if before_cbf:
            self.train_writer.add_scalar('ori_acc/fc', float(ori_acc[0]), iteration)
            self.train_writer.add_scalar('ori_acc/proto', float(ori_acc[1]), iteration)
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('ori_acc/mixed', float(ori_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('ori_acc/mixed', float(ori_acc[1]), iteration)
            #wandb.log({"ori_acc_fc": float(ori_acc[0])})
            #andb.log({"ori_acc_proto": float(ori_acc[1])})
        else:
            self.train_writer.add_scalar('ori_acc_after_cbf/fc', float(ori_acc[0]), iteration)
            self.train_writer.add_scalar('ori_acc_after_cbf/proto', float(ori_acc[1]), iteration)
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('ori_acc_after_cbf/mixed', float(ori_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('ori_acc_after_cbf/mixed', float(ori_acc[1]), iteration)
            #wandb.log({"ori_acc_after_cbf_fc": float(ori_acc[0])})
            #wandb.log({"ori_acc_after_cbf_proto": float(ori_acc[1])})

        print('Computing accuracy on the new classes...')
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_new.astype('uint8')
            self.evalset.targets = map_Y_valid_new
            pin_memory = False
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
            current_eval_set = merge_images_labels(X_valid_new, map_Y_valid_new)
            self.evalset.imgs = self.evalset.samples = current_eval_set
            pin_memory = True
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)
        # Compute the accuracies for the 0-th phase test data
        new_acc, fast_fc = compute_accuracy(self.args, intensity_gamma, b1_model, \
            current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, \
            order_list, is_start_iteration=is_start_iteration)

        # Write the results to tensorboard
        if before_cbf:
            self.train_writer.add_scalar('new_acc/fc', float(new_acc[0]), iteration)
            self.train_writer.add_scalar('new_acc/proto', float(new_acc[1]), iteration)
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('new_acc/mixed', float(new_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('new_acc/mixed', float(new_acc[1]), iteration)
            #wandb.log({"new_acc_fc": float(new_acc[0])})
            #wandb.log({"new_acc_proto": float(new_acc[1])})
        else:
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('new_acc_after_cbf/mixed', float(new_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('new_acc_after_cbf/mixed', float(new_acc[1]), iteration)
            self.train_writer.add_scalar('new_acc_after_cbf/fc', float(new_acc[0]), iteration)
            self.train_writer.add_scalar('new_acc_after_cbf/proto', float(new_acc[1]), iteration)
            #wandb.log({"new_acc_after_cbf_fc": float(new_acc[0])})
            #wandb.log({"new_acc_after_cbf_proto": float(new_acc[1])})

        if not is_start_iteration:
            print('Computing accuracy on the old classes...')
            if self.args.dataset == 'cifar100':
                self.evalset.data = X_valid_old.astype('uint8')
                self.evalset.targets = map_Y_valid_old
                pin_memory = False
            elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
                current_eval_set = merge_images_labels(X_valid_old, map_Y_valid_old)
                self.evalset.imgs = self.evalset.samples = current_eval_set
                pin_memory = True
            else:
                raise ValueError('Please set the correct dataset.')
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)
            # Compute the accuracies for the 0-th phase test data
            old_acc, fast_fc = compute_accuracy(self.args, intensity_gamma, b1_model, \
                current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, \
                order_list, is_start_iteration=is_start_iteration)

            # Write the results to tensorboard
            if before_cbf:
                if intensity_gamma==1.0:
                    self.train_writer.add_scalar('old_acc/mixed', float(old_acc[0]), iteration)
                else:
                    self.train_writer.add_scalar('old_acc/mixed', float(old_acc[1]), iteration)
                self.train_writer.add_scalar('old_acc/fc', float(old_acc[0]), iteration)
                self.train_writer.add_scalar('old_acc/proto', float(old_acc[1]), iteration)
                #wandb.log({"old_acc_fc": float(old_acc[0])})
                #wandb.log({"old_acc_proto": float(old_acc[1])})
            else:
                if intensity_gamma==1.0:
                    self.train_writer.add_scalar('old_acc_after_cbf/mixed', float(old_acc[0]), iteration)
                else:
                    self.train_writer.add_scalar('old_acc_after_cbf/mixed', float(old_acc[1]), iteration)
                self.train_writer.add_scalar('old_acc_after_cbf/fc', float(old_acc[0]), iteration)
                self.train_writer.add_scalar('old_acc_after_cbf/proto', float(old_acc[1]), iteration)
                #wandb.log({"old_acc_after_cbf_fc": float(old_acc[0])})
                #wandb.log({"old_acc_after_cbf_proto": float(old_acc[1])})   
                
            print('Computing accuracy on the balanced set...')
            if self.args.dataset == 'cifar100':
                self.evalset.data = X_train_small.astype('uint8')
                self.evalset.targets = map_Y_train_small
                pin_memory = False
            elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
                current_eval_set = merge_images_labels(X_train_small, map_Y_train_small)
                self.evalset.imgs = self.evalset.samples = current_eval_set
                pin_memory = True
            else:
                raise ValueError('Please set the correct dataset.')
            evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)
            # Compute the accuracies for the 0-th phase test data
            balanced_acc, fast_fc = compute_accuracy(self.args, intensity_gamma, b1_model, \
                current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, \
                order_list, is_start_iteration=is_start_iteration)
            self.train_writer.add_scalar('balanced_acc/fc', float(balanced_acc[0]), iteration)
            self.train_writer.add_scalar('balanced_acc/proto', float(balanced_acc[1]), iteration)

        else:
            if before_cbf:
                if intensity_gamma==1.0:
                    self.train_writer.add_scalar('old_acc/mixed', float(ori_acc[0]), iteration)
                else:
                    self.train_writer.add_scalar('old_acc/mixed', float(ori_acc[1]), iteration)
                self.train_writer.add_scalar('old_acc/fc', float(ori_acc[0]), iteration)
                self.train_writer.add_scalar('old_acc/proto', float(ori_acc[1]), iteration)
                #wandb.log({"old_acc_fc": float(ori_acc[0])})
                #wandb.log({"old_acc_proto": float(ori_acc[1])}) 
            else:
                if intensity_gamma==1.0:
                    self.train_writer.add_scalar('old_acc_after_cbf/mixed', float(ori_acc[0]), iteration)
                else:
                    self.train_writer.add_scalar('old_acc_after_cbf/mixed', float(ori_acc[1]), iteration)
                self.train_writer.add_scalar('old_acc_after_cbf/fc', float(ori_acc[0]), iteration)
                self.train_writer.add_scalar('old_acc_after_cbf/proto', float(ori_acc[1]), iteration)
                #wandb.log({"old_acc_after_cbf_fc": float(ori_acc[0])})
                #wandb.log({"old_acc_after_cbf_proto": float(ori_acc[1])})                            

        # Get mapped labels for the current-phase data, according the the order list
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        # Set a temporary dataloader for the current-phase data
        print('Computing cumulative accuracy...')
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_cumul.astype('uint8')
            self.evalset.targets = map_Y_valid_cumul
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':  
            current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.evalset.imgs = self.evalset.samples = current_eval_set
        else:
            raise ValueError('Please set the correct dataset.')
        evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=pin_memory)    
        # Compute the accuracies for the current-phase data    
        cumul_acc, _ = compute_accuracy(self.args, intensity_gamma, b1_model, \
            current_means, X_protoset_cumuls, Y_protoset_cumuls, evalloader, order_list, \
            is_start_iteration=is_start_iteration, fast_fc=fast_fc)
        # Add the results to the array, which stores all previous results
        top1_acc_list_cumul[iteration, :, 0] = np.array(cumul_acc).T
        # Write the results to tensorboard
        if before_cbf:
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('cumul_acc/mixed', float(cumul_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('cumul_acc/mixed', float(cumul_acc[1]), iteration)
            self.train_writer.add_scalar('cumul_acc/fc', float(cumul_acc[0]), iteration)
            self.train_writer.add_scalar('cumul_acc/proto', float(cumul_acc[1]), iteration)
            #wandb.log({"cumul_acc_fc": float(cumul_acc[0])})
            #wandb.log({"cumul_acc_proto": float(cumul_acc[1])})
        else:
            if intensity_gamma==1.0:
                self.train_writer.add_scalar('cumul_acc_after_cbf/mixed', float(cumul_acc[0]), iteration)
            else:
                self.train_writer.add_scalar('cumul_acc_after_cbf/mixed', float(cumul_acc[1]), iteration)
            self.train_writer.add_scalar('cumul_acc_after_cbf/fc', float(cumul_acc[0]), iteration)
            self.train_writer.add_scalar('cumul_acc_after_cbf/proto', float(cumul_acc[1]), iteration)
            #wandb.log({"cumul_acc_after_cbf_fc": float(cumul_acc[0])})
            #wandb.log({"cumul_acc_after_cbf_proto": float(cumul_acc[1])})            

        return top1_acc_list_ori, top1_acc_list_cumul

    def set_exemplar_set(self, b1_model, is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes):
        """The function to select the exemplars
        Args:
          b1_model: the 1st branch model from the current phase
          is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
          iteration: the iteration index
          last_iter: the iteration index for last phase
          order: the array for the class order
          alpha_dr_herding: the empty array to store the indexes for the exemplars
          prototypes: the array contains all training samples for all phases
        Returns:
          X_protoset_cumuls: an array that contains old exemplar samples
          Y_protoset_cumuls: an array that contains old exemplar labels
          class_means: the mean values for each class
          alpha_dr_herding: the empty array to store the indexes for the exemplars, updated
        """
        # Use the dictionary size defined in this class-incremental learning class
        dictionary_size = self.dictionary_size
        if self.args.dynamic_budget:
            # Using dynamic exemplar budget, i.e., 20 exemplars each class. In this setting, the total memory budget is increasing
            nb_protos_cl = self.args.nb_protos
        else:
            # Using fixed exemplar budget. The total memory size is unchanged
            nb_protos_cl = int(np.ceil(self.args.nb_protos*100./self.args.nb_cl/(iteration+1)))
        # Get the shape for the feature maps
        num_features = b1_model.fc.in_features
        if self.args.dataset == 'cifar100':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                self.evalset.data = prototypes[iter_dico].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the features for the current class          
                mapped_prototypes = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                    shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                num_samples = len(prototypes[iter_dico])            
                # Compute the features for the current class  
                mapped_prototypes = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)

                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
        else:
            raise ValueError('Please set correct dataset.')
        # Set two empty lists for the exemplars and the labels 
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        if self.args.dataset == 'cifar100':
            the_dim = b1_model.fc.in_features
            class_means = np.zeros((the_dim,100,2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    num_samples = self.evalset.data.shape[0]
                    mapped_prototypes = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    # Using the indexes selected by herding
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico,np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    # Compute the class mean values                  
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':            
            class_means = np.zeros((num_features, self.args.num_classes, 2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    current_eval_set = merge_images_labels(prototypes[iteration2*self.args.nb_cl+iter_dico], \
                        np.zeros(len(prototypes[iteration2*self.args.nb_cl+iter_dico])))
                    self.evalset.imgs = self.evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
                        shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
                    num_samples = len(prototypes[iteration2*self.args.nb_cl+iter_dico])
                    mapped_prototypes = compute_features(self.args, b1_model, is_start_iteration, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    D2 = D
                    # Using the indexes selected by herding
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    assert((alph[num_samples:]==0).all())
                    alph = alph[:num_samples]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico][np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    # Compute the class mean values   
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(num_samples)/num_samples
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        else:
            raise ValueError('Please set correct dataset.')

        # Save the class mean values   
        torch.save(class_means, osp.join(self.save_path, 'iter_{}_class_means.pth'.format(iteration)))
        return X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding

    def preparing_cbf(self, X_protoset_cumuls, Y_protoset_cumuls, order_list, tg_model):
        # Class balance finetuning on the protoset
        print("###############################")
        print("Class balance finetuning on the protoset")
        print("###############################")
        map_Y_protoset_cumuls = np.array([order_list.index(i) for i in np.concatenate(Y_protoset_cumuls)])
        if self.args.dataset == 'cifar100':
            self.trainset.data = np.concatenate(X_protoset_cumuls).astype('uint8')
            self.trainset.targets = map_Y_protoset_cumuls
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet': 
            current_train_imgs = merge_images_labels(np.concatenate(X_protoset_cumuls), map_Y_protoset_cumuls)
            self.trainset.imgs = self.trainset.samples = current_train_imgs
        else:
            raise ValueError('Please set correct dataset.')
        cbf_trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        print('Min and Max of train labels: {}, {}'.format(min(map_Y_protoset_cumuls), max(map_Y_protoset_cumuls)))

        if self.args.ft_flag == 2: #both the old and novel embeddings are updated with the feature extractor fixed
            ignored_params = list(map(id, tg_model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, tg_model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)
            tg_params =[{'params': base_params, 'lr': 0, 'weight_decay': 0}, \
                            {'params': tg_model.fc.fc1.parameters(), 'lr': self.args.ft_base_lr, 'weight_decay': self.args.custom_weight_decay}, \
                            {'params': tg_model.fc.fc2.parameters(), 'lr': self.args.ft_base_lr, 'weight_decay': self.args.custom_weight_decay}]
        elif self.args.args.ft_flag == 3: #everything is updated
            ignored_params = list(map(id, tg_model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, tg_model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)
            tg_params =[{'params': base_params, 'lr': self.args.ft_base_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': tg_model.fc.fc1.parameters(), 'lr': self.args.ft_base_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': tg_model.fc.fc2.parameters(), 'lr': self.args.ft_base_lr, 'weight_decay': self.args.custom_weight_decay}]

        fix_bn_flag = True
        tg_ft_optimizer = optim.SGD(tg_params, lr=self.args.ft_base_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
        tg_ft_lr_scheduler = lr_scheduler.MultiStepLR(tg_ft_optimizer, milestones=self.args.ft_lr_strat, gamma=self.args.lr_factor)

        return cbf_trainloader, tg_ft_optimizer, tg_ft_lr_scheduler


    def hyperparameter_search_alpha_beta(self, the_intensity_gamma, alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul):
        if self.args.debug_mode:
            intensity_alpha_list = [0.1, 1]
            intensity_beta_list = [0.1, 1]
        else:
            intensity_alpha_list = [0, 0.1, 0.5, 1.0, 10.0]
            intensity_beta_list = [0, 0.1, 0.5, 1.0, 10.0]
        history_array = np.zeros((len(intensity_alpha_list), len(intensity_beta_list)))

        if self.args.debug_mode:
            search_epochs = 1
        else:
            search_epochs = 20
        search_lr = 0.1
        search_lr_strat = [int(search_epochs*0.5), int(search_epochs*0.75)]

        the_number_of_actions = len(intensity_alpha_list)*len(intensity_beta_list)
        exp3_weights = [1.0] * the_number_of_actions
        eta = 0.1

        if self.args.kd_weight_iteration>the_number_of_actions:
            for the_iteration_idx in range(self.args.kd_weight_iteration):
                probabilityDistribution = self.distr(exp3_weights, eta)
                choice = self.draw(probabilityDistribution)

                the_intensity_alpha_idx = choice // len(intensity_beta_list)
                the_intensity_beta_idx = choice % len(intensity_beta_list)
                the_intensity_alpha = intensity_alpha_list[the_intensity_alpha_idx]
                the_intensity_beta = intensity_beta_list[the_intensity_beta_idx]
                print('Currently searching: alpha=' + str(the_intensity_alpha) + ' beta=' + str(the_intensity_beta))

                temp_b1_model = copy.deepcopy(b1_model)
                temp_b1_model.to(self.device)

                ignored_params = list(map(id, temp_b1_model.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, temp_b1_model.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                search_params =[{'params': base_params, 'lr': search_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': temp_b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

                search_optimizer = optim.SGD(search_params, lr=search_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                search_scheduler = lr_scheduler.MultiStepLR(search_optimizer, milestones=search_lr_strat, gamma=0.1)

                temp_b1_model = incremental_train_and_eval_mixed(self.args, search_epochs, temp_b1_model, \
                    ref_model, search_optimizer, search_scheduler, trainloader, \
                    testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, \
                    order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, \
                    intensity_alpha=the_intensity_alpha, intensity_beta=the_intensity_beta, \
                    intensity_gamma=the_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)

                temp_alpha_dr_herding = np.copy(alpha_dr_herding)

                temp_X_protoset_cumuls, temp_Y_protoset_cumuls, temp_class_means, temp_alpha_dr_herding = self.set_exemplar_set(temp_b1_model, is_start_iteration, iteration, last_iter, order, temp_alpha_dr_herding, prototypes)
                the_acc = self.compute_acc_temp(the_intensity_gamma, temp_class_means, order, order_list, temp_b1_model, temp_X_protoset_cumuls, temp_Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration)

                exp3_weights[choice] *= math.exp(the_acc * eta / the_number_of_actions)

            final_choice = self.draw(probabilityDistribution)
            final_intensity_alpha_idx = final_choice // len(intensity_beta_list)
            final_intensity_beta_idx = final_choice % len(intensity_beta_list)

            final_intensity_alpha = intensity_alpha_list[final_intensity_alpha_idx]
            final_intensity_beta = intensity_beta_list[final_intensity_beta_idx]
        else:
            for the_intensity_alpha_idx in range(len(intensity_alpha_list)):
                for the_intensity_beta_idx in range(len(intensity_beta_list)):
                    the_intensity_alpha = intensity_alpha_list[the_intensity_alpha_idx]
                    the_intensity_beta = intensity_beta_list[the_intensity_beta_idx]
                    print('Currently searching: alpha=' + str(the_intensity_alpha) + ' beta=' + str(the_intensity_beta))

                    temp_b1_model = copy.deepcopy(b1_model)
                    temp_b1_model.to(self.device)

                    ignored_params = list(map(id, temp_b1_model.fc.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, temp_b1_model.parameters())
                    base_params = filter(lambda p: p.requires_grad, base_params)
                    search_params =[{'params': base_params, 'lr': search_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': temp_b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

                    search_optimizer = optim.SGD(search_params, lr=search_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                    search_scheduler = lr_scheduler.MultiStepLR(search_optimizer, milestones=search_lr_strat, gamma=0.1)

                    temp_b1_model = incremental_train_and_eval_mixed(self.args, search_epochs, temp_b1_model, \
                        ref_model, search_optimizer, search_scheduler, trainloader, \
                        testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, \
                        order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, \
                        intensity_alpha=the_intensity_alpha, intensity_beta=the_intensity_beta, \
                        intensity_gamma=the_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)

                    temp_alpha_dr_herding = np.copy(alpha_dr_herding)

                    temp_X_protoset_cumuls, temp_Y_protoset_cumuls, temp_class_means, temp_alpha_dr_herding = self.set_exemplar_set(temp_b1_model, is_start_iteration, iteration, last_iter, order, temp_alpha_dr_herding, prototypes)
                    the_acc = self.compute_acc_temp(the_intensity_gamma, temp_class_means, order, order_list, temp_b1_model, temp_X_protoset_cumuls, temp_Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration)

                    history_array[the_intensity_alpha_idx][the_intensity_beta_idx]=the_acc

            final_intensity_alpha_idx, final_intensity_beta_idx = np.unravel_index(history_array.argmax(), history_array.shape)

            final_intensity_alpha = intensity_alpha_list[final_intensity_alpha_idx]
            final_intensity_beta = intensity_beta_list[final_intensity_beta_idx]

        self.train_writer.add_scalar('hyperparameter/alpha', float(final_intensity_alpha), iteration)
        self.train_writer.add_scalar('hyperparameter/beta', float(final_intensity_beta), iteration)

        return final_intensity_alpha, final_intensity_beta

    def hyperparameter_search_gamma(self, the_intensity_alpha, the_intensity_beta, alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul):
        intensity_gamma_list = [0, 1]
        history_array = np.zeros((len(intensity_gamma_list)))

        if self.args.debug_mode:
            search_epochs = 1
        else:
            search_epochs = 20
        search_lr = 0.1
        search_lr_strat = [int(search_epochs*0.5), int(search_epochs*0.75)]

        for the_intensity_gamma_idx in range(len(intensity_gamma_list)):
            the_intensity_gamma = intensity_gamma_list[the_intensity_gamma_idx]
            if the_intensity_gamma==1:
                the_intensity_alpha=1
                the_intensity_beta=0
            else:
                the_intensity_alpha=0
                the_intensity_beta=1
            print('Currently searching: gamma=' + str(the_intensity_gamma))

            temp_b1_model = copy.deepcopy(b1_model)
            temp_b1_model.to(self.device)

            ignored_params = list(map(id, temp_b1_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, temp_b1_model.parameters())
            base_params = filter(lambda p: p.requires_grad, base_params)
            search_params =[{'params': base_params, 'lr': search_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': temp_b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

            search_optimizer = optim.SGD(search_params, lr=search_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            search_scheduler = lr_scheduler.MultiStepLR(search_optimizer, milestones=search_lr_strat, gamma=0.1)

            temp_b1_model = incremental_train_and_eval_mixed(self.args, search_epochs, temp_b1_model, \
                ref_model, search_optimizer, search_scheduler, trainloader, \
                testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, \
                order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, \
                intensity_alpha=the_intensity_alpha, intensity_beta=the_intensity_beta, \
                intensity_gamma=the_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)

            temp_alpha_dr_herding = np.copy(alpha_dr_herding)

            temp_X_protoset_cumuls, temp_Y_protoset_cumuls, temp_class_means, temp_alpha_dr_herding = self.set_exemplar_set(temp_b1_model, is_start_iteration, iteration, last_iter, order, temp_alpha_dr_herding, prototypes)
            the_acc = self.compute_acc_temp(the_intensity_gamma, temp_class_means, order, order_list, temp_b1_model, temp_X_protoset_cumuls, temp_Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration)

            history_array[the_intensity_gamma_idx]=the_acc

        final_intensity_gamma_idx = np.unravel_index(history_array.argmax(), history_array.shape)

        #import pdb
        #pdb.set_trace()
        final_intensity_gamma = intensity_gamma_list[final_intensity_gamma_idx[0]]

        self.train_writer.add_scalar('hyperparameter/gamma', float(final_intensity_gamma), iteration)

        return final_intensity_gamma

    def hyperparameter_search_lr(self, the_intensity_alpha, the_intensity_beta, the_intensity_gamma, alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul):
        lr_list = [0.05, 0.1]
        history_array = np.zeros((len(lr_list)))

        if self.args.debug_mode:
            search_epochs = 1
        else:
            search_epochs = 160
        #search_lr = 0.1
        search_lr_strat = [int(search_epochs*0.5), int(search_epochs*0.75)]

        the_number_of_actions = len(lr_list)
        exp3_weights = [1.0] * the_number_of_actions
        eta = 0.1

        if self.args.lr_iteration>the_number_of_actions:
            for the_iteration_idx in range(self.args.kd_weight_iteration):
                probabilityDistribution = self.distr(exp3_weights, eta)
                choice = self.draw(probabilityDistribution)
                the_lr = lr_list[choice]
                print('Search the lr: ' + str(the_lr))
                
                temp_b1_model = copy.deepcopy(b1_model)
                temp_b1_model.to(self.device)

                ignored_params = list(map(id, temp_b1_model.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, temp_b1_model.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                search_params =[{'params': base_params, 'lr': the_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': temp_b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

                search_optimizer = optim.SGD(search_params, lr=the_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                search_scheduler = lr_scheduler.MultiStepLR(search_optimizer, milestones=search_lr_strat, gamma=0.1)

                temp_b1_model = incremental_train_and_eval_mixed(self.args, search_epochs, temp_b1_model, \
                    ref_model, search_optimizer, search_scheduler, trainloader, \
                    testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, \
                    order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, \
                    intensity_alpha=the_intensity_alpha, intensity_beta=the_intensity_beta, \
                    intensity_gamma=the_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)

                temp_alpha_dr_herding = np.copy(alpha_dr_herding)

                temp_X_protoset_cumuls, temp_Y_protoset_cumuls, temp_class_means, temp_alpha_dr_herding = self.set_exemplar_set(temp_b1_model, is_start_iteration, iteration, last_iter, order, temp_alpha_dr_herding, prototypes)
                the_acc = self.compute_acc_temp(the_intensity_gamma, temp_class_means, order, order_list, temp_b1_model, temp_X_protoset_cumuls, temp_Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration)

                exp3_weights[choice] *= math.exp(the_acc * eta / the_number_of_actions)


            final_choice = self.draw(probabilityDistribution)
            final_lr = lr_list[final_choice]
        else:
            for the_lr_idx in range(len(lr_list)):
                the_lr = lr_list[the_lr_idx]
                print('Search the lr: ' + str(the_lr))
                
                temp_b1_model = copy.deepcopy(b1_model)
                temp_b1_model.to(self.device)

                ignored_params = list(map(id, temp_b1_model.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, temp_b1_model.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                search_params =[{'params': base_params, 'lr': the_lr, 'weight_decay': self.args.custom_weight_decay}, {'params': temp_b1_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]

                search_optimizer = optim.SGD(search_params, lr=the_lr, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                search_scheduler = lr_scheduler.MultiStepLR(search_optimizer, milestones=search_lr_strat, gamma=0.1)

                temp_b1_model = incremental_train_and_eval_mixed(self.args, search_epochs, temp_b1_model, \
                    ref_model, search_optimizer, search_scheduler, trainloader, \
                    testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, \
                    order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, \
                    intensity_alpha=the_intensity_alpha, intensity_beta=the_intensity_beta, \
                    intensity_gamma=the_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)

                temp_alpha_dr_herding = np.copy(alpha_dr_herding)

                temp_X_protoset_cumuls, temp_Y_protoset_cumuls, temp_class_means, temp_alpha_dr_herding = self.set_exemplar_set(temp_b1_model, is_start_iteration, iteration, last_iter, order, temp_alpha_dr_herding, prototypes)
                the_acc = self.compute_acc_temp(the_intensity_gamma, temp_class_means, order, order_list, temp_b1_model, temp_X_protoset_cumuls, temp_Y_protoset_cumuls, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration)

                history_array[the_lr_idx]=the_acc

            final_lr_idx = np.unravel_index(history_array.argmax(), history_array.shape)

            #import pdb
            #pdb.set_trace()
            final_lr = lr_list[final_lr_idx[0]]

        self.train_writer.add_scalar('hyperparameter/lr', float(final_lr), iteration)

        return final_lr

    def distr(self, weights, eta=0.0):
        theSum = float(sum(weights))
        return tuple((1.0 - eta) * (w / theSum) + (eta / len(weights)) for w in weights)

    def draw(self, weights):
        choice = random.uniform(0, sum(weights))
        choiceIndex = 0

        for weight in weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1