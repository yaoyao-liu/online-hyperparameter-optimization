""" Class-incremental learning trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
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
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_accuracy import compute_accuracy
from trainer.lucir import incremental_train_and_eval as incremental_train_and_eval_lucir
from trainer.icarl import incremental_train_and_eval as incremental_train_and_eval_icarl
from trainer.mixed import incremental_train_and_eval as incremental_train_and_eval_mixed
from trainer.zeroth_phase import incremental_train_and_eval_zeroth_phase as incremental_train_and_eval_zeroth_phase
from utils.misc import process_mnemonics
from trainer.base import BaseTrainer
import warnings
warnings.filterwarnings('ignore')

class Trainer(BaseTrainer):
    def train(self):
        """The class that contains the code for the class-incremental system."""
        
        # Set tensorboard recorder
        #self.train_writer = SummaryWriter(comment=self.save_path)

        # Initial the array to store the accuracies for each phase
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))

        if self.args.cb_finetune:
            after_cbf_top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
            after_cbf_top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))            

        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()      

        # Initialize the class order
        order, order_list = self.init_class_order()
        np.random.seed(None)

        # Set empty lists for the data    
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []
        the_distillation_set_index_list = []

        # Initialize the prototypes
        alpha_dr_herding, prototypes = self.init_prototypes(self.dictionary_size, order, X_train_total, Y_train_total)

        # Set the starting iteration
        # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        # Set the models and some parameter to None
        # These models and parameters will be assigned in the following phases
        b1_model = None
        ref_model = None
        the_lambda_mult = None


        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            ### Initialize models for the current phase
            b1_model, ref_model, lambda_mult, cur_lambda, last_iter = self.init_current_phase_model(iteration, start_iter, b1_model)

            ### Initialize datasets for the current phase
            if iteration == start_iter:
                indices_train_10, X_valid_cumul, X_train_cumul, map_Y_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, map_Y_valid_cumul, X_valid_ori, Y_valid_ori, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)
            else:
                indices_train_10, X_valid_cumul, X_train_cumul, map_Y_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, map_Y_valid_cumul, X_protoset, Y_protoset, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)                

            is_start_iteration = (iteration == start_iter)

            # Imprint weights
            if iteration > start_iter:
                b1_model = self.imprint_weights(b1_model, iteration, is_start_iteration, X_train, map_Y_train, self.dictionary_size)

            # Update training and test dataloader
            trainloader, largetrainset, testloader, smalltestloader = self.update_train_and_valid_loader(X_train, map_Y_train, X_train_small, map_Y_train_small, X_train_large, map_Y_train_large, X_valid_cumul, map_Y_valid_cumul, iteration, X_train_cumul, map_Y_train_cumul, start_iter)

            # Set the names for the checkpoints
            ckp_name = osp.join(self.save_path, 'iter_{}_b1.pth'.format(iteration))          
            print('Check point name: ', ckp_name)

            if iteration==start_iter and self.args.resume_fg:
                # Resume the 0-th phase model according to the config
                b1_model = torch.load(self.args.ckpt_dir_fg)
            elif self.args.resume and os.path.exists(ckp_name):
                # Resume other models according to the config
                b1_model = torch.load(ckp_name)
            else:
                # Start training (if we don't resume the models from the checkppoints)
    
                # Set the optimizer
                tg_optimizer, tg_lr_scheduler = self.set_optimizer(iteration, start_iter, b1_model, ref_model)     

                if iteration > start_iter:
                    if self.args.baseline == 'mixed':
                        if iteration==start_iter+1:
                            final_intensity_gamma = self.hyperparameter_search_gamma(0.5, 0.5, \
                                alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, largetrainset, smalltestloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul)
                        print('*** [Phase' + str(iteration) + '] ' + 'Final gamma=' + str(final_intensity_gamma))
           
                        if iteration == start_iter+1 or iteration%self.args.num_phase_search==0:
                            final_intensity_alpha, final_intensity_beta = self.hyperparameter_search_alpha_beta(final_intensity_gamma, \
                                alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, largetrainset, smalltestloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul) 
                        else:
                            print('No updating, current alpha='+str(final_intensity_alpha)+', current beta='+\
                                str(final_intensity_beta)+', current gamma='+str(final_intensity_gamma))    

                        print('*** [Phase' + str(iteration) + '] ' + 'Final alpha=' + str(final_intensity_alpha))
                        print('*** [Phase' + str(iteration) + '] ' + 'Final beta=' + str(final_intensity_beta))
           
                        if self.args.update_lr:
                            final_lr = self.hyperparameter_search_lr(final_intensity_alpha, final_intensity_beta, final_intensity_gamma, \
                                alpha_dr_herding, is_start_iteration, last_iter, b1_model, ref_model, largetrainset, smalltestloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order, order_list, cur_lambda, prototypes, X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul)

                            print('*** [Phase' + str(iteration) + '] ' + 'Final LR this phase: ' + str(final_lr))
                            self.args.base_lr1 = final_lr
                            self.args.base_lr2 = final_lr   

                            tg_optimizer, tg_lr_scheduler = self.set_optimizer(iteration, start_iter, \
                                b1_model, ref_model)     
                         
                    else:
                        final_intensity_gamma = 1.0

                    # Training the model for different baselines        
                    if self.args.baseline == 'lucir':
                        b1_model = incremental_train_and_eval_lucir(self.args, self.args.epochs, b1_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr)
                    elif self.args.baseline == 'mixed':
                        b1_model = incremental_train_and_eval_mixed(self.args, self.args.epochs, b1_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, intensity_alpha=final_intensity_alpha, intensity_beta=final_intensity_beta, intensity_gamma=final_intensity_gamma, T=self.args.icarl_T, beta=self.args.icarl_beta)
                    elif self.args.baseline == 'icarl':
                        b1_model = incremental_train_and_eval_icarl(self.args, self.args.epochs,  b1_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, self.args.icarl_T, self.args.icarl_beta)
                    else:
                        raise ValueError('Please set the correct baseline.') 
                    torch.save(b1_model, ckp_name)
                    #torch.save(b2_model, ckp_name_b2)       
                else:         
                    # Training the class-incremental learning system from the 0th phase           
                    b1_model = incremental_train_and_eval_zeroth_phase(self.args, self.args.epochs, b1_model, \
                        ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                        cur_lambda, self.args.dist, self.args.K, self.args.lw_mr) 
                    torch.save(b1_model, ckp_name)

                    final_intensity_gamma = 1.0

                os.system('nvidia-smi')

            # Select the exemplars according to the current model
            X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding = self.set_exemplar_set(b1_model, is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes)
         
            # Compute the accuracies for current phase
            top1_acc_list_ori, top1_acc_list_cumul = self.compute_acc(final_intensity_gamma, class_means, order, order_list, b1_model, X_train_small, map_Y_train_small, X_protoset_cumuls, Y_protoset_cumuls, \
                X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old)

            # Class balance finetuning
            if iteration == start_iter and self.args.cb_finetune:
                after_cbf_top1_acc_list_ori[start_iter] = top1_acc_list_ori[start_iter]
                after_cbf_top1_acc_list_cumul[start_iter] = top1_acc_list_cumul[start_iter]
            if iteration > start_iter and self.args.cb_finetune:
                cbf_trainloader, cbf_ft_optimizer, cbf_ft_lr_scheduler = self.preparing_cbf(X_protoset_cumuls, Y_protoset_cumuls, order_list, b1_model)
        
                if self.args.baseline == 'lucir':
                    b1_model = incremental_train_and_eval_lucir(self.args, self.args.ft_epochs, b1_model, ref_model, cbf_ft_optimizer, cbf_ft_lr_scheduler, cbf_trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, fix_bn=True)
                elif self.args.baseline == 'mixed':
                    b1_model = incremental_train_and_eval_mixed(self.args, self.args.ft_epochs, b1_model, ref_model, cbf_ft_optimizer, cbf_ft_lr_scheduler, cbf_trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, intensity_alpha=intensity_alpha_list[final_intensity_alpha_idx], intensity_beta=intensity_alpha_list[final_intensity_beta_idx], intensity_gamma=intensity_gamma_list[final_intensity_gamma_idx], T=self.args.icarl_T, beta=self.args.icarl_beta, fix_bn=True)
                elif self.args.baseline == 'icarl':
                    b1_model = incremental_train_and_eval_icarl(self.args, self.args.ft_epochs,  b1_model, ref_model, cbf_ft_optimizer, cbf_ft_lr_scheduler, cbf_trainloader, testloader, iteration, start_iter, X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, self.args.icarl_T, self.args.icarl_beta, fix_bn=True)
                else:
                    raise ValueError('Please set the correct baseline.')
                after_cbf_top1_acc_list_ori, after_cbf_top1_acc_list_cumul = self.compute_acc(final_intensity_gamma, class_means, order, order_list, b1_model, X_protoset_cumuls, Y_protoset_cumuls, \
                    X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, after_cbf_top1_acc_list_ori, after_cbf_top1_acc_list_cumul, X_valid_new, Y_valid_new, X_valid_old, Y_valid_old, before_cbf=False)

            # Compute the average accuracy
            if self.args.cb_finetune:
                num_of_testing = iteration - start_iter + 1
                avg_cumul_acc_fc = np.sum(after_cbf_top1_acc_list_ori[start_iter:,0])/num_of_testing
                avg_cumul_acc_icarl = np.sum(after_cbf_top1_acc_list_cumul[start_iter:,1])/num_of_testing
                print('Computing average accuracy...')
                print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
                print("  Average accuracy (Proto)      :\t\t{:.2f} %".format(avg_cumul_acc_icarl))

                #wandb.log({"avg_acc_fc": avg_cumul_acc_fc})
                #wandb.log({"avg_acc_proto": avg_cumul_acc_icarl})

                # Write the results to the tensorboard
                self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
                self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)
    
                # Save the results and close the tensorboard writer
                torch.save(after_cbf_top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
                torch.save(after_cbf_top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
                self.train_writer.close()

            else:
                num_of_testing = iteration - start_iter + 1
                avg_cumul_acc_fc = np.sum(top1_acc_list_cumul[start_iter:,0])/num_of_testing
                avg_cumul_acc_icarl = np.sum(top1_acc_list_cumul[start_iter:,1])/num_of_testing
                avg_cumul_acc_mixed = np.sum(top1_acc_list_cumul[start_iter:,2])/num_of_testing
                print('Computing average accuracy...')
                print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
                print("  Average accuracy (Proto)      :\t\t{:.2f} %".format(avg_cumul_acc_icarl))
                print("  Average accuracy (Mixed)      :\t\t{:.2f} %".format(avg_cumul_acc_mixed))

                #wandb.log({"avg_acc_fc": avg_cumul_acc_fc})
                #wandb.log({"avg_acc_proto": avg_cumul_acc_icarl})

                # Write the results to the tensorboard
                self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
                self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)
                self.train_writer.add_scalar('avg_acc/mixed', float(avg_cumul_acc_mixed), iteration)

                # Save the results and close the tensorboard writer
                torch.save(top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
                torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
                self.train_writer.close()
