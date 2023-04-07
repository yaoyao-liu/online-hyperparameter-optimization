""" Main function for this project. """
import os
import argparse
import numpy as np
from trainer.trainer import Trainer
from utils.gpu_tools import occupy_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--gpu', default='0', help='the index of GPU')
    parser.add_argument('--multiple_gpu', action='store_true')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--imgnet_split', default='icarl', type=str, choices=['icarl', 'podnet'])
    parser.add_argument('--net_arch', default='std_resnet50', type=str, choices=['resnet32', 'std_resnet18', 'std_resnet34', 'std_resnet50', 'std_resnet101', 'wrn_28_10', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'vit'])
    parser.add_argument('--baseline', default='lucir', type=str, choices=['lucir', 'icarl', 'mixed'], help='baseline method')
    parser.add_argument('--ckpt_label', type=str, default='exp01', help='the label for the checkpoints')
    parser.add_argument('--ckpt_dir_fg', type=str, default='-', help='the checkpoint file for the 0-th phase')
    parser.add_argument('--resume_fg', action='store_true', help='resume 0-th phase model from the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from the checkpoints')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for loading data')
    parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
    parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
    parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test loader')
    parser.add_argument('--eval_batch_size', default=128, type=int, help='the batch size for validation loader')
    parser.add_argument('--disable_gpu_occupancy', action='store_false', help='disable GPU occupancy')

    ### Network architecture parameters
    parser.add_argument('--branch_type', default='ss', type=str, choices=['ss', 'fixed', 'free'], help='the network type for the first branch')

    ### Incremental learning parameters
    parser.add_argument('--num_classes', default=100, type=int, help='the total number of classes')
    parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in the 0-th phase')
    parser.add_argument('--nb_cl', default=10, type=int, help='the number of classes for each phase')
    parser.add_argument('--nb_protos', default=20, type=int, help='the number of exemplars for each class')
    parser.add_argument('--epochs', default=160, type=int, help='the number of epochs')
    parser.add_argument('--dynamic_budget', action='store_true', help='using dynamic budget setting')

    ### General learning parameters
    parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
    parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
    parser.add_argument('--base_lr1', default=0.1, type=float, help='learning rate for the 0-th phase')
    parser.add_argument('--base_lr2', default=0.1, type=float, help='learning rate for the following phases')

    ### LUCIR parameters
    parser.add_argument('--the_lambda', default=5, type=float, help='lamda for LF')
    parser.add_argument('--dist', default=0.5, type=float, help='dist for margin ranking losses')
    parser.add_argument('--K', default=2, type=int, help='K for margin ranking losses')
    parser.add_argument('--lw_mr', default=1, type=float, help='loss weight for margin ranking losses')

    ### iCaRL parameters
    parser.add_argument('--icarl_beta', default=0.25, type=float, help='beta for iCaRL')
    parser.add_argument('--icarl_T', default=2, type=int, help='T for iCaRL')

    ### Server settings
    parser.add_argument('--using_msr_server', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')

    ### Loss control
    parser.add_argument('--loss_feature_KD_weight', default=1.0, type=float)
    parser.add_argument('--loss_normal_KD_weight', default=1.0, type=float)
    parser.add_argument('--loss_MR_weight', default=1.0, type=float)

    ### Class balance finetuning
    parser.add_argument('--cb_finetune', action='store_true', help='class balance finetune')
    parser.add_argument('--ft_epochs', default=20, type=int, help='Epochs for class balance finetune')
    parser.add_argument('--ft_base_lr', default=0.01, type=float, help='Base learning rate for class balance finetune')
    parser.add_argument('--ft_lr_strat', default=[10], type=int, nargs='+', help='Lr_strat for class balance finetune')
    parser.add_argument('--ft_flag', default=2, type=int, help='Flag for class balance finetune')

    # New settings
    parser.add_argument('--num_phase_search', default=1, type=int)
    parser.add_argument('--update_lr', action='store_true')
    parser.add_argument('--kd_weight_iteration', default=1, type=int)
    parser.add_argument('--lr_iteration', default=1, type=int)

    args = parser.parse_args()

    the_args = parser.parse_args()

    # Checke the number of classes, ensure they are reasonable
    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    # Print the parameters
    print(the_args)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    # Occupy GPU memory in advance
    if the_args.disable_gpu_occupancy and not the_args.multiple_gpu:
        occupy_memory(the_args.gpu)
        print('Occupy GPU memory in advance.')

    # Set the trainer and start training
    trainer = Trainer(the_args)
    trainer.train()
