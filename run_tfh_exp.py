import os

def run_exp(dataset='cifar100', alternative_data='all', nb_cl_fg=50, nb_cl=2, num_phase_search=1, nb_protos=20, baseline='lucir', net_type='ss', net_arch='std_resnet18', gpu=0, label='0001', loss_feature_KD_weight=1.0, loss_MR_weight=1.0, exp_id=1):

    random_seed=1993
    the_command = 'python main.py' 

    the_command += ' --nb_cl_fg=' + str(nb_cl_fg)
    the_command += ' --nb_cl=' + str(nb_cl)
    the_command += ' --nb_protos=' + str(nb_protos) 
    the_command += ' --gpu=' + str(gpu) 
    the_command += ' --random_seed=' + str(random_seed)  
    the_command += ' --baseline=' + baseline 
    the_command += ' --dynamic_budget' 
    the_command += ' --branch_type=' + net_type 
    the_command += ' --net_arch=' + net_arch 
    the_command += ' --loss_MR_weight=' + str(loss_MR_weight) 
    the_command += ' --loss_feature_KD_weight=' + str(loss_feature_KD_weight)      
    the_command += ' --num_phase_search=' + str(num_phase_search) 
    the_command += ' --update_lr'  

    if dataset=='cifar100':
        the_command += ' --dataset=cifar100'

    elif dataset=='imagenet_sub':
        the_command += ' --dataset=imagenet_sub'
        the_command += ' --test_batch_size=50' 
        the_command += ' --epochs=90'  
        the_command +=  ' --num_workers=1' 
        the_command +=  ' --custom_weight_decay=0.0005' 
        the_command +=  ' --test_batch_size=50' 
        the_command +=  ' --the_lambda=10' 
        the_command +=  ' --K=2' 
        the_command +=  ' --dist=0.5' 
        the_command +=  ' --lw_mr=1' 
        the_command +=  ' --base_lr1=0.05' 
        the_command +=  ' --base_lr2=0.05' 

    elif dataset=='imagenet':
        the_command += ' --dataset=imagenet'
        the_command += ' --epochs=90'
        the_command += ' --num_classes=1000'
        the_command +=  ' --num_workers=16' 
        the_command +=  ' --custom_weight_decay=1e-4' 
        the_command +=  ' --test_batch_size=50' 
    else:
        raise ValueError('Please set correct dataset.')

    the_command += ' --ckpt_label=' + label
    
    os.system(the_command)

exp_id = 0
nb_protos_list = [20]
nb_cl_list = [10, 2]
net_arch_list = ['resnet32']
net_type_list = ['free']
loss_feature_KD_weight_list = [1]
loss_MR_weight_list = [1]
num_phase_search_list = [1]
the_gpu = 0

the_baseline = 'mixed'
for the_new_idx in range(1):
    for the_nb_protos in nb_protos_list:
        for the_nb_cl in nb_cl_list:
            for the_net_type in net_type_list:
                for the_net_arch in net_arch_list:
                    for the_loss_feature_KD_weight in loss_feature_KD_weight_list:
                        for the_loss_MR_weight in loss_MR_weight_list:
                            for the_num_phase_search in num_phase_search_list:
                                run_exp(dataset='cifar100', nb_cl_fg=50, nb_cl=the_nb_cl, num_phase_search=the_num_phase_search, nb_protos=the_nb_protos, baseline=the_baseline, net_type=the_net_type, net_arch=the_net_arch, gpu=the_gpu, label='exp'+str(exp_id), loss_feature_KD_weight=the_loss_feature_KD_weight, loss_MR_weight=the_loss_MR_weight, exp_id=exp_id)
                                exp_id += 1
