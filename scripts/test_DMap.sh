#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES= 0, 1, 2

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


#####################################################################
#                                                                   #
#                  Control Group 40 - CG 40                         #
#               (relative coordination + DMAP )                     #
#                                                                   #
#####################################################################

# ##################################
# #########
# #########    Training  generation
# #########
# ##################################



## GNN 128

python main.py configs/dcp_OE_Random.json --mode test --test_epoch 240 --test_general --log_time_trained 1602191336   --nGraphFilterTaps 2  --trained_num_agents 10 --trained_map_w 20   --commR 7  --list_map_w 20   --list_agents 10   --list_num_testset 4500    --GSO_mode dist_GSO  --action_select exp_multinorm  --guidance Project_G --CNN_mode Default --batch_numAgent --test_num_processes 2   --tb_ExpName GNN_Resnet_3Block_distGSO_baseline_128
## GAT 128

python main.py configs/dcpGAT_OE_Random.json --mode test  --test_epoch 272 --test_general --log_time_trained 1602191707  --nGraphFilterTaps 2  --trained_num_agents 10 --trained_map_w 20   --commR 7  --list_map_w 20   --list_agents 10   --list_num_testset 4500    --GSO_mode dist_GSO  --action_select exp_multinorm  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2  --nAttentionHeads 1 --attentionMode GAT_origin  --tb_ExpName GAT_origin_Resnet_3Block_MLP128_distGSO_baseline_128

## MAGAT F-128
python main.py configs/dcpGAT_OE_Random.json --mode test --best_epoch --test_general --log_time_trained 1602191363  --test_checkpoint --nGraphFilterTaps 2  --trained_num_agents 10 --trained_map_w 20   --commR 7  --list_map_w 20   --list_agents 10   --list_num_testset 4500  --GSO_mode dist_GSO  --action_select exp_multinorm  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2  --nAttentionHeads 1 --attentionMode KeyQuery  --tb_ExpName DotProduct_GAT_Resnet_3Block_distGSO_baseline_128

## MAGAT F-32-P4

 python main.py configs/dcpGAT_OE_Random.json --mode test --best_epoch --test_general --log_time_trained 1601918499  --nGraphFilterTaps 2  --trained_num_agents 10 --trained_map_w 20   --commR 7   --list_map_w 20   --list_agents 10   --list_num_testset 4500 --GSO_mode dist_GSO  --action_select exp_multinorm  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2 --nAttentionHeads 4 --attentionMode KeyQuery  --bottleneckMode BottomNeck_only --bottleneckFeature 32  --tb_ExpName DotProduct_GAT_Resnet_3Block_distGSO_bottleneck_32

## MAGAT B-32-P4

python main.py configs/dcpGAT_OE_Random.json --mode test  --test_checkpoint  --best_epoch --test_general --log_time_trained 1601918505  --nGraphFilterTaps 2  --trained_num_agents 10 --trained_map_w 20   --commR 7   --list_map_w 20   --list_agents 10   --list_num_testset 4500   --GSO_mode dist_GSO  --action_select exp_multinorm  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2 --nAttentionHeads 4 --attentionMode KeyQuery  --bottleneckMode BottomNeck_skipConcatGNN --bottleneckFeature 32 --tb_ExpName DotProduct_GAT_Resnet_3Block_distGSO_bottleneck_SkipMLPGNN_64