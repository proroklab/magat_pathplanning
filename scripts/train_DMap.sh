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

python main.py configs/dcp_OE_Random.json --mode train   --map_density 1 --map_w 20 --num_agents 10  --nGraphFilterTaps 2   --trained_num_agents 10 --commR 7 --GSO_mode dist_GSO --update_valid_set 1000 --update_valid_set_epoch 70 --threshold_SuccessRate 97 --default_actionSelect --guidance Project_G --CNN_mode ResNetLarge_withMLP --batch_numAgent --test_num_processes 2   --tb_ExpName GNN_Resnet_3Block_distGSO_baseline_128

## GAT 128

python main.py configs/dcpGAT_OE_Random.json --mode train --map_density 1 --map_w 20 --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10  --commR 7  --load_num_validset 1000 --update_valid_set 1000 --update_valid_set_epoch 100 --threshold_SuccessRate 90 --GSO_mode dist_GSO --default_actionSelect --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2  --nAttentionHeads 1 --attentionMode GAT_origin  --tb_ExpName GAT_origin_Resnet_3Block_MLP128_distGSO_baseline_128

## MAGAT F-128

python main.py configs/dcpGAT_OE_Random.json --mode train --map_density 1 --map_w 20 --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10  --commR 7  --load_num_validset 1000 --update_valid_set 1000 --update_valid_set_epoch 100 --threshold_SuccessRate 90 --GSO_mode dist_GSO --default_actionSelect --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2  --nAttentionHeads 1 --attentionMode KeyQuery  --tb_ExpName DotProduct_GAT_Resnet_3Block_distGSO_baseline_128

## MAGAT F-32-P4

python main.py  configs/dcpGAT_OE_Random.json --mode train   --map_density 1 --map_w 20 --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10  --commR 7 --load_num_validset 1000 --update_valid_set 1000 --update_valid_set_epoch 70 --threshold_SuccessRate 90 --GSO_mode dist_GSO  --default_actionSelect  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2 --nAttentionHeads 4  --attentionMode KeyQuery  --bottleneckMode BottomNeck_only --bottleneckFeature 32  --tb_ExpName DotProduct_GAT_Resnet_3Block_distGSO_bottleneck_32

## MAGAT F-B-32

python main.py  configs/dcpGAT_OE_Random.json --mode train   --map_density 1 --map_w 20 --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10  --commR 7 --load_num_validset 1000 --update_valid_set 1000 --update_valid_set_epoch 70 --threshold_SuccessRate 90 --GSO_mode dist_GSO  --default_actionSelect  --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent --test_num_processes 2 --nAttentionHeads 4  --attentionMode KeyQuery  --bottleneckMode BottomNeck_skipConcatGNN --bottleneckFeature 32  --tb_ExpName DotProduct_GAT_Resnet_3Block_MLP128_distGSO_bottleneck_SkipGNNConcat_32