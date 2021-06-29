"""
Main execution file for MAPF

-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
import torch
import random
import numpy as np


from utils.config import *
from agents import *

os.system("taskset -p -c 0 %d" % (os.getpid()))
#
# os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
os.system("taskset -p -c 0-7,16-23 %d" % (os.getpid()))
# os.system("taskset -p -c 0-7 %d" % (os.getpid()))
# os.system("taskset -p -c 0-3 %d" % (os.getpid()))
# os.system("taskset -p -c 0-1 %d" % (os.getpid()))

# os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

# Edit the env here
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def main():
    '''
    Main function
    Returns:
    '''

    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")

    '''
    See README.md for detailed explanations
    '''
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')

    arg_parser.add_argument('--mode', type=str, default='train')
    arg_parser.add_argument('--log_time_trained', type=str, default='0')

    arg_parser.add_argument('--num_agents', type=int, default=8)
    arg_parser.add_argument('--map_w', type=int, default=20)
    arg_parser.add_argument('--map_density', type=int, default=1)
    arg_parser.add_argument('--map_type', type=str, default='map')

    arg_parser.add_argument('--trained_num_agents', type=int, default=8)
    arg_parser.add_argument('--trained_map_w', type=int, default=20)
    arg_parser.add_argument('--trained_map_density', type=int, default=1)
    arg_parser.add_argument('--trained_map_type', type=str, default='map')

    arg_parser.add_argument('--nGraphFilterTaps', type=int, default=0)
    arg_parser.add_argument('--hiddenFeatures', type=int, default=0)
    arg_parser.add_argument('--numInputFeatures', type=int, default=128)

    arg_parser.add_argument('--num_testset', type=int, default=4500)
    arg_parser.add_argument('--load_num_validset', type=int, default=200)

    arg_parser.add_argument('--test_epoch', type=int, default=0)
    arg_parser.add_argument('--lastest_epoch', action='store_true', default=False)
    arg_parser.add_argument('--best_epoch', action='store_true', default=False)

    arg_parser.add_argument('--con_train', action='store_true', default=False)
    arg_parser.add_argument('--test_general', action='store_true', default=False)
    arg_parser.add_argument('--train_TL', action='store_true', default=False)
    arg_parser.add_argument('--exp_net_load', type=str, default=None)
    arg_parser.add_argument('--gpu_device', type=int, default=0)

    arg_parser.add_argument('--Use_infoMode', type=int, default=0)
    arg_parser.add_argument('--log_anime', action='store_true', default=False)
    arg_parser.add_argument('--rate_maxstep', type=int, default=2)

    arg_parser.add_argument('--vary_ComR_FOV', action='store_true', default=False)
    arg_parser.add_argument('--commR', type=int, default=7)
    arg_parser.add_argument('--dynamic_commR', action='store_true', default=False)
    arg_parser.add_argument('--symmetric_norm', action='store_true', default=False)

    arg_parser.add_argument('--FOV', type=int, default=9)
    arg_parser.add_argument('--id_env', type=int, default=None)
    arg_parser.add_argument('--guidance', type=str, default='Project_G')

    arg_parser.add_argument('--data_set', type=str, default='')

    arg_parser.add_argument('--update_valid_set', type=int, default=200)
    arg_parser.add_argument('--update_valid_set_epoch', type=int, default=100)

    arg_parser.add_argument('--threshold_SuccessRate', type=int, default=80)


    # Projected Goal                                    - Project_G
    # Local Guidance Static obstacle                    - LocalG_S
    # Local Guidance Static + Dynamic obstacle          - LocalG_SD
    # Semi Global Guidance Static + Dynamic obstacle    - SemiLG_SD
    # Global Guidance Static obstacle                   - GlobalG_S
    # Global Guidance Static + Dynamic obstacle         - GlobalG_SD

    arg_parser.add_argument('--action_select', type=str, default='soft_max')
    # softmax + max   -- soft_max                       - soft_max
    # nomralize + multinomial                           - sum_multinorm
    # exp + multinomial                                 - exp_multinorm
    arg_parser.add_argument('--nAttentionHeads', type=int, default=0)

    arg_parser.add_argument('--AttentionConcat', action='store_true', default=False)
    arg_parser.add_argument('--test_num_processes', type=int, default=2)
    arg_parser.add_argument('--test_len_taskqueue', type=int, default=4)
    arg_parser.add_argument('--test_checkpoint', action='store_true', default=False)
    arg_parser.add_argument('--test_checkpoint_restart', action='store_true', default=False)
    arg_parser.add_argument('--old_simulator', action='store_true', default=False)
    arg_parser.add_argument('--batch_numAgent', action='store_true', default=False)

    arg_parser.add_argument('--no_ReLU', action='store_true', default=False)

    arg_parser.add_argument('--use_Clip', action='store_true', default=False)

    arg_parser.add_argument('--tb_ExpName', type=str, default='')

    arg_parser.add_argument('--attentionMode', type=str, default='GAT_modified')
    # attentionMode
    #               - GAT_origin
    #               - GAT_modified
    #               - KeyQuery
    #               - GAT_DualHead
    #               - GAT_Similarity

    arg_parser.add_argument('--return_attentionGSO', action='store_true', default=False)
    arg_parser.add_argument('--use_dropout', action='store_true', default=False)

    arg_parser.add_argument('--GSO_mode', type=str, default='dist_GSO')
    # GSO_mode
    #               - dist_GSO
    #               - dist_GSO_one      - dist_GSO >0 = 1
    #               - full_GSO          - fully connective graph
    arg_parser.add_argument('--LSTM_seq_len', type=int, default=8)
    arg_parser.add_argument('--LSTM_seq_padding', action='store_true', default=False)
    arg_parser.add_argument('--label_smoothing', type=float, default=0.0)
    arg_parser.add_argument('--bottleneckMode', type=str, default=None)
    # Current Support
    #               attentionMode  -- Key and Query
    #                               BottomNeck_only
    #                               BottomNeck_skipConcat
    #                               BottomNeck_skipConcatGNN
    #                               BottomNeck_skipAddGNN

    arg_parser.add_argument('--bottleneckFeature', type=int, default=128)
    arg_parser.add_argument('--use_dilated', action='store_true', default=False)
    arg_parser.add_argument('--use_dilated_version', type=int, default=1)
    arg_parser.add_argument('--GNNGAT', action='store_true', default=False)
    arg_parser.add_argument('--CNN_mode', type=str, default="Default")
    '''
    CNN_mode
            - Default
            
    ResNetMode
            - ResNetSlim 
                            - Resnet - 2 blocks
            - ResNetLarge 
                            - Resnet - 3 blocks
            
            - ResNetLarge_withMLP
                            - Resnet - 2 blocks + MLP
                            
            - ResNetSlim_withMLP
                            - Resnet - 3 blocks + MLP
                            
    '''

    arg_parser.add_argument('--list_agents', nargs='+', type=int)
    arg_parser.add_argument('--list_map_w', nargs='+', type=int)
    arg_parser.add_argument('--list_num_testset', nargs='+', type=int)

    arg_parser.add_argument('--shuffle_testSet', action='store_true', default=False)

    arg_parser.add_argument('--test_on_ValidSet', action='store_true', default=False)
    arg_parser.add_argument('--list_model_epoch', nargs='+', type=int)
    arg_parser.add_argument('--default_actionSelect', action='store_true', default=False)

    arg_parser.add_argument('--load_memory', action='store_true', default=False)

    args = arg_parser.parse_args()

    if args.mode == 'test' and args.test_general:
        ## loop in different pair of map size and number of agent

        num_setup = len(args.list_agents)

        for id_case in range(num_setup):
            args.num_agents = args.list_agents[id_case]
            args.map_w = args.list_map_w[id_case]
            args.num_testset = args.list_num_testset[id_case]
            # parse the config json file
            config = process_config(args)
            # Create the Agent and pass all the configuration to it then run it..
            agent_class = globals()[config.agent]
            agent = agent_class(config)
            agent.run()
            agent.finalize()
    else:

        # parse the config json file
        config = process_config(args)
        # Create the Agent and pass all the configuration to it then run it..
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        agent.run()
        agent.finalize()


if __name__ == '__main__':
    main()
