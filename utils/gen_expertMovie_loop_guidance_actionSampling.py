#!/usr/bin/env python3
import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import animation
from matplotlib import lines
import matplotlib.animation as manimation
import argparse
import math
import gc
import seaborn as sns
import time
import scipy.io as sio
import sys
from easydict import EasyDict
np.set_printoptions(threshold=np.inf)
import os

from utils.visualize_expertAlg import Animation

if __name__ == "__main__":
    #Define the root path that store the data

    DATA_FOLDER = '../MultiAgentDataset/Results_best/AnimeDemo/'
    Setup_comR = 'commR_7'

    list_action = ['exp_multinorm']

    exp_setup = [
        ('dcpOE', 'K2_HS0', '1602190896'),
    ]
    list_guidance = ['Project_G']

    ############################################
    ############################################
    # ID_map, ID_case, ID_agent

    map_setup = 'map20x20_rho1_10Agent'

    selected_case = [
        (427, 0, 2),
    ]

    # selected_case = [

    map_size =  map_setup.split('map')[1].split('x')[0]
    num_guidance = len(list_guidance)
    num_action = len(list_action)
    num_exp = len(exp_setup)

    for id_exp in range(num_exp):

        for index_case in range(0,len(selected_case)):
            Id_map = selected_case[index_case][0]

            Id_case = selected_case[index_case][1]

            Id_agent = selected_case[index_case][2]

            for label_guidance in list_guidance:

                for label_action in list_action:


                    Setup = os.path.join(exp_setup[id_exp][0],map_setup, exp_setup[id_exp][1],'TR_M20p1_10Agent',exp_setup[id_exp][2])


                    expnet_label = str(exp_setup[id_exp][0])
                    network_label = str(exp_setup[id_exp][1])
                    print(network_label)

                    K = int(network_label.split('K')[-1].split('_')[0])

                    Data_path = os.path.join(DATA_FOLDER, Setup, label_guidance,label_action, Setup_comR)

                    file_name_suffix = 'map{}x{}_IDMap{:05d}_IDCase{:05d}'.format(map_size, map_size, Id_map, Id_case)
                    Path_sol = os.path.join(Data_path, 'target', 'expert_{}.yaml'.format(file_name_suffix))


                    Path_map = os.path.join(Data_path, 'input', 'input_{}.yaml'.format(file_name_suffix))
                    Path_GSO = os.path.join(Data_path, 'GSO','predict_{}.mat'.format(file_name_suffix))

                    Path_video = os.path.join(DATA_FOLDER, 'video',map_setup, 'IDMap{:05d}'.format(Id_map), 'IDCas{:05d}'.format(Id_case))
                    # Path_video = os.path.join(DATA_FOLDER, 'video', map_setup, 'IDMap{:05d}'.format( int(id_map)))
                    # Path_video = os.path.join(DATA_FOLDER, 'video', map_setup, '{}'.format(id_map))

                    try:
                        # Create target Directory
                        os.makedirs(Path_video)
                    except FileExistsError:
                        pass
                    # print(Path_video)

                    Name_video = os.path.join(Path_video, '{}_K{}_{}_{}_IDMap{}_IDcase{}_{}_{}_{}_expert.mp4'.format(
                        exp_setup[id_exp][0], K, Setup_comR, exp_setup[id_exp][2], Id_map, Id_case, label_guidance,
                        label_action, Id_agent))

                    # Name_video = '{}/expert_{}_{}_{}_K3_{}.mp4'.format(Path_video, exp_setup[id_exp][2], Setup_comR, Id_case, Id_agent)
                    # print(Name_video)
                    config = {'map': Path_map,
                              'schedule': Path_sol,
                              'GSO': Path_GSO,
                              'nGraphFilterTaps': K,
                              'id_chosenAgent': Id_agent,
                              'video': Name_video,
                              'speed': 2,
                              }
                    config_setup = EasyDict(config)

                    print("############################## ")
                    print("ExpNet: {}\t Network Label:{}\t TimeStamp:{} \t Guidance:{}\t Action:{}".format(
                        exp_setup[id_exp][0], exp_setup[id_exp][1], exp_setup[id_exp][2], label_guidance, label_action))
                    print("\t\t\t\t ID Map {} \t ID Case {} \t ID Agent {}\t ".format(Id_map, Id_case, Id_agent))

                    animation = Animation(config_setup)

                    # animation.show()
                    if config_setup.video:
                        print(config_setup.video)
                        animation.save(config_setup.video, config_setup.speed)
                        print('Movie generation finished.')
                    else:
                        animation.show()

                time.sleep(5)
