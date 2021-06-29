from scipy.io import loadmat
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
import fnmatch
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import rc
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = "serif"
plt.rcParams.update({'font.size': 20})
# rc('text', usetex=True)
# Say, "the default sans-serif font is COMIC SANS"
# plt.rcParams['font.serif'] = "Palatino"
# Then, "ALWAYS use sans-serif fonts"
# rc('font',**{'family':'serif','serif':['Palatino']})

# fm = matplotlib.font_manager.json_load(os.path.expanduser(" ~/.cache/matplotlib/fontlist-v300.json"))
# fm.findfont("serif", rebuild_if_missing=False)
# fm.findfont("serif", fontext="afm", rebuild_if_missing=False)

# rc('text', usetex=True)
# print(plt.style.available)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def return_config(title, id=0):
    if id == 0:
        ################## Figure 1 Compare GNN alone ###################
        title += '_GNN_comparison'
        img_config = {
            '1602191336': {
                'selection': {
                    'trained_model_epoch': 240,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN_baseline-F-128',
                'color': 'goldenrod',
                'dash': '',
            },
            '1602190896': {
                'selection': {
                    'trained_model_epoch': 228,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-128',
                'color': 'red',
                'dash': '',
            },
            # '1600770037': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'GNN-F-128',
            #     'color': 'red',
            #     'dash': '',
            # },
            '1601435304': {
                'selection': {
                    'trained_model_epoch': 172,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-64',
                'color': 'blue',
                'dash': '',
            },

            '1600986429': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-64',
                'color': 'blue',
                'dash': (3, 1),
            },
            '1601023487': {
                'selection': {
                    'trained_model_epoch': 208,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-32',
                'color': 'black',
                'dash': '',
            },

            '1601070736': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-32',
                'color': 'black',
                'dash': (3, 1),
            },
            '1601079602': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-16',
                'color': 'bright purple',
                'dash': '',
            },

            '1601106566': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-16',
                'color': 'bright purple',
                'dash': (3, 1),
            },
        }

    elif id == 1:
        ################## Figure 2 Compare MAGAT alone ###################
        title += '_GAT_comparison'
        img_config = {
            '1602191707': {
                'selection': {
                    'trained_model_epoch': 272,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GAT-F-128',
                'color': 'goldenrod',
                'dash': '',
            },
            '1602191363': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-128',
                'color': 'red',
                'dash': '',
            },
            # '1600770040': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'MAGAT-F-128',
            #     'color': 'red',
            #     'dash': '',
            # },
            '1601044729': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-64',
                'color': 'blue',
                'dash': '',
            },
            '1600986640': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-64',
                'color': 'blue',
                'dash': (3, 1),
            },
            '1601023527': {
                'selection': {
                    'trained_model_epoch': 288,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-32',
                'color': 'black',
                'dash': '',
            },
            '1601078926': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-32',
                'color': 'black',
                'dash': (3, 1),
            },
            '1601079593': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-16',
                'color': 'bright purple',
                'dash': '',
            },
            '1601106570': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-16',
                'color': 'bright purple',
                'dash': (3, 1),
            },

            ############### mutlihead MAGAT #################
            # '1601918482': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'MAGAT-F-64-P4',
            #     'color': 'green',
            #     'dash': '',
            # },
            '1601918499': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-32-P4',
                'color': 'green',
                'dash': '',
            },
            '1601918505': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-32-P4',
                'color': 'green',
                'dash': (3, 1),
            },
        }

    elif id == 2:
        ################### Figure 3 Compare MAGAT and AGNN ###################
        title += '_GNN_GAT'
        img_config = {
            '1602190896': {
                'selection': {
                    'trained_model_epoch': 228,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-128',
                'color': 'red',
                'dash': (3, 1),
            },
            '1601435304': {
                'selection': {
                    'trained_model_epoch': 172,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-64',
                'color': 'blue',
                'dash': (3, 1),
            },

            '1600986429': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-64',
                'color': 'sky blue',
                'dash': (3, 1),
            },
            '1601023487': {
                'selection': {
                    'trained_model_epoch': 208,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-32',
                'color': 'light grey',
                'dash': (3, 1),
            },

            '1601070736': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-32',
                'color': 'black',
                'dash': (3, 1),
            },
            '1601079602': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-16',
                'color': 'bright purple',
                'dash': (3, 1),
            },

            '1601106566': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-B-16',
                'color': 'lilac',
                'dash': (3, 1),
            },
            #####
            '1602191363': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-128',
                'color': 'red',
                'dash': '',
            },
            '1601044729': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-64',
                'color': 'blue',
                'dash': '',
            },
            '1600986640': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-64',
                'color': 'sky blue',
                'dash': '',
            },
            '1601023527': {
                'selection': {
                    'trained_model_epoch': 288,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-32',
                'color': 'light grey',
                'dash': '',
            },
            '1601078926': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-32',
                'color': 'black',
                'dash': '',
            },
            '1601079593': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-16',
                'color': 'bright purple',
                'dash': '',
            },
            '1601106570': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-16',
                'color': 'lilac',
                'dash': '',
            },
        }
    elif id == 3:
        ################## Figure 4 Large Scale ###################
        title += '_Large_Scale'
        img_config = {
            '1600770037': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN-F-128',
                'color': 'red',
                'dash': '',
            },
            '1600770040': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-128',
                'color': 'blue',
                'dash': '',
            },
            '1601918499': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-32-P4',
                'color': 'green',
                'dash': '',
            },
            '1601918505': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-B-32-P4',
                'color': 'green',
                'dash': (3, 1),
            },
        }

    elif id == 10:
        title += '_ZJU_talk'
        img_config = {
            '1602191336': {
                'selection': {
                    'trained_model_epoch': 240,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GNN_baseline-F-128',
                'color': 'goldenrod',
                'dash': (3, 1),
            },
            # '1602190896': {
            #     'selection': {
            #         'trained_model_epoch': 228,
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'GNN-F-128',
            #     'color': 'red',
            #     'dash': (3, 1),
            # },

            # '1601023487': {
            #     'selection': {
            #         'trained_model_epoch': 208,
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'GNN-F-32',
            #     'color': 'powder blue',
            #     'dash': (3, 1),
            # },

            # '1601070736': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'GNN-B-32',
            #     'color': 'cobalt',
            #     'dash': (3, 1),
            # },

            '1602191707': {
                'selection': {
                    'trained_model_epoch': 272,
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'GAT-F-128',
                'color': 'red',
                'dash': '',
            },
            '1602191363': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-128',
                # 'color': 'red',
                'color': 'blue',

                'dash': '',
            },

            # '1601023527': {
            #     'selection': {
            #         'trained_model_epoch': 288,
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'MAGAT-F-32',
            #     'color': 'bright purple',
            #     'dash': '',
            # },
            # '1601078926': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'MAGAT-B-32',
            #     'color': 'violet',
            #     'dash': '',
            # },

            '1601918499': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                },
                'label': 'MAGAT-F-32-P4',
                'color': 'green',
                'dash': '',
            },
            # '1601918505': {
            #     'selection': {
            #         'trained_model_epoch': 'b',
            #         'action_policy': 'exp_multinorm',
            #         'K': 2,
            #     },
            #     'label': 'MAGAT-B-32-P4',
            #     'color': 'forrest green',
            #     'dash': '',
            # },


        }


    return title, img_config

    ######## NOT USED #########
    # '1601597881': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'AGNN_OE_F64_SkipMLPAGNN',
    #     'color': 'burnt orange',
    #     'dash': (4, 3),
    # }
    # '1601670254': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'AGNN_OE_F32_SkipMLPAGNN_NODATA',
    #     'color': 'orange red',
    #     'dash': (4, 3),
    # },
    # '1601108329': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'AGNN_OE_F16_SkipMLPAGNN_RESELECT',
    #     'color': 'pinky purple',
    #     'dash': (4, 3),
    # },
    # '1601694357': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'MAGAT_OE_F64_SkipMLPAGNN',
    #     'color': 'ocean blue',
    #     'dash':  (4, 3),
    # },
    # '1601045034': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'MAGAT_OE_F32_SkipMLPAGNN',
    #     'color': 'blue violet',
    #     'dash': (4, 3),
    # },
    # '1601108460': {
    #     'selection': {
    #         'trained_model_epoch': 'b',
    #         'action_policy': 'exp_multinorm',
    #         'K': 2,
    #     },
    #     'label': 'MAGAT_OE_F16_SkipMLPAGNN_DOWN',
    #     'color': 'cobalt blue',
    #     'dash': (4, 3),
    # },


class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER, target_data, list_action_policy, list_metrics, list_num_agents, list_map_size):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
        self.target_data = target_data
        self.list_metrics = list_metrics
        # self.labels = labels
        self.label_num_agents = list_num_agents
        # self.text_legend = text_legend
        self.list_action_policy = list_action_policy
        self.list_map_size = list_map_size
        self.load_data()

    def load_data(self):
        print(self.target_data)
        data = self.target_data

        pure_list = {}
        data_list = {}
        if 'Results_best' in self.DATA_FOLDER:
            self.default_mode = 'best'
        else:
            self.default_mode = 'select'
        for data_type in data.keys():
            for subdir, dirs, files in os.walk(os.path.join(self.DATA_FOLDER, data_type)):
                # print(os.path.join(self.DATA_FOLDER, data_type))
                print(files)
                for file in files:
                    # print os.path.join(subdir, file)
                    filepath = subdir + os.sep + file
                    if filepath.endswith(".mat"):
                        for log_time in data[data_type].keys():
                            for action_policy in self.list_action_policy:
                                # print(log_time)
                                if str(log_time) in filepath and action_policy in filepath:
                                    print(log_time, subdir, file)

                                    # num_agents = subdir.split('_')[-1].split('Agent')[0]
                                    # num_agents = int(num_agents)
                                    mat_data = loadmat(filepath)
                                    rate_ReachGoal = mat_data['rate_ReachGoal'][0][0]
                                    mean_deltaFT = mat_data['mean_deltaFT'][0][0]
                                    mean_deltaMP = mat_data['mean_deltaMP'][0][0]
                                    hidden_state = mat_data['hidden_state'][0][0]
                                    trained_model_epoch = mat_data.get('trained_model_epoch', [[-1]])
                                    trained_model_epoch = trained_model_epoch[0][0]

                                    num_agents_trained = mat_data['num_agents_trained'][0][0]
                                    num_agents_testing = mat_data['num_agents_testing'][0][0]
                                    map_size_testing = mat_data['map_size_testing'][0][0]
                                    K = mat_data['K'][0][0]

                                    cleaned_data = {
                                        'filename': file,
                                        'type': data_type,
                                        'action_policy': action_policy,
                                        'map_size_trained': mat_data['map_size_trained'][0],
                                        'map_density_trained': mat_data['map_density_trained'][0][0],
                                        'num_agents_trained': mat_data['num_agents_trained'][0][0],

                                        'map_size_testing': mat_data['map_size_testing'][0],
                                        'map_size_testing_int': mat_data['map_size_testing'][0][0],
                                        'map_density_testing': mat_data['map_density_testing'][0][0],
                                        'num_agents_testing': mat_data['num_agents_testing'][0][0],
                                        'log_time': log_time,
                                        'K': K,
                                        # 'data_set' : mat_data['data_set'][0][0],
                                        'hidden_state': hidden_state,
                                        'rate_ReachGoal': rate_ReachGoal,
                                        'mean_deltaFT': mean_deltaFT,
                                        'std_deltaMP': mat_data['std_deltaMP'][0][0],

                                        'mean_deltaMP': mean_deltaMP,
                                        'std_deltaFT': mat_data['std_deltaFT'][0][0],
                                        'trained_model_epoch': trained_model_epoch,
                                        'list_deltaFT': mat_data['list_deltaFT'][0],
                                        # 'list_FT_predict': mat_data['list_FT_predict'][0],
                                        # 'list_FT_target': mat_data['list_FT_target'][0],
                                        #
                                        # 'list_reachGoal': mat_data['list_reachGoal'][0],
                                        # 'list_numAgentReachGoal': mat_data['list_numAgentReachGoal'][0],
                                        'hist_numAgentReachGoal': mat_data['hist_numAgentReachGoal'][0],
                                    }
                                    data_list.setdefault(log_time, {}).setdefault(trained_model_epoch, {}).setdefault(action_policy, []).append(cleaned_data)
                                    metric_title = "{}({}x{})".format(num_agents_testing, map_size_testing, map_size_testing)
                                    data[data_type][log_time].setdefault(trained_model_epoch, {}).setdefault(action_policy, {}).setdefault(metric_title, []).append(cleaned_data)
                                    pure_list.setdefault(log_time, []).append(cleaned_data)

        self.data_list = data_list
        self.data = data
        self.pure_list = pure_list

    def print_table(self):
        print('All Data within the loaded dataset')
        for log_time in self.data_list.keys():
            for trained_model_epoch in self.data_list[log_time].keys():
                if trained_model_epoch == -1:
                    print('=====Model at {} trained at {}====='.format(self.default_mode, log_time))
                else:
                    print('=====Model at {} trained at {}====='.format(trained_model_epoch, log_time))
                for action_policy in self.list_action_policy:
                    print("-----Policy {}-----".format(action_policy))
                    list_data_log_time = self.data_list[log_time][trained_model_epoch][action_policy]
                    column_text = 'Metric_name\t'
                    for index, num_agents in enumerate(self.label_num_agents):
                        column_text+= '{}\t'.format("{}({}x{})".format(num_agents, self.list_map_size
                                                                       [index], self.list_map_size
                                                                       [index]))
                    row_texts = []
                    for metric in self.list_metrics:
                        row_text = '{}\t'.format(metric)
                        for index, num_agents in enumerate(self.label_num_agents):
                            scanned_list = [i for i in list_data_log_time
                                            if i['num_agents_testing'] == num_agents
                                            and i['map_size_testing'][0] == self.list_map_size
                                                                       [index]]
                            # print(log_time, action_policy, num_agents, len(scanned_list))
                            if len(scanned_list)>0:
                                row_text+='{}\t'.format(round(scanned_list[0][metric], 3))
                            else:
                                row_text += 'N/A\t'
                        row_texts.append(row_text)
                    print(column_text)
                    for row_text in row_texts:
                        print(row_text)


    def print_error_bar_img(self, config, title):
        for metrics in self.list_metrics:
            self.fig, ax = plt.subplots()




            if metrics == "rate_ReachGoal":
                label_text_y = "Success Rate"
                # ax.set_ylabel(r'$\alpha$')
                # ax.set_ylabel(label_text_y)
                self.fig.set_size_inches(8, 7)
            elif metrics == "mean_deltaFT":
                label_text_y = "Flowtime Increase"
                # ax.set_ylabel(r'$\delta_{FT}$')
                # ax.set_ylabel(label_text_y)
                self.fig.set_size_inches(8, 7)
            if 'Large' in title:
                self.fig.set_size_inches(8, 6)
            label_text_x = r'W$\times$H' + ' (#robots)'
            if 'Increase_Density' in title:
                if metrics == "rate_ReachGoal":
                    label_text_x = 'Increasing Robot Density Set'
            elif 'Large' in title:
                if metrics == "rate_ReachGoal":
                    label_text_x = 'Large Scale Map Set'
            else:
                if metrics == "rate_ReachGoal":
                    label_text_x = 'Same Robot Density Set'

            ax.set_xlabel(label_text_x)

            summary = []
            colors = []
            dashes = []
            for log_time in config.keys():
                colors.append(config[log_time]['color'])
                dashes.append(config[log_time]['dash'])
                select_data_list = self.pure_list[log_time]
                # pprint(select_data_list)
                conditions = config[log_time]['selection']

                satisfy_items = []
                for item in select_data_list:

                    satisfy = True
                    for condition_key, condition_value in conditions.items():
                        if item[condition_key] != condition_value:
                            satisfy = False

                    if satisfy:
                        satisfy_items.append(item)

                print('found results', len(satisfy_items))


                for index, num_agents in enumerate(self.label_num_agents):
                    column_text = "{}x{}({})".format(self.list_map_size[index],
                                                     self.list_map_size[index],
                                                     num_agents)
                    print(column_text)
                    # print(satisfy_items[index])

                    result_items = [item for item in satisfy_items
                                   if item['map_size_testing'][0] == self.list_map_size[index] and item['num_agents_testing'] == num_agents]
                    if len(result_items) == 0:
                        summary.append([config[log_time]['label'], column_text, 0])
                    else:
                        if metrics == 'mean_deltaFT':
                            values = result_items[0]['list_deltaFT']
                            for value in values:
                                summary.append([config[log_time]['label'], column_text, value])
                        else:
                            value = result_items[0][metrics]
                            summary.append([config[log_time]['label'], column_text, value])

            data = pd.DataFrame(summary, columns=['label', 'settings', metrics])
            print(data)

            with plt.rc_context({'lines.linewidth': 3}):
                ax = sns.lineplot(x='settings', y=metrics, data=data, hue='label', sort=False, style='label',
                                   palette=sns.xkcd_palette(colors), dashes=dashes)
            plt.grid()
            plt.xticks(rotation=20)
            if 'Large' in title:
                plt.xticks(rotation=0)
            if metrics == "rate_ReachGoal":

                handles, labels = ax.get_legend_handles_labels()
                if 'Large' in title:
                    ax.legend(handles=handles[0:], labels=labels[0:], loc='lower left', fontsize=17.5, ncol=1, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
                              labelspacing=0.05)

                else:
                    # ax.legend(handles=handles[0:], labels=labels[0:], loc='lower left',  fontsize=16, ncol=1, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
                    #               labelspacing=0.05)
                    ax.legend(handles=handles[0:], labels=labels[0:], loc='lower left', fontsize=17.5, ncol=1,
                              borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
                              labelspacing=0.05)
                # handles,labels = ax.get_legend_handles_labels()
                # ax.legend(handles=handles[1:], labels=labels[1:])
                start, end = ax.get_ylim()
                if 'Increase_Density' in title:
                    ax.set_ylim([0.4, 1])
                elif 'Large' in title:
                    ax.set_ylim([0, end*1.1])
                else:
                    ax.set_ylim([0.7, 1])
                # ax.legend(bbox_to_anchor=(-0.2, 1), borderaxespad=0)
            elif metrics == "mean_deltaFT":
                # ## GNN
                handles, labels = ax.get_legend_handles_labels()
                # ax.legend()

                ax.legend(handles=handles[0:], labels=labels[0:], loc='lower left', fontsize=17.5, ncol=1, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
                          labelspacing=0.05).remove()
                # ax.legend(loc='upper left', ncol=1, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,labelspacing=0.05)
                start, end = ax.get_ylim()


                # ax.set_ylim([0, end*1.5])
                # ax.set_ylim([0.02, 3])
                # ax.set_ylim([0.05, end * 1.1])
                if 'Increase_Density' in title:
                    ax.set_ylim([0.03, 0.25])
                elif 'Large' in title:
                    ax.set_ylim([0.03, end * 1.1])
                else:
                    ax.set_ylim([0.03, 0.14])

            plt.ylabel(label_text_y)
            plt.xlabel(label_text_x)
            plt.tight_layout()
            name_save_fig_pdf = "{}/{}_{}.pdf".format(self.DATA_FOLDER, metrics, title)
            self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)

            plt.savefig("{}/{}_{}".format(self.DATA_FOLDER, metrics, title))

    def print_hist_img(self, config):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(4, 3)

        self.ax.set_xlabel('# robot')
        width = 0.35  # the width of the bars
        label_width = 1.05


        self.ax.set_ylabel('Proportion of cases')

        for log_time in config.keys():
            select_data_list = self.pure_list[log_time]
            # pprint(select_data_list)
            conditions = config[log_time]['selection']

            satisfy_items = []
            for item in select_data_list:

                satisfy = True
                for condition_key, condition_value in conditions.items():
                    if item[condition_key] != condition_value:
                        satisfy = False

                if satisfy:
                    satisfy_items.append(item)

            print('found results', len(satisfy_items))
            # print(satisfy_items)
            label = config[log_time]['label']
            color = config[log_time]['color']
            hist_data = satisfy_items[0]['hist_numAgentReachGoal']
            print(label, hist_data)
            step_size = int(len(hist_data)/10)

            label_pos = np.arange(len(hist_data))
            hist_set = self.ax.bar(label_pos, hist_data,
                                   align='center', label=label, color=color,
                                   ls='dotted', lw=3, alpha=0.5)

            start, end = self.ax.get_xlim()
            self.ax.xaxis.set_ticks(np.arange(0, len(hist_data), step_size))

        handles, labels = self.ax.get_legend_handles_labels()
        # ax.legend()

        self.ax.legend(handles=handles[1:], labels=labels[1:])
        plt.savefig('test_histgram_img.png')

if __name__ == '__main__':

    DATA_FOLDER ='../MultiAgentDataset/Results_best_ICRA2021/Statistics_generalization_ICRA_Final_Lin/Statistics_generalization_ICRA_Draft/'
    title_text = "ICRA2021"

    # use_log = True
    use_log = False

    if use_log:
        title_text = "{}_logscale".format(title_text)
    else:
        title_text = "{}".format(title_text)

    target_data = {
        'dcpOE': {
                '1602190896': {},
                '1602191336': {},

                '1600770037': {},

                '1601435304': {},
                '1601597881': {},
                '1600986429': {},

                '1601023487': {},
                '1601670254': {},
                '1601070736': {},

                '1601079602': {},
                '1601108329': {},
                '1601106566': {},
            },
            'dcpOEGAT': {
                '1602191363': {},
                '1602191707': {},
                '1600770040': {},

                '1601044729': {},
                '1601694357': {},
                '1600986640': {},

                '1601023527': {},
                '1601045034': {},
                '1601078926': {},

                '1601079593': {},
                '1601108460': {},
                '1601106570': {},

                '1601918482': {},
                '1602004717': {},
                '1601918499': {},
                '1601918505': {},
            },
    }

    # pd.set_option('display.max_rows', None)
    # list_num_agents = [10, 20, 30, 40, 50, 60, 100, 100]
    # list_map_size = [20, 28, 35, 40, 45, 50, 65, 50]
    # title = 'Same_Effective_Density'
    # list_num_agents.extend([10, 20, 30, 40, 50, 60, 100])
    # list_map_size.extend([50, 50, 50, 50, 50, 50,  50])

    # list_num_agents = [10, 20, 30, 40, 50, 60, 100]
    # list_map_size = [50, 50, 50, 50, 50, 50,  50]
    # list_map_size = [100, 100, 100, 100, 100, 100,  100]
    # list_metrics = ['rate_ReachGoal', 'mean_deltaFT', 'mean_deltaMP']
    list_metrics = ['rate_ReachGoal', 'mean_deltaFT']
    list_action_policy = ['exp_multinorm']
    # list_metrics = ['rate_ReachGoal',]
    # # list_metrics = [ 'mean_deltaFT']
    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)

    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    for title in ['Same_Effective_Density', 'Increase_Density']:
        if title == 'Same_Effective_Density':
            list_num_agents = [10, 20, 30, 40, 50, 60, 100]
            list_map_size = [20, 28, 35, 40, 45, 50, 65]
        else:
            list_num_agents = [10, 20, 30, 40, 50, 60, 100]
            list_map_size = [50, 50, 50, 50, 50, 50, 50]

        ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, target_data, list_action_policy, list_metrics, list_num_agents, list_map_size)
        # ResultAnalysis.summary_result(title_text, text_legend, save_fig=True, use_log_scale=use_log, save_table=True)
        ResultAnalysis.print_table()
    #

    #     '''
    #     id: 0 figure 1 GNN
    #         1 figure 2 GAT
    #         2 figure 3 GNN and GAT
    #         10 figure 4 ZJU_talk
    #     '''

        extended_title, img_config = return_config(title=title, id=10)

        print(extended_title)
        ResultAnalysis.print_error_bar_img(img_config, extended_title)

    # title = 'Large Scale Map Set'
    # list_num_agents = [500, 1000, 500]
    # list_map_size = [200, 200, 100]
    #
    # ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, target_data, list_action_policy, list_metrics,
    #                                    list_num_agents, list_map_size)
    #
    # # ResultAnalysis.summary_result(title_text, text_legend, save_fig=True, use_log_scale=use_log, save_table=True)
    #
    #
    # ResultAnalysis.print_table()
    #
    # '''
    # id: 0 figure 1 GNN
    #     1 figure 2 GAT
    #     2 figure 3 GNN and GAT
    # '''
    # extended_title, img_config = return_config(title=title, id=3)
    # print(extended_title)
    # ResultAnalysis.print_error_bar_img(img_config, extended_title)

    # hist_config = {
    #     '1598494744': {
    #         'selection': {
    #             'trained_model_epoch': 'b',
    #             'action_policy': 'exp_multinorm',
    #             'num_agents_testing': 100,
    #             'map_size_testing_int': 50,
    #             'K': 2,
    #         },
    #         'label': 'GAT_OE_F32',
    #         'color': 'b',
    #     },
    #     '1599332980': {
    #         'selection': {
    #             'trained_model_epoch': 'b',
    #             'action_policy': 'exp_multinorm',
    #             'num_agents_testing': 100,
    #             'map_size_testing_int': 50,
    #             'K': 2,
    #         },
    #         'label': 'GNN_OE_F32',
    #         'color': 'r',
    #     },
    # }
    # ResultAnalysis.print_hist_img(hist_config)






