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
                # print(files)
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
                                        'list_reachGoal': mat_data.get('list_reachGoal', None),
                                        'list_computationTime': mat_data.get('list_computationTime', None),
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
                                        'list_numAgentReachGoal': mat_data['list_numAgentReachGoal'][0],
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


    def print_analysis_large_set(self,config):
        self.fig, ax = plt.subplots()
        self.fig.set_size_inches(8, 6)
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
                # print(item['map_size_testing_int'], self.label_num_agents)
                if item['map_size_testing_int'] not in self.list_map_size:
                    satisfy = False
                if item['num_agents_testing'] not in self.label_num_agents:
                    satisfy = False

                if satisfy:
                    satisfy_items.append(item)

            print('found results', len(satisfy_items))

            for item in satisfy_items:
                list_reachGoal = item['list_reachGoal'][0]
                list_computationTime = item['list_computationTime'][0]
                hist_numAgentReachGoal = item['hist_numAgentReachGoal']
                list_numAgentReachGoal = item['list_numAgentReachGoal']
                list_computationTime = np.array(list_computationTime)
                total_computationTime = np.sum(list_computationTime)
                total_computationTime_without_failure = np.sum(list_computationTime[np.array(list_reachGoal)==1])

                print("----- {} ---{}x{}({})-----".format(
                    config[log_time]['label'],
                    item['map_size_testing_int'],
                    item['map_size_testing_int'],
                    item['num_agents_testing'],
                ))
                total_success_number = np.count_nonzero(np.array(list_reachGoal))
                mean_computationTime = total_computationTime / len(list_computationTime)
                mean_computationTime_without_failure = total_computationTime_without_failure / total_success_number

                print('total_computationTime', total_computationTime)
                print('total_computationTime_without_failure', total_computationTime_without_failure)
                print('mean_computationTime', mean_computationTime)
                print('mean_computationTime_without_failure', mean_computationTime_without_failure)
                print('rate_ReachGoal', item['rate_ReachGoal'])
                print('mean_deltaFT', item['mean_deltaFT'])
                print('std_deltaFT', item['std_deltaFT'])
                print('list_numAgentReachGoal', list_numAgentReachGoal)

                print('{}({})'.format(
                    round(mean_computationTime_without_failure, 2),
                    round(np.std(list_computationTime[np.array(list_reachGoal)==1])), 2))
                # print(list_reachGoal, list_computationTime, total_computationTime, total_computationTime_without_failure)
            # print(satisfy_items)
            label = config[log_time]['label']
            # hist_data = satisfy_items[0]['hist_numAgentReachGoal']
            # print(label, hist_data)
            # step_size = int(len(hist_data)/10)



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
                    values = result_items[0]['list_numAgentReachGoal'] / num_agents
                    # total_success_number = np.count_nonzero(np.array(list_reachGoal))
                    for value in values:
                        summary.append([config[log_time]['label'], column_text, value])

        data = pd.DataFrame(summary, columns=['label', 'settings', '#agents'])
        print(data)

        with plt.rc_context({'lines.linewidth': 3}):
            ax = sns.lineplot(x='settings', y='#agents', data=data, hue='label', sort=False, style='label',
                               palette=sns.xkcd_palette(colors), dashes=dashes)
        plt.grid()
        # plt.xticks(rotation=20)
        ax.legend(loc='lower left', ncol=1, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
                  labelspacing=0.05)
        ax.set_ylim([0.985, 1.001])
        plt.ylabel('Percentage of Successful Agents')
        plt.xlabel(r'W$\times$H' + ' (#agents)')
        plt.tight_layout()
        name_save_fig_pdf = "{}/{}_{}.pdf".format(self.DATA_FOLDER, 'large_scale', 'success_agents')
        self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)

        plt.savefig("{}/{}_{}".format(self.DATA_FOLDER, 'large_scale', 'success_agents'))


    def print_hist_img(self, config):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 6)

        self.ax.set_xlabel('# robot')
        width = 0.35  # the width of the bars
        label_width = 1.05


        self.ax.set_ylabel('Proportion of cases')

        id_model = 0
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
            config_ls = config[log_time]['ls']
            config_fc = config[log_time]['fc']

            map_size_selected = conditions['map_size_testing_int']
            num_agent_selected = conditions['num_agents_testing']

            # hist_data = satisfy_items[0]['hist_numAgentReachGoal']
            # print(label, hist_data)
            # step_size = int(len(hist_data)/10)
            #
            # # label_pos = np.arange(len(hist_data))
            # label_pos = np.arange(len(hist_data))
            # hist_set = self.ax.bar(label_pos, hist_data,
            #                        align='center', label=label, color=color,
            #                        ls='dotted', lw=3, alpha=0.5)
            #
            # start, end = self.ax.get_xlim()
            # self.ax.xaxis.set_ticks(np.arange(0, len(hist_data), step_size))

            if num_agent_selected == 1000:
                num_start = 995
            elif num_agent_selected == 500 and map_size_selected == 100:
                num_start = 475
            elif num_agent_selected == 500 and map_size_selected == 200:
                num_start = 495

            total_num = 50
            hist_data_total = satisfy_items[0]['hist_numAgentReachGoal']/total_num

            hist_data = satisfy_items[0]['hist_numAgentReachGoal'][num_start:]/total_num
            print(label, hist_data)

            step_size = int((len(hist_data_total)+1-num_start)/5)
            print("step_size", step_size)
            # label_pos = np.arange(len(hist_data))
            bar_width = 0.22
            label_pos = np.arange(num_start, len(hist_data_total)) + bar_width*id_model - 1.5*bar_width
            print("label_pos",label_pos)
            # hist_set = self.ax.bar(label_pos, hist_data,
            #                        align='center', label=label, color=color,
            #                        ls=config_ls, lw=3, alpha=0.5)

            hist_set = self.ax.bar(label_pos, hist_data,
                                   align='center', label=label, color=color, width=bar_width,
                                   ls=config_ls, lw=1, alpha=0.5)

            start, end = self.ax.get_xlim()
            self.ax.xaxis.set_ticks(np.arange(num_start, len(hist_data_total)+1, step_size))
            id_model+=1

        self.ax.legend()
        # plt.savefig('test_histgram_img.png')
        name_save_fig_pdf = "{}/{}_{}_map{}_{}Agent.pdf".format(self.DATA_FOLDER, 'large_scale', 'success_agents_hist', map_size_selected, num_agent_selected)

        self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)

        plt.savefig("{}/{}_{}_map{}_{}Agent".format(self.DATA_FOLDER, 'large_scale', 'success_agents_hist', map_size_selected, num_agent_selected))



    def print_distplot_img(self, config):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 6)

        self.ax.set_xlabel('# robot')
        width = 0.35  # the width of the bars
        label_width = 1.05


        self.ax.set_ylabel('Proportion of cases')

        id_model = 0
        # list_label = ['','','','']
        list_label = []
        list_color = []
        summary_df= []
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
            config_ls = config[log_time]['ls']
            config_fc = config[log_time]['fc']

            map_size_selected = conditions['map_size_testing_int']
            num_agent_selected = conditions['num_agents_testing']

            list_label.append(label)
            list_color.append(color)
            if num_agent_selected == 1000:
                num_start = 995
            elif num_agent_selected == 500 and map_size_selected == 100:
                num_start = 475
            elif num_agent_selected == 500 and map_size_selected == 200:
                num_start = 495


            total_num = 50
            hist_data_total = satisfy_items[0]['hist_numAgentReachGoal']/total_num

            hist_data_count = satisfy_items[0]['hist_numAgentReachGoal'][num_start:]
            hist_data = satisfy_items[0]['hist_numAgentReachGoal'][num_start:]/total_num
            # print(label, hist_data)

            step_size = int((len(hist_data_total)+1-num_start)/5)
            # print("step_size", step_size)
            # label_pos = np.arange(len(hist_data))
            bar_width = 0.22
            label_pos = np.arange(num_start, len(hist_data_total)) + bar_width*id_model - 1.5*bar_width

            # label_pos = np.arange(num_start, len(hist_data_total))

            # print("label_pos",label_pos)

            id_model += 1

            for i in range(len(hist_data)):

                for j in range(hist_data_count[i]):
                    summary_df.append([label, label_pos[i], hist_data_count[i]])

        summary_pd = pd.DataFrame(summary_df, columns=['label', '# robot', 'Proportion of cases'])
        print(summary_pd)

        # hist_set = sns.histplot(data=summary_pd, x='# robot', hue='label', kde=True, stat="density", common_norm=False, multiple="stack", binwidth=bar_width)
        # hist_set = sns.histplot(data=summary_pd, x='# robot', hue='label', kde=True, stat="density", common_norm=False)

        hist_set = sns.histplot(data=summary_pd, x='# robot', label=label, hue='label', kde=True, stat="density", common_norm=False, binwidth=bar_width, palette=sns.xkcd_palette(list_color), line_kws={'linewidth':4})

        # hist_set = sns.displot(summary_pd, x='# robot', hue='label', y='Proportion of cases', kind="kde")
        # hist_set = sns.distplot(summary_pd['# robot'],  hue='label')

        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(num_start, len(hist_data_total)+1, step_size))

        # list_label.extend()
        print(list_label)
        self.ax.legend(list_label)
        handles, labels = self.ax.get_legend_handles_labels()
        print(handles, labels, list_label)
        handles.reverse()
        # list_label.reverse()
        self.ax.legend(handles=handles, labels=list_label)
        # plt.savefig('test_histgram_img.png')
        name_save_fig_pdf = "{}/{}_{}_map{}_{}Agent_distplot.pdf".format(self.DATA_FOLDER, 'large_scale', 'success_agents_hist', map_size_selected, num_agent_selected)

        self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)

        plt.savefig("{}/{}_{}_map{}_{}Agent_distplot".format(self.DATA_FOLDER, 'large_scale', 'success_agents_hist', map_size_selected, num_agent_selected))

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
    # list_metrics = ['rate_ReachGoal', 'mean_deltaFT']
    list_action_policy = ['exp_multinorm']
    list_metrics = ['rate_ReachGoal',]
    # list_metrics = [ 'mean_deltaFT']
    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    # for title in ['Same_Effective_Density', 'Increase_Density']:
    #     if title == 'Same_Effective_Density':
    #         list_num_agents = [10, 20, 30, 40, 50, 60, 100]
    #         list_map_size = [20, 28, 35, 40, 45, 50, 65]
    #     else:
    #         list_num_agents = [10, 20, 30, 40, 50, 60, 100]
    #         list_map_size = [50, 50, 50, 50, 50, 50, 50]
    list_num_agents = [500, 1000, 500]
    list_map_size = [200, 200, 100]
    # list_num_agents = [500]
    # list_map_size = [200]
    # list_num_agents = [500]
    # list_map_size = [100]
    ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, target_data, list_action_policy, list_metrics, list_num_agents, list_map_size)
    # ResultAnalysis.summary_result(title_text, text_legend, save_fig=True, use_log_scale=use_log, save_table=True)
    ResultAnalysis.print_table()

    for id_item in range(len(list_num_agents)):
        num_agent_selected = list_num_agents[id_item]
        map_size_selected = list_map_size[id_item]
        print(num_agent_selected, map_size_selected)

        hist_config = {
            '1600770037': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                    'map_size_testing_int': map_size_selected,
                    'num_agents_testing': num_agent_selected,

                },
                'label': 'GNN-F-128',
                'color': 'red',
                'dash': '',
                'ls': '-',
                'fc': (0, 0, 1, 0.5),
            },
            '1600770040': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                    'map_size_testing_int': map_size_selected,
                    'num_agents_testing': num_agent_selected,
                },
                'label': 'MAGAT-F-128',
                'color': 'blue',
                'dash': '',
                'ls': '--',
                'fc': (1, 0, 0, 0.5),
            },
            '1601918499': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                    'map_size_testing_int': map_size_selected,
                    'num_agents_testing': num_agent_selected,
                },
                'label': 'MAGAT-F-32-P4',
                'color': 'green',
                'dash': '',
                'ls': ':',
                'fc': (0, 0, 1, 0.5),
            },
            '1601918505': {
                'selection': {
                    'trained_model_epoch': 'b',
                    'action_policy': 'exp_multinorm',
                    'K': 2,
                    'map_size_testing_int': map_size_selected,
                    'num_agents_testing': num_agent_selected,
                },
                'label': 'MAGAT-B-32-P4',
                'color': 'yellow',
                'dash': (3, 1),
                'ls': 'dotted',
                'fc': (0, 0, 1, 0.5),
            },
        }
    # ResultAnalysis.print_analysis_large_set(hist_config)

        ResultAnalysis.print_hist_img(hist_config)
        ResultAnalysis.print_distplot_img(hist_config)






