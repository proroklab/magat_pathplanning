import os
import cv2
import sys
import time
import yaml
import random
import signal
import argparse
import itertools
import subprocess

import numpy as np
import matplotlib.cm as cm
import drawSvg as draw
import scipy.io as sio
from PIL import Image
from multiprocessing import Queue, Pool, Lock, Manager, Process
from os.path import dirname, realpath, pardir

os.system("taskset -p -c 0 %d" % (os.getpid()))
# os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

parser = argparse.ArgumentParser("Input width and #Agent")
parser.add_argument('--map_width', type=int, default=10)
parser.add_argument('--map_density', type=float, default=0.1)
parser.add_argument('--map_complexity', type=float, default=0.01)
parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--num_dataset', type=int, default=30000)
parser.add_argument('--random_map', action='store_true', default=False)
parser.add_argument('--gen_CasePool', action='store_true', default=False)
parser.add_argument('--chosen_solver', type=str, default='ECBS')
parser.add_argument('--num_caseSetup_pEnv', type=int, default=100)
parser.add_argument('--path_loadSourceMap', type=str, default='/local/scratch/ql295/Data/MultiAgentDataset/Solution_DMap/CroppedMap/Set1/Source')
parser.add_argument('--loadmap_DATATYPE', type=str, default='.mat')
parser.add_argument('--loadmap_TYPE', type=str, default='maze')
parser.add_argument('--path_save', type=str, default='/local/scratch/ql295/Data/MultiAgentDataset/Solution_DMap')

args = parser.parse_args()

# set random seed
np.random.seed(1337)



def handler(signum, frame):
    raise Exception("Solution computed by Expert is timeout.")


class CasesGen:
    def __init__(self, config):
        self.config = config

        self.random_map = config.random_map
        print(self.random_map)

        self.path_loadSourceMap = config.path_loadSourceMap
        self.num_agents = config.num_agents
        self.num_data = config.num_dataset
        self.path_save = config.path_save

        self.list_path_loadSourceMap = self.search_Cases(os.path.join(self.path_loadSourceMap,'mapSet'), self.config.loadmap_DATATYPE)
        self.list_path_loadSourceMap = sorted(self.list_path_loadSourceMap)

        self.map_density = config.map_density
        self.label_density = str(config.map_density).split('.')[-1]
        self.map_TYPE = 'map'
        self.size_load_map = (config.map_width, config.map_width)

        self.map_complexity = config.map_complexity
        self.createFolder()

        self.pair_CasesPool = []
        self.PROCESS_NUMBER = 4
        self.timeout = 300
        self.task_queue = Queue()


    def createFolder(self):
        self.dirName_root = os.path.join(self.path_save,'{}{:02d}x{:02d}_density_p{}/{}_Agent/'.format(self.map_TYPE, self.size_load_map[0],
                                                                                                         self.size_load_map[1],
                                                                                                         self.label_density,
                                                                                                         self.num_agents))

        self.dirName_input = os.path.join(self.dirName_root, 'input/')
        self.dirName_mapSet_png = os.path.join(self.dirName_root, 'mapSet_png/')
        self.dirName_mapSet = os.path.join(self.dirName_root, 'mapSet/')
        try:
            # Create target Directory
            os.makedirs(self.dirName_root)

            print("Directory ", self.dirName_root, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass
        try:
            # Create target Directory
            os.makedirs(self.dirName_input)
            os.makedirs(self.dirName_mapSet_png)
            os.makedirs(self.dirName_mapSet)
            print("Directory ", self.dirName_input, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass



    def search_Cases(self, dir, DATA_EXTENSIONS='.yaml'):
        # make a list of file name of input yaml
        list_path = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname,DATA_EXTENSIONS):
                    path = os.path.join(root, fname)
                    list_path.append(path)

        return list_path

    def is_target_file(self, filename, DATA_EXTENSIONS='.yaml'):
        # DATA_EXTENSIONS = ['.yaml']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)



    def img_fill(self, im_in, n):  # n = binary image threshold
        th, im_th = cv2.threshold(im_in, n, 1, cv2.THRESH_BINARY)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (int(w/2), int(h/2)), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # print(im_floodfill_inv)
        # Combine the two images to get the foreground.
        fill_image = im_th | im_floodfill_inv

        return fill_image

    def load_Map(self, id_env):

        filename = self.list_path_loadSourceMap[id_env]

        data_contents = sio.loadmat(filename)

        map_array = data_contents['map']

        return map_array


    def setup_map(self, id_random_env, num_cases_PEnv):
        map_env_raw = self.load_Map(id_random_env)

        map_env = self.img_fill(map_env_raw.astype(np.uint8), 0.5)

        size_originMap = map_env.shape
        Center_OriginMap = int(size_originMap[0]/2), int(size_originMap[1]/2)
        Crop_width = int(self.size_load_map[0]/2)
        Crop_Height = int(self.size_load_map[1] / 2)

        map_CropMap_X = [Center_OriginMap[0]-Crop_width, Center_OriginMap[0] + self.size_load_map[0]-Crop_width]
        map_CropMap_Y = [Center_OriginMap[1]-Crop_Height, Center_OriginMap[1] + self.size_load_map[1]-Crop_Height]


        map_env_crop = map_env[map_CropMap_X[0]:map_CropMap_X[1], map_CropMap_Y[0]:map_CropMap_Y[1]]

        array_freespace = np.argwhere(map_env_crop == 0)
        num_freespace = array_freespace.shape[0]

        array_obstacle = np.transpose(np.nonzero(map_env_crop))
        num_obstacle = array_obstacle.shape[0]


        if num_freespace == 0 or num_obstacle == 0:
            # print(array_freespace)
            map_env_crop = self.setup_map(id_random_env, num_cases_PEnv)


        return map_env_crop

    def setup_cases(self, id_random_env, num_cases_PEnv):
        # Randomly generate certain number of unique cases in same map

        # print(map_env)
        if self.config.loadmap_TYPE == 'free':
            map_env = np.zeros(self.size_load_map, dtype=np.int64)
        else:
            map_env = self.setup_map(id_random_env, num_cases_PEnv)

        self.size_load_map = np.shape(map_env)


        array_freespace = np.argwhere(map_env == 0)
        num_freespace = array_freespace.shape[0]
        array_obstacle = np.transpose(np.nonzero(map_env))
        num_obstacle = array_obstacle.shape[0]

        print(
            "###### Check Map Size: [{},{}]- density: {} - Actual [{},{}] - #Obstacle: {}".format(self.size_load_map[0],
                                                                                                  self.size_load_map[1],
                                                                                                  self.map_density,
                                                                                                  self.size_load_map[0],
                                                                                                  self.size_load_map[1],
                                                                                                  num_obstacle))
        # time.sleep(3)
        list_freespace = []
        list_obstacle = []


        # transfer into list (tuple)
        for id_FS in range(num_freespace):
            list_freespace.append((array_freespace[id_FS, 0], array_freespace[id_FS, 1]))

        for id_Obs in range(num_obstacle):
            list_obstacle.append((array_obstacle[id_Obs, 0], array_obstacle[id_Obs, 1]))

        # print(list_freespace)
        pair_CaseSet_PEnv = []
        pairStore = []
        pair_agent = list(itertools.combinations(range(self.num_agents), 2))

        num_cases_PEnv_exceed = int(5 * num_cases_PEnv)

        for _ in range(num_cases_PEnv_exceed):
            pairset = []
            for id_agents in range(self.num_agents):
                ID_cases_agent = random.sample(list_freespace, 2)
                pairset.append(ID_cases_agent)
            pair_CaseSet_PEnv.append(pairset)

        for pair_CaseSet in pair_CaseSet_PEnv:

            check_condition = []
            for id_pairagent in range(len(pair_agent)):
                firstAgent = pair_agent[id_pairagent][0]
                secondAgent = pair_agent[id_pairagent][1]
                # print("pair", pairset)
                if pair_CaseSet[firstAgent][0] == pair_CaseSet[secondAgent][0] or pair_CaseSet[firstAgent][1] == \
                        pair_CaseSet[secondAgent][1]:
                    print("Remove pair \t", pair_CaseSet)
                    check_condition.append(0)
                else:
                    check_condition.append(1)
                    # pairStore.append(pairset)

            if sum(check_condition) == len(pair_agent):
                pairStore.append(pair_CaseSet)
                # print("Remove cases ID-{}:\t {}".format(id_random_env, pair_CaseSet))

        # todo: generate n-agent pair start-end position - start from single agent CBS
        # todo: non-swap + swap

        for initialCong in pairStore:
            count_repeat = pairStore.count(initialCong)
            if count_repeat > 1:
                id_repeat = pairStore.index(initialCong)
                pairStore.remove(initialCong)
                print('Repeat cases ID {} from ID#{} Map:{}\n'.format(id_repeat, id_random_env, pairStore[id_repeat]))

        CasePool = pairStore[:num_cases_PEnv]


        ### Version 2 ##
        ###  stack cases with same envs into a pool

        random.shuffle(CasePool)
        random.shuffle(CasePool)

        self.save_CasePool(CasePool, id_random_env, list_obstacle)
        self.saveMap(id_random_env,list_obstacle)
        self.saveMap_numpy(id_random_env, map_env)

    def saveMap(self,Id_env,list_obstacle):
        num_obstacle = len(list_obstacle)
        map_data = np.zeros([self.size_load_map[0], self.size_load_map[1]])


        aspect = self.size_load_map[0] / self.size_load_map[1]
        xmin = -0.5
        ymin = -0.5
        xmax = self.size_load_map[0] - 0.5
        ymax = self.size_load_map[1] - 0.5



        d = draw.Drawing(self.size_load_map[0], self.size_load_map[1], origin=(xmin,ymin))
        # d.append(draw.Rectangle(xmin, ymin, self.size_load_map[0], self.size_load_map[1], stroke='black',fill = 'white'))
        # d.append(draw.Rectangle(xmin, ymin, xmax, ymax, stroke_width=0.1, stroke='black', fill='white'))
        d.append(draw.Rectangle(xmin, ymin, self.size_load_map[0], self.size_load_map[1], stroke_width=0.1, stroke='black', fill='white'))

        # d = draw.Drawing(self.size_load_map[0], self.size_load_map[1], origin=(0, 0))
        # d.append(draw.Rectangle(0, 0, self.size_load_map[0], self.size_load_map[1], stroke_width=0, stroke='black', fill='white'))

        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_obstacle[ID_obs][0]
            obstacleIndexY = list_obstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1
            d.append(draw.Rectangle(obstacleIndexY-0.5, obstacleIndexX-0.5, 1, 1, stroke='black', stroke_width=0, fill='black'))
            # d.append(draw.Rectangle(obstacleIndexX, obstacleIndexY, 0.5, 0.5, stroke='black', fill='black'))
            # d.append(draw.Rectangle(obstacleIndexX - 0.5, obstacleIndexY - 0.5, 1, 1, stroke='black', stroke_width=1,
            #                         fill='black'))

        # setup figure
        name_map = os.path.join(self.dirName_mapSet_png, 'Map_ID{:05d}.png'.format(Id_env))

        # d.setPixelScale(2)  # Set number of pixels per geometry unit
        d.setRenderSize(200, 200)  # Alternative to setPixelScale
        d.savePng(name_map)



    def saveMap_numpy(self, Id_env, map_env):
        map_data = {"map":map_env}

        # name_map = os.path.join(self.dirName_mapSet, 'Map_{:02d}x{:02d}_density_p{}_{:05d}.mat'.format(self.size_load_map[0], self.size_load_map[1],
        #                                                                                     self.map_density, Id_env))

        name_map = os.path.join(self.dirName_mapSet,'Map_ID{:05d}.mat'.format(Id_env))
        sio.savemat(name_map, map_data)

    def setup_CasePool(self):

        num_data_exceed = int(self.num_data)

        num_cases_PEnv = self.config.num_caseSetup_pEnv
        num_Env = int(round(num_data_exceed / num_cases_PEnv))

        # print(num_Env)
        for id_random_env in range(num_Env):
            # print(id_random_env)
            self.setup_cases(id_random_env, num_cases_PEnv)

    def get_numEnv(self):
        return len(self.list_path_loadSourceMap)

    def setup_CasePool_(self, id_env):
        filename = self.list_path_loadSourceMap[id_env]
        print(filename)
        map_width = int(filename.split('{}-'.format(self.config.loadmap_TYPE))[-1].split('-')[0])

        self.map_TYPE = self.config.loadmap_TYPE
        self.size_load_map = (map_width, map_width)
        self.label_density = int(
            filename.split('{}-'.format(self.config.loadmap_TYPE))[-1].split('-')[-1].split('.map')[0])
        self.map_density = int(self.label_density)
        self.createFolder()

        num_cases_PEnv = self.config.num_caseSetup_pEnv #int(round(num_data_exceed / num_Env))

        # print(num_Env)

        self.setup_cases(id_env, num_cases_PEnv)

    def setup_CasePool__(self, id_env):
        filename = self.list_path_loadSourceMap[id_env]
        print(filename)
        map_width = int(filename.split('{}-'.format(self.config.loadmap_TYPE))[-1].split('-')[0])

        self.map_TYPE = self.config.loadmap_TYPE
        self.size_load_map = (map_width, map_width)
        self.label_density = int(
            filename.split('{}-'.format(self.config.loadmap_TYPE))[-1].split('-')[-1].split('_ID')[0])
        self.map_density = int(self.label_density)
        self.createFolder()

        num_cases_PEnv = self.config.num_caseSetup_pEnv #int(round(num_data_exceed / num_Env))

        # print(num_Env)

        self.setup_cases(id_env, num_cases_PEnv)

    def save_CasePool(self, pairPool, ID_env, env):
        for id_case in range(len(pairPool)):
            inputfile_name = self.dirName_input + \
                             'input_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.yaml'.format(self.size_load_map[0], self.size_load_map[1],ID_env,
                                                                           id_case)
            self.dump_yaml(self.num_agents, self.size_load_map[0], self.size_load_map[1],
                           pairPool[id_case], env, inputfile_name)

    def dump_yaml(self, num_agent, map_width, map_height, agents, obstacle_list, filename):
        f = open(filename, 'w')
        f.write("map:\n")
        f.write("    dimensions: {}\n".format([map_width, map_height]))
        f.write("    obstacles:\n")
        for id_Obs in range(len(obstacle_list)):
            f.write("    - [{}, {}]\n".format(obstacle_list[id_Obs][0], obstacle_list[id_Obs][1]))
        f.write("agents:\n")
        for n in range(num_agent):
            # f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(n, agents[n][0], agents[n][1]))
            # f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(n, agents[n]['start'], agents[n]['goal']))
            f.write("  - name: agent{}\n    start: [{}, {}]\n    goal: [{}, {}]\n".format(n, agents[n][0][0],
                                                                                          agents[n][0][1],
                                                                                          agents[n][1][0],
                                                                                          agents[n][1][1]))
        f.close()

    def computeSolution(self, chosen_solver):

        self.list_Cases_input = self.search_Cases(self.dirName_input)
        self.list_Cases_input = sorted(self.list_Cases_input)

        self.len_pair = len(self.list_Cases_input)

        self.dirName_output = os.path.join(self.dirName_root,'output_{}/'.format(chosen_solver))

        try:
            # Create target Directory
            os.makedirs(self.dirName_output)
            print("Directory ", self.dirName_output, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass

        for id_case in range(self.len_pair):
            self.task_queue.put(id_case)

        time.sleep(0.3)
        processes = []
        for i in range(self.PROCESS_NUMBER):
            # Run Multiprocesses
            p = Process(target=self.compute_thread, args=(str(i), chosen_solver))

            processes.append(p)
        [x.start() for x in processes]



    def compute_thread(self, thread_id, chosen_solver):
        while True:
            try:
                # print(thread_id)
                id_case = self.task_queue.get(block=False)
                print('thread {} get task:{}'.format(thread_id, id_case))
                self.runExpertSolver(id_case, chosen_solver)
                # print('thread {} finish task:{}'.format(thread_id, id_case))
            except:
                # print('thread {} no task, exit'.format(thread_id))
                return


    def runExpertSolver(self, id_case, chosen_solver):

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        try:
            # load
            name_inputfile = self.list_Cases_input[id_case]
            id_input_map = name_inputfile.split('_IDMap')[-1].split('_IDCase')[0]
            id_input_case = name_inputfile.split('_IDCase')[-1].split('.yaml')[0]
            name_outputfile = self.dirName_output + 'output_map{:02d}x{:02d}_IDMap{}_IDCase{}_{}.yaml'.format(self.size_load_map[0],
                                                                                               self.size_load_map[1],id_input_map,
                                                                                               id_input_case, chosen_solver)
            command_dir = dirname(realpath(__file__))
            # print(command_dir)
            # command_dir = '/local/scratch/ql295/Data/Project/GraphNeural_Planner/onlineExpert'
            # print(name_inputfile)
            # print(name_outputfile)
            if chosen_solver.upper() == "ECBS":
                command_file = os.path.join(command_dir, "ecbs")
                # run ECBS
                subprocess.call(
                    [command_file,
                     "-i", name_inputfile,
                     "-o", name_outputfile,
                     "-w", str(1.1)],
                    cwd=command_dir)
            elif chosen_solver.upper() == "CBS":
                command_file = os.path.join(command_dir, "cbs")
                subprocess.call(
                    [command_file,
                     "-i", name_inputfile,
                     "-o", name_outputfile],
                    cwd=command_dir)
            elif chosen_solver.upper() == "SIPP":
                command_file = os.path.join(command_dir, "mapf_prioritized_sipp")
                subprocess.call(
                    [command_file,
                     "-i", name_inputfile,
                     "-o", name_outputfile],
                    cwd=command_dir)

            log_str = 'map{:02d}x{:02d}_{}Agents_#{}_in_IDMap_#{}'.format(self.size_load_map[0], self.size_load_map[1],
                                                             self.num_agents, id_input_case, id_input_map)
            print('############## Find solution by {} for {} generated  ###############'.format(chosen_solver,log_str))
            with open(name_outputfile) as output_file:
                return yaml.safe_load(output_file)
        except Exception as e:
            print(e)


if __name__ == '__main__':



    path_savedata = '../Solution_DMap'

    # num_dataset = 10 #16**2
    # size_map = (5, 5)




    dataset = CasesGen(args)
    timeout = 300

    if args.random_map:
        path_loadSourceMap = ''
        if args.gen_CasePool:
            dataset.setup_CasePool()
        time.sleep(10)
        dataset.computeSolution(args.chosen_solver)
    else:
        path_loadSourceMap = args.path_loadSourceMap
        num_Env = dataset.get_numEnv()
        # print(num_Env)
        for id_Env in range(num_Env):
            print('\n################## {}  ####################\n'.format(id_Env))
            # dataset.setup_CasePool_(id_Env)
            dataset.setup_CasePool__(id_Env)
            time.sleep(20)
            dataset.computeSolution(args.chosen_solver)






