import csv
import os
import sys
import shutil
import time
import numpy as np
import scipy.io as sio
import yaml
import signal
import argparse
import subprocess

from easydict import EasyDict
from os.path import dirname, realpath, pardir
from hashids import Hashids
import hashlib

sys.path.append(os.path.join(dirname(realpath(__file__)), pardir))

from multiprocessing import Queue, Pool, Lock, Manager, Process
from multiprocessing import Queue, Process

parser = argparse.ArgumentParser("Input width and #Agent")
parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--map_w', type=int, default=10)
parser.add_argument('--map_density', type=float, default=0.1)
parser.add_argument('--loadmap_TYPE', type=str, default='map')
parser.add_argument('--solCases_dir', type=str, default='../MultiAgentDataset/Solution_DMap')
parser.add_argument('--chosen_solver', type=str, default='ECBS')
parser.add_argument('--base_solver', type=str, default='ECBS')

parser.add_argument('--id_start', type=int, default=0)
parser.add_argument('--div_train', type=int, default=21000)
parser.add_argument('--div_valid', type=int, default=200)
parser.add_argument('--div_test', type=int, default=4500)
parser.add_argument('--FOV', type=int, default=9)
parser.add_argument('--guidance', type=str, default='')

args = parser.parse_args()

def handler(signum, frame):
    raise Exception("Solution computed by Expert is timeout.")

class CasesSolver:
    def __init__(self, config):
        self.config = config
        self.PROCESS_NUMBER = 4
        self.timeout = 300
        self.num_agents = self.config.num_agents
        self.size_map = [self.config.map_w, self.config.map_w]
        self.label_density = str(self.config.map_density).split('.')[-1]

        self.zeroTolerance = 1e-9
        self.chosen_solver = config.chosen_solver

        self.hashids = Hashids(alphabet='01234567789abcdef', min_length=5)

        self.label_setup = '{}{:02d}x{:02d}_density_p{}/{}_Agent'.format(self.config.loadmap_TYPE, self.size_map[0],
                                                                         self.size_map[1],
                                                                         self.label_density,
                                                                         self.num_agents)
        self.dirName_parent = os.path.join(self.config.solCases_dir, self.label_setup)

        self.dirName_input = os.path.join(self.dirName_parent, 'input')
        self.dirName_output = os.path.join(self.dirName_parent, 'output_{}'.format(config.chosen_solver))
        self.dirname_base_alg = os.path.join(self.dirName_parent, 'output_{}'.format(config.base_solver))
        self.set_up()

    def set_up(self):
        self.task_queue = Queue()

        self.list_Cases_Sol = self.search_Cases(self.dirname_base_alg)
        self.list_Cases_input = self.search_Cases(self.dirName_input)
        self.list_Cases_input = sorted(self.list_Cases_input)
        self.len_pair = len(self.list_Cases_input)

        self.nameprefix_input = self.list_Cases_input[0].split('input/')[-1].split('ID')[0]

        self.list_Cases_Sol = sorted(self.list_Cases_Sol)
        self.len_Cases_Sol = len(self.list_Cases_Sol)

        print(self.dirName_output)
        try:
            # Create target Directory
            os.makedirs(self.dirName_output)
            print("Directory ", self.dirName_output, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass



    def computeSolution(self):

        div_train = self.config.div_train
        div_valid = self.config.div_valid
        div_test = self.config.div_test


        num_used_data = div_train + div_valid + div_test

        num_data_loop = min(num_used_data, self.len_Cases_Sol)
        # for id_sol in range(num_data_loop):
        for id_sol in range(self.config.id_start, num_data_loop):
            if id_sol < div_train:
                mode = "train"
                case_config = (mode, id_sol)
                self.task_queue.put(case_config)
            elif id_sol < (div_train + div_valid):
                mode = "valid"
                case_config = (mode, id_sol)
                self.task_queue.put(case_config)
            elif id_sol <= num_used_data:
                mode = "test"
                case_config = (mode, id_sol)
                self.task_queue.put(case_config)

        time.sleep(0.3)
        processes = []
        for i in range(self.PROCESS_NUMBER):
            # Run Multiprocesses
            p = Process(target=self.compute_thread, args=(str(i)))

            processes.append(p)

        [x.start() for x in processes]

    def compute_thread(self, thread_id):
        while True:
            try:
                case_config = self.task_queue.get(block=False)
                (mode, id_sol) = case_config
                print('thread {} get task:{} - {}'.format(thread_id, mode, id_sol))
                self.runExpertSolver(id_sol, self.chosen_solver)

            except:
                # print('thread {} no task, exit'.format(thread_id))
                return

    def runExpertSolver(self, id_case, chosen_solver):

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        try:
            # load
            name_solution_file = self.list_Cases_Sol[id_case]

            map_setup = name_solution_file.split('output_')[-1].split('_IDMap')[0]
            id_sol_map = name_solution_file.split('_IDMap')[-1].split('_IDCase')[0]
            id_sol_case = name_solution_file.split('_IDCase')[-1].split('_')[0]

            name_inputfile = os.path.join(self.dirName_input, 'input_{}_IDMap{}_IDCase{}.yaml'.format(map_setup, id_sol_map, id_sol_case))
            name_outputfile = os.path.join(self.dirName_output, 'output_{}_IDMap{}_IDCase{}.yaml'.format(map_setup, id_sol_map, id_sol_case))

            command_dir = dirname(realpath(__file__))
            # print(command_dir)

            print(name_inputfile)
            print(name_outputfile)
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

            log_str = 'map{:02d}x{:02d}_{}Agents_#{}_in_IDMap_#{}'.format(self.size_map[0], self.size_map[1],
                                                             self.num_agents, id_sol_case, id_sol_map)
            print('############## Find solution by {} for {} generated  ###############'.format(chosen_solver,log_str))
            with open(name_outputfile) as output_file:
                return yaml.safe_load(output_file)
        except Exception as e:
            print(e)



    def search_Cases(self, dir):
        # make a list of file name of input yaml
        list_path = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    path = os.path.join(root, fname)
                    list_path.append(path)

        return list_path

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.yaml']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)




if __name__ == '__main__':
    Cases_Solver = CasesSolver(args)
    time.sleep(5)
    Cases_Solver.computeSolution()


