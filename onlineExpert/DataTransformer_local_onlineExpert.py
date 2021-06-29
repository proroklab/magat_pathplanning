
import csv
import os
import sys
import shutil
import time
import numpy as np
import scipy.io as sio
import yaml

from easydict import EasyDict
from os.path import dirname, realpath, pardir
from hashids import Hashids
import hashlib
sys.path.append(os.path.join(dirname(realpath(__file__)), pardir))

import utils.graphUtils.graphTools as graph
# from utils.graphUtils.graphTools import isConnected

# from dataloader.statetransformer import AgentState
# from dataloader.statetransformer_localGuidance import AgentState
# from dataloader.statetransformer_localGuidance_SDObs import AgentState
# from dataloader.statetransformer_localGuidance_SemiLocal import AgentState
# from dataloader.statetransformer_globalGuidance import AgentState

from dataloader.statetransformer_Guidance import AgentState

from scipy.spatial.distance import squareform, pdist
from multiprocessing import Queue, Process


class DataTransformer:
    def __init__(self, config):
        self.config = config
        self.PROCESS_NUMBER = 4
        self.num_agents = self.config.num_agents

        self.size_map = [self.config.map_w, self.config.map_h]
        self.AgentState = AgentState(self.config)
        # self.communicationRadius = 5 # communicationRadius
        # self.communicationRadius = 7 # communicationRadius
        self.communicationRadius = self.config.commR # communicationRadius

        self.zeroTolerance = 1e-9
        self.delta = [[-1, 0],  # go up
                 [0, -1],  # go left
                 [1, 0],  # go down
                 [0, 1],  # go right
                 [0, 0]]  # stop
        self.num_actions = 5
        self.root_path_save = self.config.failCases_dir
        self.list_seqtrain_file = []
        self.list_train_file = []
        self.pathtransformer = self.pathtransformer_RelativeCoordinate
        if self.config.dynamic_commR:
            # comm radius that ensure initial graph connected
            print("run on multirobotsim (radius dynamic) with collision shielding")
            self.getAdjacencyMatrix = self.computeAdjacencyMatrix
        else:
            # comm radius fixed
            print("run on multirobotsim (radius fixed) with collision shielding")
            self.getAdjacencyMatrix = self.computeAdjacencyMatrix_fixedCommRadius

    def set_up(self, epoch):
        self.dir_input = os.path.join(self.config.failCases_dir, "input/")
        self.dir_sol = os.path.join(self.config.failCases_dir, "output_ECBS/")
        self.list_failureCases_solution = self.search_failureCases(self.dir_sol)
        self.list_failureCases_input = self.search_failureCases(self.dir_input)
        self.nameprefix_input = self.list_failureCases_input[0].split('input/')[-1].split('ID')[0]
        self.list_failureCases_solution = sorted(self.list_failureCases_solution)
        self.len_failureCases_solution = len(self.list_failureCases_solution)
        self.current_epoch = epoch
        self.task_queue = Queue()

        self.path_save_solDATA = os.path.join(self.root_path_save, "Cache_data", "Epoch_{}".format(epoch))
        try:
            # Create target Directory
            os.makedirs(self.path_save_solDATA)
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass


    def solutionTransformer(self):



        for id_sol in range(self.len_failureCases_solution):
        # for id_sol in range(21000):
            self.task_queue.put(id_sol)

        time.sleep(0.3)
        processes = []
        for i in range(self.PROCESS_NUMBER):
            # Run Multiprocesses
            p = Process(target=self.compute_thread, args=(str(i)))

            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]


    def compute_thread(self,thread_id):
        while True:
            try:
                id_sol = self.task_queue.get(block=False)
                print('thread {} get task:{}'.format(thread_id, id_sol))
                self.pipeline(id_sol)

            except:
                # print('thread {} no task, exit'.format(thread_id))
                return

    def pipeline(self,id_sol):
        agents_schedule, agents_goal, makespan, map_data, id_case = self.load_ExpertSolution(id_sol)
        log_str = 'Transform_failureCases_ID_#{:05d} from ID_MAP {:05d} in Epoch{}'.format(id_case[2],id_case[1],id_case[0])
        print('############## {} ###############'.format(log_str))
        self.pathtransformer(map_data, agents_schedule, agents_goal, makespan + 1, id_case)
        

    def load_ExpertSolution(self, ID_case):

        name_solution_file = self.list_failureCases_solution[ID_case]

        map_setup = name_solution_file.split('output_')[-1].split('_IDMap')[0]
        id_sol_map = name_solution_file.split('_IDMap')[-1].split('_IDCase')[0]
        id_sol_case = name_solution_file.split('_IDCase')[-1].split('_')[0]


        name_inputfile = os.path.join(self.dir_input,
                                      'input_{}_IDMap{}_IDCase{}.yaml'.format(map_setup, id_sol_map, id_sol_case))


        with open(name_inputfile, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(name_solution_file, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        agentsConfig = data_config['agents']
        num_agent = len(agentsConfig)
        list_posObstacle = data_config['map']['obstacles']

        if list_posObstacle == None:
            map_data = np.zeros(self.size_map, dtype=np.int64)
        else:
            map_data = self.setup_map(list_posObstacle)
        schedule = data_output['schedule']
        makespan = data_output['statistics']['makespan']

        # print(data_config)
        # print(data_output)
        goal_allagents = np.zeros([num_agent, 2])
        schedule_agentsState = np.zeros([makespan + 1, num_agent, 2])
        schedule_agentsActions = np.zeros([makespan + 1, num_agent, self.num_actions])
        schedule_agents = [schedule_agentsState, schedule_agentsActions]
        hash_ids = np.zeros(self.num_agents)
        for id_agent in range(num_agent):
            goalX = agentsConfig[id_agent]['goal'][0]
            goalY = agentsConfig[id_agent]['goal'][1]
            goal_allagents[id_agent][:] = [goalX, goalY]

            schedule_agents = self.obtainSchedule(id_agent, schedule, schedule_agents, goal_allagents, makespan + 1)

            str_id = '{}_{}_{}'.format(self.current_epoch,id_sol_case,id_agent)
            int_id = int(hashlib.sha256(str_id.encode('utf-8')).hexdigest(), 16) % (10 ** 5)
            # hash_ids[id_agent]=np.divide(int_id,10**5)
            hash_ids[id_agent] = int_id

        # print(id_sol_map, id_sol_case, hash_ids)
        return schedule_agents, goal_allagents, makespan, map_data, (self.current_epoch, int(id_sol_map), int(id_sol_case), hash_ids)

    def obtainSchedule(self, id_agent, agentplan, schedule_agents, goal_allagents, teamMakeSpan):

        name_agent = "agent{}".format(id_agent)
        [schedule_agentsState, schedule_agentsActions] = schedule_agents
        
        planCurrentAgent = agentplan[name_agent]
        pathLengthCurrentAgent = len(planCurrentAgent)

        actionKeyListAgent = []

        for step in range(teamMakeSpan):
            if step < pathLengthCurrentAgent:
                currentX = planCurrentAgent[step]['x']
                currentY = planCurrentAgent[step]['y']
            else:
                currentX = goal_allagents[id_agent][0]
                currentY = goal_allagents[id_agent][1]
                
            schedule_agentsState[step][id_agent][:] = [currentX, currentY]
            # up left down right stop
            actionVectorTarget = [0, 0, 0, 0, 0]

            # map action with respect to the change of position of agent
            if step < (pathLengthCurrentAgent - 1):
                nextX = planCurrentAgent[step + 1]['x']
                nextY = planCurrentAgent[step + 1]['y']
                # actionCurrent = [nextX - currentX, nextY - currentY]

            elif step >= (pathLengthCurrentAgent - 1):
                nextX = goal_allagents[id_agent][0]
                nextY = goal_allagents[id_agent][1]

            actionCurrent = [nextX - currentX, nextY - currentY]


            actionKeyIndex = self.delta.index(actionCurrent)
            actionKeyListAgent.append(actionKeyIndex)

            actionVectorTarget[actionKeyIndex] = 1
            schedule_agentsActions[step][id_agent][:] = actionVectorTarget


        return [schedule_agentsState,schedule_agentsActions]

    def setup_map(self, list_posObstacle):
        num_obstacle = len(list_posObstacle)
        map_data = np.zeros(self.size_map)
        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_posObstacle[ID_obs][0]
            obstacleIndexY = list_posObstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1

        return map_data



    def pathtransformer_RelativeCoordinate(self, map_data, agents_schedule, agents_goal, makespan, ID_case):
        # input: start and goal position,
        # output: a set of file,
        #         each file consist of state (map. goal, state) and target (action for current state)


        mode = 'train'
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}

        (current_epoch,id_sol_map, id_sol_case, hash_ids) = ID_case
        # compute AdjacencyMatrix
        GSO, communicationRadius = self.getAdjacencyMatrix(schedule_agentsState, self.communicationRadius)

        # transform into relative Coordinate, loop "makespan" times
        # print(map_data)
        # print(agents_goal, schedule_agentsState, makespan)
        self.AgentState.setmap(map_data)
        input_seq_tensor = self.AgentState.toSeqInputTensor(agents_goal, schedule_agentsState, makespan)
        # print(input_seq_tensor)

        list_input = input_seq_tensor.cpu().detach().numpy()
        save_PairredData.update({'map': map_data, 'goal': agents_goal, 'inputState': schedule_agentsState,
                                 'inputTensor': list_input, 'target': schedule_agentsActions,
                                 'GSO': GSO,'makespan':makespan, 'HashIDs':hash_ids, 'ID_Map':int(id_sol_map), 'ID_case':int(id_sol_case)})


        self.save(mode, save_PairredData, ID_case, makespan)
        print("Save as Relative Coordination - {}set_#{} at ID_Map{} from Epoch {}.".format(mode, id_sol_case, id_sol_map, current_epoch))

    def save(self, mode, save_PairredData, ID_case, makespan):
        (current_epoch, id_sol_map, id_sol_case, hash_ids) = ID_case
        file_name = os.path.join(self.path_save_solDATA, '{}_IDMap{:05d}_IDCase{:05d}_MP{}.mat'.format(mode, int(id_sol_map), int(id_sol_case), makespan))
        # print(file_name)
        sio.savemat(file_name, save_PairredData)


    def search_failureCases(self, dir):
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

    def computeAdjacencyMatrix(self, pos, CommunicationRadius, connected=True):

        # First, transpose the axis of pos so that the rest of the code follows
        # through as legible as possible (i.e. convert the last two dimensions
        # from 2 x nNodes to nNodes x 2)
        # pos: TimeSteps x nAgents x 2 (X, Y)

        # Get the appropriate dimensions
        nSamples = pos.shape[0]
        len_TimeSteps = pos.shape[0]  # length of timesteps
        nNodes = pos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices
        W = np.zeros([len_TimeSteps, nNodes, nNodes])
        threshold = CommunicationRadius  # We compute a different
        # threshold for each sample, because otherwise one bad trajectory
        # will ruin all the adjacency matrices

        for t in range(len_TimeSteps):
            # Compute the distances
            distances = squareform(pdist(pos[t]))  # nNodes x nNodes
            # Threshold them
            W[t] = (distances < threshold).astype(pos.dtype)
            # And get rid of the self-loops
            W[t] = W[t] - np.diag(np.diag(W[t]))
            # Now, check if it is connected, if not, let's make the
            # threshold bigger
            while (not graph.isConnected(W[t])) and (connected):
                # while (not graph.isConnected(W[t])) and (connected):
                # Increase threshold
                threshold = threshold * 1.1  # Increase 10%
                # Compute adjacency matrix
                W[t] = (distances < threshold).astype(pos.dtype)
                W[t] = W[t] - np.diag(np.diag(W[t]))

        # And since the threshold has probably changed, and we want the same
        # threshold for all nodes, we repeat:
        W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
        for t in range(len_TimeSteps):
            # Initial matrix
            allagentPos = pos[t]
            distances = squareform(pdist(allagentPos, 'euclidean'))  # nNodes x nNodes

            W_t = (distances < threshold).astype(allagentPos.dtype)
            W_t = W_t - np.diag(np.diag(W_t))

            if np.any(W):
                # if W is all non-zero matrix, do normalization
                if self.config.symmetric_norm:
                    deg = np.sum(W_t, axis=0)  # nNodes (degree vector)
                    zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)
                    deg[zeroDeg] = 1.
                    invSqrtDeg = np.sqrt(1. / deg)
                    invSqrtDeg[zeroDeg] = 0.
                    Deg = np.diag(invSqrtDeg)
                    W_t = Deg @ W_t @ Deg

                maxEigenValue = self.get_maxEigenValue(W_t)
                W_norm[t] = W_t/maxEigenValue
            else:
                # if W is all zero matrix, don't do any normalization
                W_norm[t] = W

        return W_norm, threshold

    def get_maxEigenValue(self, matrix):

        isSymmetric = np.allclose(matrix, np.transpose(matrix, axes=[1, 0]))
        if isSymmetric:
            W = np.linalg.eigvalsh(matrix)
        else:
            W = np.linalg.eigvals(matrix)

        maxEigenvalue = np.max(np.real(W), axis=0)
        return maxEigenvalue
        # return np.max(np.abs(np.linalg.eig(matrix)[0]))

    def computeAdjacencyMatrix_fixedCommRadius(self, pos, CommunicationRadius, connected=True):
        len_TimeSteps = pos.shape[0]  # length of timesteps
        nNodes = pos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices

        W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
        for t in range(len_TimeSteps):
            # Initial matrix
            allagentPos = pos[t]
            distances = squareform(pdist(allagentPos, 'euclidean'))  # nNodes x nNodes

            W = (distances < CommunicationRadius).astype(allagentPos.dtype)
            W = W - np.diag(np.diag(W))

            if np.any(W):
                # if W is all non-zero matrix, do normalization
                if self.config.symmetric_norm:
                    deg = np.sum(W, axis=0)  # nNodes (degree vector)
                    zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)
                    deg[zeroDeg] = 1.
                    invSqrtDeg = np.sqrt(1. / deg)
                    invSqrtDeg[zeroDeg] = 0.
                    Deg = np.diag(invSqrtDeg)
                    W = Deg @ W @ Deg

                maxEigenValue = self.get_maxEigenValue(W)
                W_norm[t] = W/maxEigenValue
            else:
                # if W is all zero matrix, don't do any normalization
                W_norm[t] = W
        return W_norm, CommunicationRadius

    def pathtransformer_GlobalCoordinate(self, map_data, agents_schedule, agents_goal, makespan, ID_case):
        # input: start and goal position,
        # output: a set of file,
        #         each file consist of state (map. goal, state) and target (action for current state)

        mode = 'train'
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}
        save_PairredData.update({'map': map_data, 'goal': agents_goal,
                                 'inputState': schedule_agentsState,
                                 'target': schedule_agentsActions,
                                 'makespan': makespan})

        self.save(mode, save_PairredData, ID_case)
        # print("Save as Global Coordination - {}set_#{}.".format(mode, ID_case))

# if __name__ == '__main__':

    # config = {'num_agents': 12,
    #           'map_w': 20,
    #           'map_h': 20,
    #           'failCases_dir': '/local/scratch/ql295/Data/MultiAgentDataset/test',
    #           'exp_net': 'dcp'
    #           }
    # config = {'num_agents': 12,
    #           'map_w': 20,
    #           'map_h': 20,
    #           'failCases_dir': '/local/scratch/ql295/Data/MultiAgentDataset/experiments/dcpOEGAT_map20x20_rho1_10Agent/K3_HS0/1591839220/failure_cases/save',
    #           'exp_net': 'dcp',
    #           'FOV':9,
    #           'guidance': 'Project_G'
    #           }
    # config = {'num_agents': 20,
    #           'map_w': 28,
    #           'map_h': 28,
    #           'failCases_dir': '/local/scratch/ql295/Data/MultiAgentDataset/experiments/dcpOEGAT_map20x20_rho1_10Agent/K3_HS0/1591839220/failure_cases',
    #           'exp_net': 'dcp',
    #           'FOV':9,
    #           'guidance': 'Project_G'
    #           }
    # config = {'num_agents': 10,
    #           'map_w': 20,
    #           'map_h': 20,
    #           'failCases_dir': '/local/scratch/ql295/Data/Project_testbed/Quick_Test',
    #           'exp_net': 'dcp',
    #           'FOV':9,
    #           'guidance': 'Project_G',
    #           'commR': 7,
    #           'dynamic_commR':False,
    #           'symmetric_norm': False,
    #           }
    # config_setup = EasyDict(config)
    # DataTransformer = DataTransformer(config_setup)
    # DataTransformer.set_up('1')
    # DataTransformer.solutionTransformer()
