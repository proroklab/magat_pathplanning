import numpy as np
import torch
from offlineExpert.a_star import PathPlanner


class AgentState:
    def __init__(self, config):
        # self.config = config
        # self.num_agents = self.config.num_agents

        self.config = config
        self.num_agents = self.config.num_agents
        # self.FOV = 5
        self.FOV = self.config.FOV
        self.FOV_width = int(self.FOV/2)
        self.border = 1
        self.W = self.FOV + 2
        self.H = self.FOV + 2

        self.dist = int(np.floor(self.W/2))
        self.border_down = 0
        self.border_left = 0

        self.centerX = self.dist #+ 1
        self.centerY = self.dist #+ 1

        self.map_pad = None
        self.Path_Planner = PathPlanner(False)

        # a fuction to merge all different guidance
        self.mode_Guidance = self.config.guidance.split('_')[0]
        self.mode_Obstacle = self.config.guidance.split('_')[1]
        if self.mode_Guidance == 'LocalG':
            # print('####################  use local G')
            self.agentStateToTensor = self.agentStateToTensor_LocalG

        elif self.mode_Guidance == 'SemiLG':
            # print('####################  use SemiLG G')
            self.agentStateToTensor = self.agentStateToTensor_SemiLG

        elif self.mode_Guidance == 'GlobalG':
            # print('####################  use GlobalG G')
            self.agentStateToTensor = self.agentStateToTensor_globalG

        if self.config.guidance == 'Project_G':
            # print('####################  use Project_G G')
            # print("hello project")
            self.agentStateToTensor = self.agentStateToTensor_projectG


        # self.consider_DObs = True
        if self.mode_Obstacle =='S':
            self.consider_DObs = False
        elif  self.mode_Obstacle =='SD':
            self.consider_DObs = True



    def pad_with(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def setmap(self, map_channel):
        self.map_global = np.array(map_channel,dtype=np.int64)
        self.map_pad = np.pad(map_channel, self.FOV_width, self.pad_with, padder=1).astype(np.int64)
        # print(self.map_pad.shape)

        if self.mode_Guidance == 'Project_G':
            self.max_localPath = 2
        elif self.mode_Guidance == 'LocalG':
            self.max_localPath = self.W * 4

        else:
            self.max_localPath = self.map_pad.shape[0] + self.map_pad.shape[1]

        if self.mode_Guidance == 'SemiLG':
            # store map know the border
            # self.map_global_empty = np.zeros_like(self.map_pad,dtype=np.int64)
            # self.map_global_empty_pad = np.pad(self.map_global_empty, self.FOV_width, self.pad_with, padder=1).astype(np.int64)
            # self.store_map_agentView = np.tile(self.map_global_empty_pad,[self.num_agents,1,1]).astype(np.int64)

            # store map don't know border
            self.store_map_agentView = np.tile(np.zeros_like(self.map_pad),[self.num_agents,1,1]).astype(np.int64)


    def setPosAgents(self, state_allagents):
        # the second channel represent position of local agent and agents within FOV
        # channel_allstate = np.zeros([self.W, self.H], dtype=np.int64)
        channel_allstate = np.zeros_like(self.map_global, dtype=np.int64)

        for id_agent in range(self.num_agents):
            currentX = int(state_allagents[id_agent][0])
            currentY = int(state_allagents[id_agent][1])
            channel_allstate[currentX][currentY] = 1

        channel_allstate_pad = np.pad(channel_allstate, self.FOV_width, self.pad_with, padder=0)

        return channel_allstate_pad

    def projectedgoal(self, goal_FOV, state_agent, goal_agent):
        channel_goal = np.pad(goal_FOV, self.border, self.pad_with, padder=0)

        dy = float(goal_agent[1]-state_agent[1])
        dx = float(goal_agent[0]-state_agent[0])
        y_sign = np.sign(dy)
        x_sign = np.sign(dx)

        # angle between position of agent an goal
        angle = np.arctan2(dy,dx)

        if (angle >= np.pi / 4 and angle <= np.pi * 3 / 4) or (angle >= -np.pi * (3 / 4) and angle <= -np.pi / 4):
            goalY_FOV = int(self.dist * (y_sign + 1))
            goalX_FOV = int(self.centerX + np.round(self.dist * dx / np.abs(dy)))
        else:
            goalX_FOV = int(self.dist * (x_sign + 1))
            goalY_FOV = int(self.centerX + np.round(self.dist * dy / np.abs(dx)))

        channel_goal[goalX_FOV][goalY_FOV] = 1
        return channel_goal

    def stackinfo_(self, goal_allagents, state_allagents):
        input_step = np.stack((goal_allagents, state_allagents),axis=1)

        input_tensor = torch.FloatTensor(input_step)
        return input_tensor

    def stackinfo(self, goal_allagents, state_allagents):

        input_tensor = np.stack((goal_allagents, state_allagents))

        input_tensor = torch.FloatTensor(input_tensor)

        return input_tensor

    def toInputTensor(self, goal_allagents, state_allagents):
        # print("run on semilocal")
        channel_allstate_pad = self.setPosAgents(state_allagents)
        self.record_localPath = np.zeros([self.num_agents, self.max_localPath, 2])


        input_step = []
        select_agent = 5
        for id_agent in range(self.num_agents): #range(3,6):#
            load_state = (goal_allagents, state_allagents, channel_allstate_pad, id_agent)

            mode = (id_agent == select_agent)
            # mode = False
            input_step_currentAgent, record_localPath = self.agentStateToTensor(load_state, mode)

            input_step.append(input_step_currentAgent)
            self.record_localPath[id_agent, :, :] = record_localPath

            # if mode:
            # print("\n", self.record_localPath[id_agent, :, :])

        input_tensor = torch.FloatTensor(input_step)
        return input_tensor

    def get_localPath(self):

        return self.record_localPath

    def toSeqInputTensor(self, goal_allagents, state_AgentsSeq, makespan):
        list_input = []
        select_agent = 2
        for step in range(makespan):
            state_allagents = state_AgentsSeq[step][:]
            channel_allstate_pad = self.setPosAgents(state_allagents)

            input_step = []
            for id_agent in range(self.num_agents): #range(3,6):#
                load_state = (goal_allagents, state_allagents, channel_allstate_pad, id_agent)

                mode = (id_agent == select_agent)
                # mode = False

                input_step_currentAgent, record_localPath = self.agentStateToTensor(load_state, mode)
                input_step.append(input_step_currentAgent)
            list_input.append(input_step)

        input_tensor = torch.FloatTensor(list_input)
        return input_tensor

    def agentStateToTensor_projectG(self, load_state, mode):
        goal_allagents, state_allagents, channel_allstate_pad, id_agent = load_state
        input_step_currentAgent = []

        currentX_global = int(state_allagents[id_agent][0])
        currentY_global = int(state_allagents[id_agent][1])

        currentPos = np.array([[currentX_global, currentY_global]])
        record_localPath = np.repeat(currentPos, [self.max_localPath], axis=0)

        goalX_global = int(goal_allagents[id_agent][0])
        goalY_global = int(goal_allagents[id_agent][1])

        # check position
        FOV_X = [currentX_global, currentX_global + 2 * self.FOV_width + 1]
        FOV_Y = [currentY_global, currentY_global + 2 * self.FOV_width + 1]

        channel_map_FOV = self.map_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        channel_map = np.pad(channel_map_FOV, self.border, self.pad_with, padder=0)

        channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0)

        channel_goal_global = np.zeros_like(self.map_global, dtype=np.int64)
        channel_goal_global[goalX_global][goalY_global] = 1
        channel_goal_pad = np.pad(channel_goal_global, self.FOV_width, self.pad_with, padder=0)
        channel_goal_FOV = channel_goal_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]

        if (channel_goal_FOV > 0).any():
            channel_goal = np.pad(channel_goal_FOV, self.border, self.pad_with, padder=0)
        else:
            channel_goal = self.projectedgoal(channel_goal_FOV, [currentX_global, currentY_global],
                                              [goalX_global, goalY_global])

        goalX_FOV, goalY_FOV = np.nonzero(channel_goal)

        if goalX_FOV.shape[0] == 1:

            for id_path in range(goalX_FOV.shape[0]):
                posX_goalFOV = goalX_FOV[0]
                posY_goalFOV = goalY_FOV[0]

                record_localPath[id_path][0] = posX_goalFOV + (currentX_global - self.FOV_width) - 1
                record_localPath[id_path][1] = posY_goalFOV + (currentY_global - self.FOV_width) - 1

        # if mode:
        # print("\n-----------{} - Goal -----------\n{}".format(id_agent, channel_goal_globalmap_pad))
        # print("\n", channel_goal)
        # print("\n", goalX_global, goalY_global)

        input_step_currentAgent.append(channel_map)
        input_step_currentAgent.append(channel_goal)
        input_step_currentAgent.append(channel_state)

        return input_step_currentAgent, record_localPath

    def agentStateToTensor_LocalG(self, load_state, mode):
        goal_allagents, state_allagents, channel_allstate_pad, id_agent = load_state
        input_step_currentAgent = []

        currentX_global = int(state_allagents[id_agent][0])
        currentY_global = int(state_allagents[id_agent][1])

        currentPos = np.array([[currentX_global, currentY_global]])
        record_localPath = np.repeat(currentPos, [self.max_localPath], axis=0)

        goalX_global = int(goal_allagents[id_agent][0])
        goalY_global = int(goal_allagents[id_agent][1])

        # check position
        FOV_X = [currentX_global, currentX_global + 2 * self.FOV_width + 1]
        FOV_Y = [currentY_global, currentY_global + 2 * self.FOV_width + 1]


        channel_map_FOV = self.map_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        channel_map = np.pad(channel_map_FOV, self.border, self.pad_with, padder=0)

        if self.consider_DObs:
            channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
            channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0)
        else:
            channel_state = np.zeros_like(channel_map, dtype=np.int64)


        channel_goal_global = np.zeros_like(self.map_global, dtype=np.int64)
        channel_goal_global[goalX_global][goalY_global] = 1
        channel_goal_pad = np.pad(channel_goal_global, self.FOV_width, self.pad_with, padder=0)
        channel_goal_FOV = channel_goal_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]

        if (channel_goal_FOV > 0).any():
            channel_goal = np.pad(channel_goal_FOV, self.border, self.pad_with, padder=0)
        else:
            channel_goal = self.projectedgoal(channel_goal_FOV, [currentX_global, currentY_global],
                                              [goalX_global, goalY_global])

        goalX_FOV, goalY_FOV = np.nonzero(channel_goal)

        if goalX_FOV.shape[0] == 1:

            channel_combine = np.add(channel_map, channel_state)
            channel_combine[self.centerX][self.centerY] = 0

            check_collidedWithGoal = np.add(channel_state, channel_goal)
            collided_goalX_FOV, collided_goalY_FOV = np.where(check_collidedWithGoal == 2)

            if collided_goalX_FOV.shape[0] >= 1:
                channel_combine[int(collided_goalX_FOV[0])][int(collided_goalY_FOV[0])] = 0

            full_path, full_path_length = self.Path_Planner.a_star(channel_combine,
                                                                   [self.centerX, self.centerY],
                                                                   [goalX_FOV[0], goalY_FOV[0]])

            for id_path in range(full_path_length):
                posX_goalFOV = full_path[id_path][0]
                posY_goalFOV = full_path[id_path][1]
                channel_goal[posX_goalFOV][posY_goalFOV] = 1
                record_localPath[id_path][0] = posX_goalFOV + (currentX_global - self.FOV_width) - 1
                record_localPath[id_path][1] = posY_goalFOV + (currentY_global - self.FOV_width) - 1


        # if mode:
        # print("\n-----------{} - Goal -----------\n{}".format(id_agent, channel_goal_globalmap_pad))
        # print("\n", channel_goal)
        # print("\n", goalX_global, goalY_global)

        input_step_currentAgent.append(channel_map)
        input_step_currentAgent.append(channel_goal)
        input_step_currentAgent.append(channel_state)

        return input_step_currentAgent, record_localPath

    def agentStateToTensor_SemiLG(self, load_state, mode):
        goal_allagents, state_allagents, channel_allstate_pad, id_agent = load_state

        input_step_currentAgent = []

        currentX_global = int(state_allagents[id_agent][0])
        currentY_global = int(state_allagents[id_agent][1])

        currentPos = np.array([[currentX_global,currentY_global]])
        record_localPath = np.repeat(currentPos, [self.max_localPath], axis=0)

        goalX_global = int(goal_allagents[id_agent][0])
        goalY_global = int(goal_allagents[id_agent][1])

        # check position
        FOV_X = [currentX_global, currentX_global + 2*self.FOV_width + 1]
        FOV_Y = [currentY_global, currentY_global + 2*self.FOV_width + 1]

        # crop the state within FOV from global map
        channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
        # (input tensor-ready) pad the state within FOV with border 1
        channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0,  dtype=np.int64)

        # array to store robot state (within FOV) and goal
        channel_state_globalmap = np.zeros_like(self.map_pad,  dtype=np.int64)
        channel_goal_globalmap = np.zeros_like(self.map_pad, dtype=np.int64)
        channel_goal_globalmap_pad =  np.pad(channel_goal_globalmap, self.border, self.pad_with, padder=0,  dtype=np.int64)

        FOV_X = [currentX_global, currentX_global + 2 * self.FOV_width + 1]
        FOV_Y = [currentY_global, currentY_global + 2 * self.FOV_width + 1]
        channel_state_globalmap[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]] = channel_state_FOV

        # the map that current robot observe
        # if mode:
        #     print(channel_fullmap_FOV)
        channel_fullmap_FOV = self.map_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        channel_map = np.pad(channel_fullmap_FOV, self.border, self.pad_with, padder=0,  dtype=np.int64)

        # use to update the data of the map in the memory of current robot

        self.store_map_agentView[id_agent,FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]] = channel_fullmap_FOV
        channel_agentmap = self.store_map_agentView[id_agent,:,:]

        channel_combine = np.add(channel_agentmap, channel_state_globalmap)
        channel_combine_pad = np.pad(channel_combine, self.border, self.pad_with, padder=0,  dtype=np.int64)


        currentX_global_pad = currentX_global + self.FOV_width  + self.border
        currentY_global_pad = currentY_global + self.FOV_width  + self.border
        goalX_global_pad = goalX_global + self.FOV_width  + self.border
        goalY_global_pad = goalY_global + self.FOV_width  + self.border

        FOV_X_pad = [currentX_global, currentX_global + 2 * self.FOV_width + 1 + 2 * self.border]
        FOV_Y_pad = [currentY_global, currentY_global + 2 * self.FOV_width + 1 + 2 * self.border]

        # check if the agents standing at current agent's goal
        if channel_combine_pad[goalX_global_pad][goalY_global_pad] == 1:

            channel_combine_pad[goalX_global_pad][goalY_global_pad] = 0

        # if mode:
        # print("-------------------------\n------------ {} ------------\n------------------------\n".format(id_agent))
        # print("------------{} World Map -----------\n{}".format(id_agent,self.map_pad))
        # print("------------ Agent{} storedMap-----------\n{} \n".format(id_agent,self.store_map_agentView[id_agent, :, :]))
        # print("------------{} A* Map -----------\n{}\n".format(id_agent,channel_combine_pad))

        full_path, full_path_length = self.Path_Planner.a_star(channel_combine_pad,
                                                                   [currentX_global_pad, currentY_global_pad],
                                                                   [goalX_global_pad, goalY_global_pad])


        # print(full_path)
        for id_path in range(full_path_length):

            posX_goalFOV = full_path[id_path][0]
            posY_goalFOV = full_path[id_path][1]
            channel_goal_globalmap_pad[posX_goalFOV][posY_goalFOV] = 1

            record_localPath[id_path][0] = posX_goalFOV - 1 - self.FOV_width
            record_localPath[id_path][1] = posY_goalFOV - 1 - self.FOV_width


        channel_goal = np.zeros_like(channel_map,  dtype=np.int64)
        # print(channel_goal_globalmap_pad.shape, FOV_X_pad[0], FOV_X_pad[1], FOV_Y_pad[0],FOV_Y_pad[1])
        channel_goal[:, :] = channel_goal_globalmap_pad[FOV_X_pad[0]:FOV_X_pad[1], FOV_Y_pad[0]:FOV_Y_pad[1]]

        # if mode:
        # print("\n-----------{} - Goal -----------\n{}".format(id_agent, channel_goal_globalmap_pad))
        # print("\n", channel_goal)
        # print("\n", goalX_global, goalY_global)

        input_step_currentAgent.append(channel_map.astype(np.float64))
        input_step_currentAgent.append(channel_goal.astype(np.float64))
        input_step_currentAgent.append(channel_state.astype(np.float64))

        return input_step_currentAgent, record_localPath

    def agentStateToTensor_globalG(self, load_state, mode):

        goal_allagents, state_allagents, channel_allstate_pad, id_agent = load_state
        input_step_currentAgent = []

        currentX_global = int(state_allagents[id_agent][0])
        currentY_global = int(state_allagents[id_agent][1])

        currentPos = np.array([[currentX_global, currentY_global]], dtype=np.int64)
        record_localPath = np.repeat(currentPos, [self.max_localPath], axis=0)

        goalX_global = int(goal_allagents[id_agent][0])
        goalY_global = int(goal_allagents[id_agent][1])

        # define the region of FOV
        FOV_X = [currentX_global, currentX_global + 2 * self.FOV_width + 1]
        FOV_Y = [currentY_global, currentY_global + 2 * self.FOV_width + 1]

        # crop the state within FOV from global map
        channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        # (input tensor-ready) pad the state within FOV with border 1
        channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0, dtype=np.int64)

        # crop the map within FOV from global map
        # if mode:
        #     print(channel_fullmap_FOV)
        channel_fullmap_FOV = self.map_pad[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]]
        # (input map-ready) pad the map within FOV with border 1
        channel_map = np.pad(channel_fullmap_FOV, self.border, self.pad_with, padder=0, dtype=np.int64)

        # array to store robot state (within FOV) and goal into map size array
        channel_state_globalmap = np.zeros_like(self.map_pad, dtype=np.int64)
        channel_goal_globalmap = np.zeros_like(self.map_pad, dtype=np.int64)
        channel_goal_globalmap_pad = np.pad(channel_goal_globalmap, self.border, self.pad_with, padder=0,
                                            dtype=np.int64)

        if self.consider_DObs:
            channel_state_globalmap[FOV_X[0]:FOV_X[1], FOV_Y[0]:FOV_Y[1]] = channel_state_FOV

        channel_combine = np.add(self.map_pad, channel_state_globalmap)
        channel_combine_pad = np.pad(channel_combine, self.border, self.pad_with, padder=0, dtype=np.int64)

        currentX_global_pad = currentX_global + self.FOV_width + self.border
        currentY_global_pad = currentY_global + self.FOV_width + self.border
        goalX_global_pad = goalX_global + self.FOV_width + self.border
        goalY_global_pad = goalY_global + self.FOV_width + self.border

        FOV_X_pad = [currentX_global, currentX_global + 2 * self.FOV_width + 1 + 2 * self.border]
        FOV_Y_pad = [currentY_global, currentY_global + 2 * self.FOV_width + 1 + 2 * self.border]

        # check if the agents standing at current agent's goal
        if channel_combine_pad[goalX_global_pad][goalY_global_pad] == 1:
            channel_combine_pad[goalX_global_pad][goalY_global_pad] = 0

        # if mode:
        # print("-------------------------\n------------ {} ------------\n------------------------\n".format(id_agent))
        # print("------------{} World Map -----------\n{}".format(id_agent,self.map_pad))
        # print("------------{} A* Map -----------\n{}\n".format(id_agent,channel_combine_pad))

        full_path, full_path_length = self.Path_Planner.a_star(channel_combine_pad,
                                                               [currentX_global_pad, currentY_global_pad],
                                                               [goalX_global_pad, goalY_global_pad])

        for id_path in range(full_path_length):
            posX_goalFOV = full_path[id_path][0]
            posY_goalFOV = full_path[id_path][1]
            # print(posX_goalFOV,posY_goalFOV)
            channel_goal_globalmap_pad[posX_goalFOV][posY_goalFOV] = 1
            record_localPath[id_path][0] = posX_goalFOV - 1 - self.FOV_width
            record_localPath[id_path][1] = posY_goalFOV - 1 - self.FOV_width

        channel_goal = np.zeros_like(channel_map, dtype=np.int64)
        channel_goal[:, :] = channel_goal_globalmap_pad[FOV_X_pad[0]:FOV_X_pad[1], FOV_Y_pad[0]:FOV_Y_pad[1]]
        # if mode:
        # print("\n-----------{} - Goal -----------\n{}".format(id_agent, channel_goal_globalmap_pad))
        # print("\n", channel_goal)
        # print("\n", goalX_global, goalY_global)

        input_step_currentAgent.append(channel_map.astype(np.float64))
        input_step_currentAgent.append(channel_goal.astype(np.float64))
        input_step_currentAgent.append(channel_state.astype(np.float64))

        return input_step_currentAgent, record_localPath

