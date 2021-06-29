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
sys.setrecursionlimit(5000)

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
parser.add_argument('--path_loadmap', type=str, default='../MultiAgentDataset/Solution_BMap/Storage_Map/BenchMarkMap')
parser.add_argument('--loadmap_TYPE', type=str, default='maze')
parser.add_argument('--path_save', type=str, default='../data/test_data_gen')
parser.add_argument('--gen_map_type', type=str, default='maze')
parser.add_argument('--path_size', type=int, default=0)
parser.add_argument('--central_path_size', type=int, default=1)

args = parser.parse_args()

# set random seed
np.random.seed(1337)


def handler(signum, frame):
    raise Exception("Solution computed by Expert is timeout.")


class EnvsGen:
    def __init__(self, config):
        self.config = config

        self.random_map = config.random_map
        print(self.random_map)
        self.path_loadmap = config.path_loadmap
        self.num_agents = config.num_agents
        self.num_data = config.num_dataset
        self.path_save = config.path_save

        self.map_density = config.map_density
        self.label_density = str(config.map_density).split('.')[-1]
        self.map_TYPE = 'map'
        self.size_load_map = (config.map_width, config.map_width)
        self.map_complexity = config.map_complexity
        self.gen_map_type = config.gen_map_type
        self.path_size = config.path_size
        self.central_path_size = config.central_path_size
        self.createFolder()

        self.pair_CasesPool = []
        self.PROCESS_NUMBER = 4
        self.timeout = 300
        self.task_queue = Queue()
        self.maxNumObstacle = 1.3*self.map_density*config.map_width*config.map_width


    def createFolder(self):
        self.dirName_root = os.path.join(self.path_save,'{}{:02d}x{:02d}_density_p{}/'.format(self.map_TYPE, self.size_load_map[0],
                                                                                                         self.size_load_map[1],
                                                                                                         self.label_density))

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

            os.makedirs(self.dirName_mapSet)
            os.makedirs(self.dirName_mapSet_png)
            print("Directory ", self.dirName_mapSet, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass



    def mapGen(self, width=10, height=10, complexity=0.01, density=0.1,
               path_size=0, central_path_size=2,
               seed_range=2, slice_num=(3, 3), random_ratio=0.1):
        # Only odd shapes
        # world_size = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # world_size = ((height // 2) * 2 , (width // 2) * 2 )
        world_size = (height, width)



        if self.gen_map_type == 'maze':

            # Adjust complexity and density relative to maze size

            # number of components
            complexity = int(complexity * (5 * (world_size[0] + world_size[1])))
            # size of components
            density = int(density * ((world_size[0] // 2) * (world_size[1] // 2)))

            # density = int(density * world_size[0] * world_size[1])

            # Build actual maze
            maze = np.zeros(world_size, dtype=np.int64)

            # Make aisles
            for i in range(density):
                # x, y = np.random.randint(0, world_size[1]), np.random.randint(0, world_size[0])

                # pick a random position
                x, y = np.random.randint(0, world_size[1] // 2) * 2, np.random.randint(0, world_size[0] // 2) * 2

                maze[y, x] = 1
                for j in range(complexity):
                    neighbours = []
                    if x > 1:             neighbours.append((y, x - 2))
                    if x < world_size[1] - 2:  neighbours.append((y, x + 2))
                    if y > 1:             neighbours.append((y - 2, x))
                    if y < world_size[0] - 2:  neighbours.append((y + 2, x))
                    if len(neighbours):
                        y_, x_ = neighbours[np.random.randint(0, len(neighbours) - 1)]
                        if maze[y_, x_] == 0:
                            maze[y_, x_] = 1
                            maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
                # print(np.count_nonzero(maze))
            return maze
        elif self.gen_map_type == 'warehouse':
            # Build actual maze
            maze = np.ones(world_size, dtype=np.int64)
            if path_size == 0: # no regulatoin on path size
                maze[0] = 0
                maze[width-1] = 0
                maze[:, 0] = 0
                maze[:, height-1] = 0
                total_area = width * height
                current_density = np.count_nonzero(maze)/total_area
                while current_density > density:
                    if np.random.randint(2, size=1) == 0: # delete row
                        maze[np.random.randint(height, size=1)] = 0
                    else: # delete column
                        maze[:, np.random.randint(width, size=1)] = 0
                    current_density = np.count_nonzero(maze)/total_area
                return maze
            else:
                maze[:central_path_size] = 0
                maze[height-central_path_size:] = 0
                maze[:, :central_path_size] = 0
                maze[:, width-central_path_size:] = 0

                occupied_row = np.zeros(height)
                occupied_col = np.zeros(width)
                occupied_row[:central_path_size + 1] = 1
                occupied_row[height-central_path_size-1:] = 1
                occupied_col[:central_path_size + 1] = 1
                occupied_col[width-central_path_size-1:] = 1

                center_x = (
                int((width - central_path_size) / 2), int((width - central_path_size) / 2) + central_path_size)
                center_y = (
                    int((height - central_path_size) / 2), int((height - central_path_size) / 2) + central_path_size)

                maze[center_y[0]:center_y[1]] = 0
                maze[:, center_x[0]:center_x[1]] = 0

                occupied_row[center_y[0] - 1:center_y[1] + 1] = 1
                occupied_col[center_x[0] - 1:center_x[1] + 1] = 1

                # print(occupied_row, occupied_col)

                total_area = width * height
                current_density = np.count_nonzero(maze) / total_area
                fail_count = 0
                while current_density > density:
                    fail_count+=1
                    if fail_count>100:
                        print('Timeout to find solution. Please check whether the path '
                              'size is too strict. Density at {}, target is {}'.format(current_density, density))
                        break
                    if np.random.randint(2, size=1) == 0: # delete row
                        # print('row')
                        chosen = int(np.random.randint(height, size=1))
                        # print(chosen, occupied_row[chosen: chosen+path_size])
                        if np.count_nonzero(occupied_row[chosen: chosen+path_size]) == 0:
                            occupied_row[chosen - 1: chosen + path_size + 1] = 1
                            maze[chosen: chosen + path_size] = 0
                            fail_count = 0
                    else: # delete column
                        # print('col')
                        chosen = int(np.random.randint(width, size=1))
                        # print(chosen, occupied_col[chosen: chosen + path_size])
                        if np.count_nonzero(occupied_col[chosen: chosen+path_size]) == 0:
                            occupied_col[chosen - 1: chosen + path_size + 1] = 1
                            maze[:, chosen: chosen + path_size] = 0
                            fail_count = 0
                    current_density = np.count_nonzero(maze)/total_area
                    # print(current_density)

                print(maze)
                return maze

        elif self.gen_map_type == 'real_warehouse':
            maze = np.zeros(world_size, dtype=np.int64)
            slice_size = (world_size[0]/slice_num[0], world_size[1]/slice_num[1])
            group = np.random.randint(0, seed_range, size=slice_num)
            final_group = np.zeros(slice_num)
            # print(group)
            assigned_index = []
            group_index = 1
            for i in range(slice_num[0]):
                for j in range(slice_num[1]):
                    if (i, j) not in assigned_index:
                        # print('checking', i, j)
                        current_value = group[i, j]
                        # print('current value', current_value)
                        X_, Y_ = False, False
                        if (i+1) < slice_num[0]:
                            if group[i+1, j] == current_value:
                                X_ = True
                        if (j+1) < slice_num[1]:
                            if group[i, j+1] == current_value:
                                Y_ = True
                        if X_ == True and Y_ == True:
                            chosen_direction = int(np.random.randint(2, size=1))
                        else:
                            if X_ == True:
                                chosen_direction = 1
                            else:
                                chosen_direction = 0
                        if chosen_direction == 1:
                            # check along x
                            x = i
                            y = j
                            while group[x, y] == current_value:
                                # print(x, y)
                                final_group[x,y] = group_index
                                assigned_index.append((x,y))
                                x += 1
                                if x >= slice_num[0]:
                                    break
                            # print(final_group)
                        else:
                            # check along y
                            x = i
                            y = j
                            while group[x, y] == current_value:
                                # print(x, y)
                                final_group[x, y] = group_index
                                assigned_index.append((x, y))
                                y += 1
                                if y >= slice_num[1]:
                                    break
                            # print(final_group)
                        group_index+= 1

            # print(final_group)
            def generate_small_maze(size, density):
                # print(size, density)
                small_maze = np.zeros(size)
                chosen_direction = int(np.random.randint(2, size=1))
                # print('=====', chosen_direction)
                if chosen_direction == 1:
                    effective_density = (size[0] * size[1] * density) / (size[0] * size[1] - size[0] - 2 * size[1] + 2)
                    print('effective density', effective_density)
                    if effective_density == 1:
                        print('Too narrow to have this density! You should reduce slice num or other params!')
                        effective_density = 0.99
                    # wall along x
                    bar_number = int((size[1] - 2) * (effective_density))
                    interval = (size[1] - 2) / bar_number

                    print(bar_number, interval)
                    i = 1
                    while i < size[1]-2:
                        print(i, int(i))
                        small_maze[1:-1, int(i):int(i)+1] = 1
                        i+= interval
                else:
                    effective_density = (size[0] * size[1] * density) / (size[0] * size[1] - size[1] - 2 * size[0])
                    print('effective density', effective_density)
                    if effective_density == 1:
                        print('Too narrow to have this density! You should reduce slice num or other params!')
                        effective_density = 0.99
                    # wall along y
                    bar_number = int((size[0] - 2) * (effective_density))
                    interval = (size[0] - 2) / bar_number

                    print(bar_number, interval)
                    i = 1
                    while i < size[0] - 2:
                        print(i, int(i))
                        small_maze[int(i):int(i) + 1, 1:-1] = 1
                        i += interval
                print(small_maze)
                return small_maze

            for i in range(1, group_index):
                index_pos = np.where(final_group==i)
                min_index_x = min(index_pos[0])
                max_index_x = max(index_pos[0]) + 1
                min_index_y = min(index_pos[1])
                max_index_y = max(index_pos[1]) + 1
                # print(min_index_x, max_index_x, min_index_y, max_index_y)
                pos_min_x = int(height * (min_index_x / slice_num[0]) + 2*(np.random.rand(1)-0.5)*random_ratio*slice_size[0])
                pos_max_x = int(height * (max_index_x / slice_num[0])+ 2*(np.random.rand(1)-0.5)*random_ratio*slice_size[0])
                pos_min_y = int(width * (min_index_y / slice_num[1])+ 2*(np.random.rand(1)-0.5)*random_ratio*slice_size[1])
                pos_max_y = int(width * (max_index_y / slice_num[1])+ 2*(np.random.rand(1)-0.5)*random_ratio*slice_size[1])
                pos_min_x = max(0, pos_min_x)
                pos_max_x = min(world_size[0], pos_max_x)
                pos_min_y = max(0, pos_min_y)
                pos_max_y = min(world_size[1], pos_max_y)
                # print(pos_min_x, pos_max_x, pos_min_y, pos_max_y)
                maze[pos_min_x:pos_max_x, pos_min_y:pos_max_y] = generate_small_maze((pos_max_x-pos_min_x, pos_max_y-pos_min_y), density=density)

            print(maze)
            print('real density', np.count_nonzero(maze) / world_size[0] / world_size[1])
            return maze

        elif self.gen_map_type == 'random':
            # Build actual maze
            maze = np.random.random(world_size)
            total_area = width * height
            assert density <= 1
            assert density >= 0

            dense_area = (maze<density)
            inverse_dense_area = (maze>=density)
            maze[dense_area] = 1
            maze[inverse_dense_area] = 0
            return maze

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

    def mapload(self, id_env):
        load_env = self.path_loadmap + 'map_{:02d}x{:02d}_density_p{}_id_{:02d}.npy'.format(self.size_load_map[0], self.size_load_map[1],
                                                                                            self.map_density, id_env)
        map_env = np.load(load_env)
        return map_env


    def genMap(self):

        for id_Env in range(self.num_data):
            map_env, list_obstacle = self.setup_map(id_Env)

            self.saveMap(id_Env, list_obstacle)
            self.saveMap_numpy(id_Env, map_env)



    def setup_map_array(self, list_posObstacle):
        num_obstacle = len(list_posObstacle)
        map_data = np.zeros(self.size_load_map)

        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_posObstacle[ID_obs][0]
            obstacleIndexY = list_posObstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1

        return map_data

    def setup_map(self, id_random_env):

        # randomly generate map with specific setup
        map_env_raw = self.mapGen(width=self.size_load_map[0], height=self.size_load_map[1],
                              complexity=self.map_complexity, density=self.map_density,
                                  path_size=self.path_size, central_path_size=self.central_path_size)


        if self.gen_map_type == 'maze' or self.gen_map_type == 'random':
            map_env = self.img_fill(map_env_raw.astype(np.uint8), 0.5)
        else:
            map_env = map_env_raw*255

        array_freespace = np.argwhere(map_env == 0)
        num_freespace = array_freespace.shape[0]

        array_obstacle = np.transpose(np.nonzero(map_env))
        num_obstacle = array_obstacle.shape[0]

        print("###### ID_Env: {} - Check Map Size: [{},{}]- density: {} - Actual [{},{}] - #Obstacle: {} - #FreeSpace: {}".format(id_random_env, self.size_load_map[0],
                                                                                                      self.size_load_map[1], self.map_density,
                                                                                                      self.size_load_map[0], self.size_load_map[1],
                                                                                                      num_obstacle, num_freespace))

        list_obstacle = []
        for id_Obs in range(num_obstacle):
            list_obstacle.append((array_obstacle[id_Obs, 0], array_obstacle[id_Obs, 1]))

        if num_freespace == 0 or num_obstacle == 0:
            # print(array_freespace)
            print("ID_Env: {} - #Obstacle: {} - #FreeSpace: {}".format(id_random_env, num_obstacle, num_freespace))
            map_env,_ = self.setup_map(id_random_env)

        if  num_obstacle>=self.maxNumObstacle:
            map_env, _ = self.setup_map(id_random_env)

        return map_env, list_obstacle



    def saveMap(self,Id_env,list_obstacle):
        num_obstacle = len(list_obstacle)
        map_data = np.zeros([self.size_load_map[0], self.size_load_map[1]])


        aspect = self.size_load_map[0] / self.size_load_map[1]
        xmin = -0.5
        ymin = -0.5
        xmax = self.size_load_map[0] - 0.5
        ymax = self.size_load_map[1] - 0.5


        d = draw.Drawing(self.size_load_map[0], self.size_load_map[1], origin=(xmin,ymin))

        d.append(draw.Rectangle(xmin, ymin, self.size_load_map[0], self.size_load_map[1], stroke_width=0.1, stroke='black', fill='white'))



        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_obstacle[ID_obs][0]
            obstacleIndexY = list_obstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1
            d.append(draw.Rectangle(obstacleIndexY-0.5, obstacleIndexX-0.5, 1, 1, stroke='black', stroke_width=0, fill='black'))


        # setup figure
        # name_map = os.path.join(self.dirName_mapSet_png, 'Map_{:02d}x{:02d}_density_p{}_{:05d}.png'.format(self.size_load_map[0], self.size_load_map[1],
        #                                                                                     self.map_density,Id_env))

        name_map = os.path.join(self.dirName_mapSet_png, 'Map_ID{:05d}.png'.format(Id_env))
        d.setRenderSize(200, 200)  # Alternative to setPixelScale
        d.savePng(name_map)

    def saveMap_numpy(self, Id_env, map_env):
        map_data = {"map":map_env}

        # name_map = os.path.join(self.dirName_mapSet, 'Map_{:02d}x{:02d}_density_p{}_{:05d}.mat'.format(self.size_load_map[0], self.size_load_map[1],
        #                                                                                     self.map_density, Id_env))

        name_map = os.path.join(self.dirName_mapSet,'Map_ID{:05d}.mat'.format(Id_env))
        sio.savemat(name_map, map_data)




if __name__ == '__main__':



    path_savedata = '/home/pc/projects/graph_mapf/data/test_data_gen'



    dataset = EnvsGen(args)
    timeout = 300



    dataset.genMap()






