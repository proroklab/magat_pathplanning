'''
This file contains utility of AStarSearch.
Thanks to Binyu Wang for providing the codes.
'''

from random import randint
import numpy as np


class SearchEntry():
    def __init__(self, x, y, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        # cost move form start entry to this entry
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pre_entry = pre_entry

    def getPos(self):
        return (self.x, self.y)


def AStarSearch(img, source, dest):
    def getNewPosition(img, location, offset):
        x, y = (location.x + offset[0], location.y + offset[1])
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or img[x, y] == 1 or img[x, y] == 3:
            return None
        return (x, y)

    def getPositions(img, location):
        # use four ways or eight ways to move
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        # offsets = [(-1,0), (0, -1), (1, 0), (0, 1), (-1,-1), (1, -1), (-1, 1), (1, 1)]
        poslist = []
        for offset in offsets:
            pos = getNewPosition(img, location, offset)
            if pos is not None:
                poslist.append(pos)
        return poslist

    # imporve the heuristic distance more precisely in future
    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0]) + abs(dest.y - pos[1])

    def getMoveCost(location, pos):
        if location.x != pos[0] and location.y != pos[1]:
            return 1.4
        else:
            return 1

    # check if the position is in list
    def isInList(list, pos):
        if pos in list:
            return list[pos]
        return None

    # add available adjacent positions
    def addAdjacentPositions(img, location, dest, openlist, closedlist):
        poslist = getPositions(img, location)
        for pos in poslist:
            # if position is already in closedlist, do nothing
            if isInList(closedlist, pos) is None:
                findEntry = isInList(openlist, pos)
                h_cost = calHeuristic(pos, dest)
                g_cost = location.g_cost + getMoveCost(location, pos)
                if findEntry is None:
                    # if position is not in openlist, add it to openlist
                    openlist[pos] = SearchEntry(pos[0], pos[1], g_cost, g_cost + h_cost, location)
                elif findEntry.g_cost > g_cost:
                    # if position is in openlist and cost is larger than current one,
                    # then update cost and previous position
                    findEntry.g_cost = g_cost
                    findEntry.f_cost = g_cost + h_cost
                    findEntry.pre_entry = location

    # find a least cost position in openlist, return None if openlist is empty
    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None:
                fast = entry
            elif fast.f_cost > entry.f_cost:
                fast = entry
        return fast

    all_path = []
    openlist = {}
    closedlist = {}
    location = SearchEntry(source[0], source[1], 0.0)
    dest = SearchEntry(dest[0], dest[1], 0.0)
    openlist[source] = location
    while True:
        location = getFastPosition(openlist)
        if location is None:
            # not found valid path
            # 			print("can't find valid path")
            return ([source])

        if location.x == dest.x and location.y == dest.y:
            break

        closedlist[location.getPos()] = location
        openlist.pop(location.getPos())
        addAdjacentPositions(img, location, dest, openlist, closedlist)

    while location is not None:
        all_path.append([location.x, location.y])
        # img[location.x][location.y] = 2
        location = location.pre_entry

    return all_path[::-1]


def hca(img, all_start, all_end, steps=100):
    all_path = []
    robot_loc = np.where(img == 3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 3:
                img[i, j] = 0
    res_imgs = np.expand_dims(img, axis=0).repeat(steps, axis=0)
    for i in range(len(robot_loc[0])):
        res_imgs[0, robot_loc[0][i], robot_loc[1][i]] = 3
    for i in range(len(all_start)):
        robot_path = AStarTime(res_imgs, (all_start[i][0], all_start[i][1]), (all_end[i][0], all_end[i][1]))
        # print(i)
        if len(robot_path) == 1:
            new_path = []
            for j in range(steps - 1):
                res_imgs[j, all_start[i][0], all_start[i][1]] = 3
                new_path.append([all_start[i][0], all_start[i][1], j])
            all_path.append(new_path)
            continue
        else:
            for loc in robot_path:
                res_imgs[loc[2], loc[0], loc[1]] = 3
            all_path.append(robot_path)
    return all_path


class SearchEntryTime():
    def __init__(self, x, y, z, g_cost, f_cost=0, pre_entry=None):
        self.x = x
        self.y = y
        self.z = z
        # cost move form start entry to this entry
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pre_entry = pre_entry

    def getPos(self):
        return (self.x, self.y, self.z)


def AStarTime(imgs, source, dest, total_steps=80):
    def getNewPosition(img, location, offset, step=0):
        x, y = (location.x + offset[0], location.y + offset[1])
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or img[x, y] == 1 or img[x, y] == 3:
            return None
        return (x, y, step)

    def getPositions(img, location, step=0):
        # use four ways or eight ways to move
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        # offsets = [(-1,0), (0, -1), (1, 0), (0, 1), (-1,-1), (1, -1), (-1, 1), (1, 1)]
        poslist = []
        for offset in offsets:
            pos = getNewPosition(img, location, offset, step)
            if pos is not None:
                poslist.append(pos)
        return poslist

    # imporve the heuristic distance more precisely in future
    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0]) + abs(dest.y - pos[1])

    def getMoveCost(location, pos):
        if location.x != pos[0] and location.y != pos[1]:
            return 1.4
        else:
            return 1

    # check if the position is in list
    def isInList(list, pos):
        if pos in list:
            return list[pos]
        return None

    # add available adjacent positions
    def addAdjacentPositions(imgs, location, dest, openlist, closedlist, steps):
        img = imgs[int(steps + 1), :, :]
        poslist = getPositions(img, location, steps)
        for pos in poslist:
            # if position is already in closedlist, do nothing
            if isInList(closedlist, pos) is None:
                findEntry = isInList(openlist, pos)
                h_cost = calHeuristic(pos, dest)
                g_cost = location.g_cost + getMoveCost(location, pos)
                if findEntry is None:
                    # if position is not in openlist, add it to openlist
                    steps = int(g_cost)
                    openlist[(pos[0], pos[1], steps)] = SearchEntryTime(pos[0], pos[1], steps, g_cost, g_cost + h_cost,
                                                                        location)
                elif findEntry.g_cost > g_cost:
                    # if position is in openlist and cost is larger than current one,
                    # then update cost and previous position
                    findEntry.g_cost = g_cost
                    findEntry.f_cost = g_cost + h_cost
                    findEntry.z = int(g_cost)
                    findEntry.pre_entry = location

    # find a least cost position in openlist, return None if openlist is empty
    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None:
                fast = entry
            elif fast.f_cost > entry.f_cost:
                fast = entry
        return fast

    all_path = []
    openlist = {}
    closedlist = {}
    location = SearchEntryTime(source[0], source[1], 0, 0.0)
    dest = SearchEntryTime(dest[0], dest[1], 0, 0.0)
    openlist[(source[0], source[1], 0)] = location
    steps = 0
    while steps < total_steps:
        location = getFastPosition(openlist)
        if location is None:
            # not found valid path
            # 			print("can't find valid path")
            return ([source])

        if location.x == dest.x and location.y == dest.y:
            break

        closedlist[location.getPos()] = location
        openlist.pop(location.getPos())
        steps = int(location.g_cost)
        addAdjacentPositions(imgs, location, dest, openlist, closedlist, steps)

    while location is not None:
        all_path.append([location.x, location.y, location.z])
        # img[location.x][location.y] = 2
        location = location.pre_entry

    return all_path[::-1]

# img = np.zeros((20,20))
# source = (0,0)
# dest = (img.shape[0]-1, img.shape[1]-1)
# path = AStarSearch(img, source, dest)
