# myPacmanAgents.py

import game
from game import Actions
from game import Directions
import random
import util
from util import Queue
import numpy as np
from util import manhattanDistance


def shortest_path(walls, start, end):
    """Search the shallowest nodes in the search tree first."""
    start = [start[0], start[1]]
    end = [end[0], end[1]]
    if start == end:
        return [start]
    neighbours = Queue()  # queue storing the next positions to explore
    neighbours.push(start)
    counts = np.zeros((walls.width, walls.height),
                      dtype=int)  # 2D array to store the distance from the start to all visted points
    predecessors = np.zeros((counts.shape[0], counts.shape[1], 2),
                            dtype=int)  # 2D array storing the predecessors (past points allowing path to be retraced)
    counts[start[0], start[1]] = 1
    # loop until the end position is found
    while not neighbours.isEmpty():
        n = neighbours.pop()

        if n == end:

            break  # path found
        # add all the valid neighbours to the list and remember from where they came from
        for neighbour in [[n[0] - 1, n[1]], [n[0] + 1, n[1]], [n[0], n[1] - 1], [n[0], n[1] + 1]]:
            if not walls[neighbour[0]][neighbour[1]] and counts[neighbour[0], neighbour[1]] == 0:
                neighbours.push(neighbour)
                predecessors[neighbour[0], neighbour[1]] = n
                counts[neighbour[0], neighbour[1]] = counts[n[0], n[1]] + 1

    if counts[end[0], end[1]] == 0:
        return []  # path not found

    path = []
    n = end

    # reconstruct the path

    while n != start:
        # print n
        if n == start:
            break
        path.append(n)
        n = predecessors[n[0], n[1]].tolist()
    path.append(start)


    return path




class MyAgent(game.Agent):

    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):

        ghostState = state.getGhostStates()
        isScared = ghostState > 0
        legalActions = state.getLegalActions(self.index)
        walls = state.getWalls()
        Ghostpos=state.getGhostPosition(self.index)
        pos = state.getPacmanPosition()
        capsulePosition = state.getCapsules()
        newFood = state.getFood()
        foodlist = newFood.asList()

        speed = 1

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(int(pos[0] + a[0]), int(pos[1] + a[1])) for a in actionVectors]
        
        distancesToGhost = [len(shortest_path(walls, pos, G)) for G in Ghostpos]  # PROBLEM finds the distance from ghost to pacman after tking the next step
        distanceToCapsule = [len(shortest_path(walls,pos,C)) for C in capsulePosition]
        distancesToFood = [len(shortest_path(walls, pos, f)) for f in foodlist]

        d = [len(shortest_path(walls, pos, Ghostpos)) for G in Ghostpos]  # PROBLEM #  current distance to pacman


        if isScared == False and d < 10:
            safe = False
        else:
            safe = True

        if safe:
            if isScared:  # do this when ghost is safe and scared or safe and pacman is far
                bestScore = min(distancesToGhost)  # keep aiming for ghost
                bestActions = [action for action, distance in zip(legalActions, distancesToGhost) if distance == bestScore]
            else:  # do this if pacman is safe and ghost not scared
                bestScore = min(distancesToFood)  #finddistancetofood
                bestActions = [action for action, distance in zip(legalActions, distancesToFood) if distance == bestScore]
        else:  # do this when not safe
            bestScore = min(distanceToCapsule)
            bestActions = [action for action, distance in zip(legalActions,distanceToCapsule) if distance == bestScore]



        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = 1 / len(bestActions)
        #for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist





def scoreEvaluation(state):
   return state.getScore()
