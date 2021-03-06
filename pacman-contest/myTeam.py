# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions, Configuration
import game
from collections import defaultdict
import operator
import os
import heapq
import pickle
from util import nearestPoint, Counter, manhattanDistance


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ApproximateQAgent', second='DefensiveAgent', numTraining=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    distributions = dict()
    teamsInitialPosition = dict()
    enemiesStartingPos = dict()
    teamsRegistered = False
    walls = []

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.walls = gameState.getWalls()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def debugBelieveSystem(self):
        for (opponent, distribution) in self.distributions.items():
            self.debugDraw(distribution.keys(), [1, 0, 0], True)

    def setInitialDistributions(self, gameState):
        """
        Initialize the enemies position distribution to 100% at their starting point.
        """

        wallPos = gameState.getWalls()
        opponents = self.getOpponents(gameState)

        # set initial position
        teamIndex = self.getTeam(gameState)
        self.teamsInitialPosition[teamIndex[0]] = gameState.getAgentState(teamIndex[0]).getPosition()
        self.teamsInitialPosition[teamIndex[1]] = gameState.getAgentState(teamIndex[1]).getPosition()

        i = 0
        for (agent, position) in self.teamsInitialPosition.items():
            enemieStart = (wallPos.width - position[0] - 1,
                           wallPos.height - position[1] - 1)

            self.enemiesStartingPos[opponents[i]] = enemieStart
            self.distributions[opponents[i]] = Counter()
            self.distributions[opponents[i]][enemieStart] = 1
            i += 1

        self.updateEnemyDistributions(gameState)

        # reverse positions asigned to each agent
        if not self.distributions[opponents[0]]:
            i = 0
            for (agent, position) in self.teamsInitialPosition.items():
                enemieStart = (wallPos.width - position[0] - 1,
                               position[1])

                self.enemiesStartingPos[opponents[i]] = enemieStart
                self.distributions[opponents[i]] = Counter()
                self.distributions[opponents[i]][enemieStart] = 1
                i += 1

    def updateEnemyDistributions(self, gameState, secoundPass=False, deepth=0):

        for (opponent, positions) in self.distributions.items():
            opponentAgentPosition = gameState.getAgentState(opponent).getPosition()
            if not opponentAgentPosition == None:
                self.distributions[opponent] = dict()
                self.distributions[opponent][opponentAgentPosition] = 1
            else:
                for (position, status) in self.distributions[opponent].items():
                    config = Configuration(position, Directions.STOP)
                    actions = Actions.getPossibleActions(config, self.walls)

                    if (secoundPass):
                        actions = [Directions.STOP]

                    for action in actions:
                        dir = Actions.directionToVector(action)
                        new_position = (abs(dir[0] + position[0]), abs(dir[1] + position[1]))

                        noise_distance = self.getCurrentObservation().getAgentDistances()[opponent]

                        # distance is 8 as we need to include their next move taking them outside of our ping area
                        max_distance = noise_distance + 8
                        min_distance = noise_distance - 8

                        myPos = gameState.getAgentState(self.index).getPosition()
                        distance = util.manhattanDistance(myPos, new_position)

                        # Check possible locations that fit inside the noise reading minus the area we can see
                        if distance >= min_distance and distance <= max_distance and distance > 4:
                            self.distributions[opponent][new_position] = 1
                        else:
                            self.distributions[opponent].pop(new_position, None)

            if deepth == 1:
                return False

            # If we kill them reset distributions
            if not self.distributions[opponent]:
                self.distributions[opponent][self.enemiesStartingPos[opponent]] = 1
                return self.updateEnemyDistributions(gameState, deepth=deepth + 1)

        return True


class ApproximateQAgent(DummyAgent):

    def __init__(self, index, epsilon=0.001, gamma=0.9, alpha=0.1, numTraining=0, saveWeights=False, loadWeights=False,
                 savePath="weigths.pickle", loadPath="weigths.pickle", **args):

        # alpha - learning
        # rate
        # epsilon - exploration
        # rate
        # gamma - discount
        # factor

        CaptureAgent.__init__(self, index)
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining

        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.weights = util.Counter()

        self.saveWeights = saveWeights
        self.savePath = savePath
        self.loadWeights = loadWeights

        self.counter = 0

        if (self.loadWeights):

            with open(loadPath, 'rb') as f:
                print("loading weights from " + os.path.realpath(f.name))
                self.weights = pickle.load(f)
                # print(self.weights)
        else:
            self.weights = util.Counter()

        # IF its the comp use these weights
        if True:
            self.weights = {'enemyPacManDistance': -300.5177004950197802, 'scoredPoints': 17.392823999999997,
                            'distanceToFood': 1.0827753607280046, 'foodICanReturn': -14.0866415843544661,
                            'ghostDistance': -100.19742345269455042, 'foodEaten': 6.2513475884878105, 'invaders': -300.130592394829,
                            'defenders': -300.5013917348572023, 'estimatedEnemiYCord': 100.04375342308572,
                            'stop': -10.23098420937109398}

        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)
        self.teamsInitialPosition[self.index] = gameState.getAgentState(self.index).getPosition()

    def computeActionFromQValues(self, state):
        legal_actions = state.getLegalActions(self.index)

        # Dirty hack to make sure it doesnt stand still on food
        myPos = state.getAgentState(self.index).getPosition()
        prevGameState = self.getPreviousObservation()
        if prevGameState and self.getFood(prevGameState)[int(myPos[0])][int(myPos[1])]:  # and myPos!= teamMatePosition:
            legal_actions.remove(Directions.STOP)

        random.shuffle(legal_actions)

        successor_q_values = {action: self.getQValue(state, action) for action in legal_actions}
        max_action = max(successor_q_values.iterkeys(), key=(lambda key: successor_q_values[key]))

        return max_action if legal_actions else None

    def computeValueFromQValues(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        successor_q_values = [self.getQValue(gameState, action) for action in legal_actions]
        return max(successor_q_values) if legal_actions else 0.0

    def findOptimalAction(self, gameState):

        legalActions = gameState.getLegalActions(self.index)

        if self.numTraining > 0:
            self.epsilon * 0.9
            random_action = util.flipCoin(self.epsilon)
        else:
            random_action = False

        action = self.computeActionFromQValues(gameState)

        return random.choice(legalActions) if random_action else action

    # Find all dead end in the maze
    def getAllDeadEnd(self):
        observation = self.getCurrentObservation()
        validPosition = list()
        wallPos = observation.getWalls()
        for x in range(wallPos.width):
            for y in range(wallPos.height):
                if not wallPos[x][y]:
                    validPosition.append((x, y))

        allDeadEnd = list()
        for pos in validPosition:
            (x, y) = pos
            count = 0
            if wallPos[x + 1][y] and x + 1 < wallPos.width:
                count += 1
            if wallPos[x - 1][y] and x >= 1:
                count += 1
            if wallPos[x][y + 1] and y + 1 < wallPos.height:
                count += 1
            if wallPos[x][y - 1] and y >= 1:
                count += 1
            if count >= 3:
                allDeadEnd.append(pos)
        return allDeadEnd

    # Recursively get the next valid position until reach the entrance to dead end
    def getNextPos(self, pos, prePos):
        observation = self.getCurrentObservation()
        wallPos = observation.getWalls()
        (x, y) = pos
        count = 0
        if wallPos[x + 1][y] and x + 1 < wallPos.width:
            count += 1
        if wallPos[x - 1][y] and x >= 1:
            count += 1
        if wallPos[x][y + 1] and y + 1 < wallPos.height:
            count += 1
        if wallPos[x][y - 1] and y >= 1:
            count += 1
        if count <= 1:
            return prePos

        if not wallPos[x + 1][y] and x + 1 < wallPos.width and (x + 1, y) != prePos:
            return self.getNextPos((x + 1, y), pos)
        if not wallPos[x - 1][y] and x >= 1 and (x - 1, y) != prePos:
            return self.getNextPos((x - 1, y), pos)
        if not wallPos[x][y + 1] and y + 1 < wallPos.height and (x, y + 1) != prePos:
            return self.getNextPos((x, y + 1), pos)
        if not wallPos[x][y - 1] and y >= 1 and (x, y - 1) != prePos:
            return self.getNextPos((x, y - 1), pos)

    # Return a list of each entrance to each dead end with the distance to it
    def getDeadEndDistance(self):
        distanceList = list()
        for pos in self.getAllDeadEnd():
            entrance = self.getNextPos(pos, None)
            distance = self.getMazeDistance(entrance, pos)
            node = entrance, distance
            if distance > 0:
                distanceList.append(node)
        return distanceList

    # find the path to the closet border if we are close to ghost and have food in carriage
    def aStarSearch(self, gameState):
        visited = list()
        initialPosition = gameState.getAgentState(self.index).getPosition()

        grid = gameState.getWalls()
        borderX = grid.width / 2

        if self.red:
            borderX -= 1

        goal = [(borderX, y) for y in range(self.getFood(gameState).height) if
                not gameState.hasWall(borderX, y)]

        agenda = PriorityQueue()
        agenda.push((initialPosition, [], 0), 0)

        while not agenda.isEmpty():
            current, path, cost = agenda.pop()
            if not current in visited:
                visited.append(current)
                if current in goal:
                    return path
                else:
                    config = Configuration(current, Directions.STOP)
                    actions = Actions.getPossibleActions(config, self.walls)
                    nextStates = {((abs(Actions.directionToVector(action)[0] + current[0]),
                                    abs(Actions.directionToVector(action)[1] + current[1])),
                                   action, cost) for action in actions}
                    for state in nextStates:
                        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
                        ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
                        nogo = [(position + action for (k, action) in Actions._directions) for position in ghosts]
                        nogo += ghosts
                        if not state in visited:
                            if current in nogo:
                                continue
                            elif current in self.getFood(gameState).asList():
                                additional_cost = min([manhattanDistance(current, g) for g in goal]) - 50
                            else:
                                additional_cost = min([manhattanDistance(current, g) for g in goal])
                            new_cost = cost + state[2]
                            agenda.update((state[0], path + [state[1]], new_cost), new_cost + additional_cost)
            else:
                continue

        return path

    def chooseAction(self, gameState):

        myPostion = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghost) > 0:
            if ghost[0].scaredTimer < 2:
                dists = [self.getMazeDistance(myPostion, a.getPosition()) for a in ghost]
                if min(dists) < 5 and gameState.getAgentState(self.index).isPacman:
                    escapeRoute = self.aStarSearch(gameState)
                    if escapeRoute:
                        return escapeRoute.pop(0)


        if not self.teamsRegistered:
            self.setInitialDistributions(gameState)
            self.teamsRegistered = True

        self.updateEnemyDistributions(gameState)

        action = self.findOptimalAction(gameState)

        # For dead end implementation
        deadEndDistances = self.getDeadEndDistance()
        if len(ghost) > 0:
            successor = gameState.generateSuccessor(self.index, action)
            myNextPos = successor.getAgentState(self.index).getPosition()

            for opponent in self.getOpponents(gameState):
                if self.distributions[opponent]:
                    minGhostDist = min(self.getMazeDistance(myPostion, pos) for pos in self.distributions[opponent])
            for pos, dist in deadEndDistances:
                if myNextPos == pos:
                    if dist * 2 < minGhostDist:
                        self.escapeRoute = self.aStarSearch(gameState)
                        action = self.escapeRoute.pop(0)
                        return action

        if self.numTraining > 0:
            self.observation(gameState)

        self.lastState = gameState
        self.lastAction = action

        return action

    def getWeights(self):

        return self.weights

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)

        return Counter(self.weights) * Counter(features)

    def getCustomScore(self, currentGameState):

        prevGameState = self.getPreviousObservation()
        if prevGameState == None:
            return 0

        myAgentState = currentGameState.getAgentState(self.index)

        teamIndexs = self.getTeam(currentGameState)
        teamIndexs.remove(self.index)
        teamMateIndex = teamIndexs[0]

        myPostion = currentGameState.getAgentPosition(self.index)
        teamMatePosition = currentGameState.getAgentPosition(teamMateIndex)

        score = 0
        if self.red:
            if myAgentState.numCarrying == 0 and prevGameState.getAgentState(self.index).numCarrying > 0:
                score = self.getScore(currentGameState) - self.getScore(self.getPreviousObservation())
        else:
            if myAgentState.numCarrying == 0 and prevGameState.getAgentState(self.index).numCarrying > 0:
                score = self.getScore(self.getPreviousObservation()) - self.getScore(currentGameState)

        pickedUpFood = 1 if self.getFood(prevGameState)[myPostion[0]][
                                myPostion[1]] and myPostion != teamMatePosition else 0

        enemies = [currentGameState.getAgentState(i) for i in self.getOpponents(currentGameState)]
        ghost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        enemyPacMan = [a for a in enemies if a.isPacman and a.getPosition() != None]

        inKillZone = 0
        nearGhost = 0
        if len(ghost) > 0:
            dists = [self.getMazeDistance(myPostion, a.getPosition()) for a in ghost]
            closestGhostDistance = min(dists)
            if min(dists) < 2:
                inKillZone = -3
                pickedUpFood = -1

        customScore = 10 * score + pickedUpFood + inKillZone

        return customScore

    def getFeatures(self, state, action):
        features = util.Counter()

        successor = self.getSuccessor(state, action)
        currentAgentState = state.getAgentState(self.index)
        successorAgentState = successor.getAgentState(self.index)

        prevGameState = self.getPreviousObservation()
        foodList = self.getFood(successor).asList()
        defendingFood = self.getFoodYouAreDefending(state)
        myPrevPos = state.getAgentState(self.index).getPosition()
        myPos = successor.getAgentState(self.index).getPosition()

        teamIndexs = self.getTeam(state)
        teamIndexs.remove(self.index)
        teamMateIndex = teamIndexs[0]
        teamMatePosition = state.getAgentPosition(teamMateIndex)


        if prevGameState:
            prevScore = self.getScore(prevGameState)
            currentScore = self.getScore(state)

            if self.red:
                if prevScore - currentScore > 0:
                    features['scoredPoints'] = 1
            else:
                if currentScore - prevScore < 0:
                    features['scoredPoints'] = 1

            features['foodEaten'] = 1 if self.getFood(prevGameState)[int(myPos[0])][
                                             int(myPos[1])] and myPos != teamMatePosition else 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        closestGhostDistance = float('inf')
        closestPacmanDistance = float('inf')

        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            closestGhostDistance = min(dists)

            locationsOfGhosts = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            for ghost in ghosts:
                if closestGhostDistance < 3 or self.getMazeDistance(myPrevPos,
                                                                    ghost.getPosition()) == 1:  
                    if not ghost.scaredTimer:
                        features['ghostDistance'] = 1

                if not currentAgentState.isPacman and closestGhostDistance < 4:
                    features['ghostDistance'] = 1

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacMan = [a for a in enemies if a.isPacman and a.getPosition() != None]
        attackRange = 4
        if len(enemyPacMan) > 0:
            dists = [self.getMazeDistance(myPrevPos, a.getPosition()) for a in enemyPacMan]
            closestPacmanDistance = min(dists)

            for pacman in enemyPacMan:
                if self.getMazeDistance(myPos, pacman.getPosition()) == 1 or closestPacmanDistance < attackRange:
                    attack = []
                    attack = range(1, attackRange + 1)
                    attack.reverse
                    features['enemyPacManDistance'] = attack[closestPacmanDistance]

        # also check we didnt die 
        if len(foodList) > 0 and not features['ghostDistance'] == 1:
            # myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])

            # If zero we are standing on food so ignore and look for next food
            features['distanceToFood'] = -(minDistance / 100.0)

        if state.isOver() or successor.isOver():
            features.clear()
            features['scoredPoints'] = currentScore
            return features

        # First round initalising new game ignore feature space
        if util.manhattanDistance(currentAgentState.getPosition(), self.teamsInitialPosition[self.index]) == 0:
            return features


        grid = state.getWalls()
        halfway = grid.width / 2
        borderPositions = [(halfway, y) for y in range(self.getFood(state).height) if
                           not state.hasWall(halfway, y)]


        if not currentAgentState.isPacman and features['numInvaders'] == 0:
            borderPositions = [(halfway, y) for y in range(self.getFood(state).height) if
                               not state.hasWall(halfway, y)]

            if not self.red:
                xrange = range(halfway)
            else:
                xrange = range(halfway, grid.width)

        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)


        elif self.distributions:
            positions = [distribution.keys() for (enemie, distribution) in self.distributions.items()]
            positions = positions[0] + positions[1]
            positions = [(abs(position[0] - halfway), position[1]) for position in positions]
            entryDict = defaultdict(float)
            for (x, y) in positions:
                entryDict[y] += 1.0 / x if x != 0 else 0.0
            features['estimatedEnemiYCord'] = abs(
                myPos[1] - max(entryDict.iteritems(), key=operator.itemgetter(1))[0]) / 100.0 * ( -1  * features['distanceToFood'] / 100.0)


        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        enemyPacMan = [a for a in enemies if a.isPacman and a.getPosition() != None]

        locationsOfGhosts = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
        scaredEnemyGhost = False
        attackRange = 2
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            scared = sum([ghost.scaredTimer for ghost in ghosts]) / len(ghosts)

            if min(dists) < 2 and scared > 2:
                attack = []
                attack = range(1, attackRange + 1)
                attack = [-x for x in attack]
                attack.reverse()

                features['ghostDistance'] = attack[min(dists)]
            elif min(dists) < 4 and scared > 2:
                scaredEnemyGhost = True

        if len(enemyPacMan) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemyPacMan]

            if min(dists) < 4 and state.getAgentState(self.index).scaredTimer > 2:
                features['enemyPacManDistance'] = -1  

        if action == Directions.STOP: features['stop'] = 1

        features['foodICanReturn'] = self.getMazeDistance(myPos, self.teamsInitialPosition[
            self.index]) / 100.0 * currentAgentState.numCarrying + 1

        foodLeft = len(self.getFood(state).asList())

        if features['ghostDistance'] == 1 and not scaredEnemyGhost:

            temp = features['foodICanReturn']
            features.clear()

            features['distanceToFood'] = 0
            features['foodICanReturn'] = temp

            features['ghostDistance'] = 1
            return features

        # kill scared enemy ghost
        elif features['ghostDistance'] < 0:

            temp = features['ghostDistance']
            tempFood = features['distanceToFood']
            features.clear()
            features['distanceToFood'] = tempFood
            defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            features['defenders'] = len(defenders)
            features['ghostDistance'] = 1
            return features

        # Kill enemy pacman
        elif features['enemyPacManDistance'] > 0:
            tempDist = features['enemyPacManDistance']
            tempFood = features['distanceToFood']

            features['distanceToFood'] = tempFood
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = len(invaders)
            features['enemyPacManDistance'] = 5
            return features

        # Score one for the home team ^_^
        elif features['scoredPoints'] == 1:
            features.clear()
            features['scoredPoints'] = 1
            return features

        elif features['scoredPoints'] == -1:
            features.clear()
            features['scoredPoints'] = -1

        # Enemy pacman can kill us (we are scared)
        elif features['enemyPacManDistance'] == -1:
            features.clear()
            features['enemyPacManDistance'] = -1
            return features


        # Num num eat some tasy food stuffs
        elif features['foodEaten'] == 1:
            temp = features['distanceToFood']
            features.clear()
            features['distanceToFood'] = temp
            features['foodEaten'] = 1
            return features


        elif closestGhostDistance < 4 and not scaredEnemyGhost or (
                successorAgentState.numCarrying > 5 and features['distanceToFood'] < -0.03) or foodLeft < 3:

            temp = features['foodICanReturn']
            ghostCanKill = features['ghostDistance']
            features.clear()
            features['foodICanReturn'] = temp
            features['ghostDistance'] = ghostCanKill
            return features

        # Check if we got points for no reason, bug not to do with what we did
        temp = features['distanceToFood']
        if self.red:
            if self.getCustomScore(state) > 0:
                features.clear()
                features['distanceToFood'] = temp
                return features

        else:
            if self.getCustomScore(state) < 0:
                features.clear()
                features['distanceToFood'] = temp
                return features

        # Keep hunting for food if we reach this point
        features['foodICanReturn'] = 0

        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        features['defenders'] = len(defenders)
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        return features

    def update(self, gameState, action, nextState, reward):

        features = self.getFeatures(gameState, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(gameState,
                                                                                                         action)
        self.weights = {k: self.weights.get(k, 0) + self.alpha * difference * features.get(k, 0) for k in
                        set(self.weights) | set(features)}

    def observeTransition(self, state, action, nextState, deltaReward):

        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def observation(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """

        if not self.lastState == None:
            reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
            self.observeTransition(self.lastState, self.lastAction, gameState, reward)

        return gameState

    def terminal(self, state):
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))

    def final(self, state):
        # call the super-class final method
        self.terminal(state)

        if self.saveWeights:
            with open(self.savePath, 'wb') as f:
                pickle.dump(self.weights, f, pickle.HIGHEST_PROTOCOL)
                print "Saving weights %s" % self.weights
        if self.episodesSoFar == self.numTraining:
            print("episode equals numTraining, begin testing afterwards")


class DefensiveAgent(DummyAgent):

    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        self.updateEnemyDistributions(gameState, True)

        actions = gameState.getLegalActions(self.index)
        myState = gameState.getAgentState(self.index)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        if invaders:
            dist = self.getMazeDistance(invaders[0].getPosition(), myState.getPosition())
            if myState.scaredTimer < 2 and dist < 3:
                enemyY = invaders[0].getPosition()[1]
                myY = myState.getPosition()[1]
                if enemyY == myY:
                    return Directions.STOP
                elif enemyY < myY and Directions.SOUTH in actions:
                    return Directions.SOUTH
                elif enemyY > myY and Directions.NORTH in actions:
                    return Directions.NORTH


        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        agentDistances = gameState.getAgentDistances()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        grid = gameState.getWalls()
        halfway = grid.width / 2

        features['defendFood'] = len(self.getFoodYouAreDefending(successor).asList())

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 0 if myState.isPacman else 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        if not myState.isPacman and features['numInvaders'] == 0:
            borderPositions = [(halfway, y) for y in range(self.getFood(gameState).height) if
                               not gameState.hasWall(halfway, y)]

            if not self.red:
                xrange = range(halfway)
            else:
                xrange = range(halfway, grid.width)

            for x in xrange:
                for y in range(grid.height):
                    if not grid[x][y]:
                        borderDistances = min(
                            self.getMazeDistance(myPos, borderPos) for borderPos in borderPositions)
                        features['boarderDistance'] = (borderDistances / 100.0)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        elif self.distributions:
            positions = [distribution.keys() for (enemie, distribution) in self.distributions.items()]
            positions = positions[0] + positions[1]
            positions = [(abs(position[0] - halfway), position[1]) for position in positions]
            entryDict = defaultdict(float)
            for (x, y) in positions:
                entryDict[y] += 1.0 / x if x != 0 else 0.0
            features['estimatedEntryPoint'] = abs(
                myPos[1] - max(entryDict.iteritems(), key=operator.itemgetter(1))[0]) / 100.0

        return features

    def getWeights(self, gameState, action):

        return {'numInvaders': -1000, 'onDefense': 20, 'invaderDistance': -10, 'stop': -10,
                'defendFood': 1, 'reverse': -2, 'boarderDistance': -70, 'estimatedEntryPoint': -150}


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
