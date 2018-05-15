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
import os
import pickle
from util import nearestPoint, Counter


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
        i = 0

        for (agent, position) in self.teamsInitialPosition.items():
            # For one opponent's distribution
            enemieStart = (wallPos.width - position[0] - 1,
                           wallPos.height - position[1] - 1)

            self.enemiesStartingPos[opponents[i]] = enemieStart
            self.distributions[opponents[i]] = Counter()
            self.distributions[opponents[i]][enemieStart] = 1
            i += 1

    # def reverseManhatten(self, position, distance):
    #     up = [(position[0] + x, position[1] + distance - abs(x)) for x in range(-distance, distance + 1)]
    #     down = [(position[0] + distance - abs(x), position[1] + x) for x in range(-distance, distance + 1)]
    #
    #     print [(position[0] + x, position[1] + distance - x) for x in range(distance + 1)]

    # def manhattenDistanceGrid(self, gameState, opponent):
    #     distance = self.getCurrentObservation().getAgentDistances()[opponent]
    #     noise = 6
    #     position = gameState.getAgentState(self.index).getPosition()
    #     self.reverseManhatten(position, distance)
    #     pass

    def updateEnemyDistributions(self, gameState, secoundPass=False):


        for (opponent, positions) in self.distributions.items():
            opponentAgentPosition = gameState.getAgentState(opponent).getPosition()
            if not opponentAgentPosition == None:
                self.distributions[opponent] = dict()
                self.distributions[opponent][opponentAgentPosition] = 1
            else:
                for (position, status) in self.distributions[opponent].items():
                    config = Configuration(position, Directions.STOP)
                    actions = Actions.getPossibleActions(config, self.walls)

                    if(secoundPass):
                        actions = [Directions.STOP] 

                    for action in actions:
                        dir = Actions.directionToVector(action)
                        new_position = (abs(dir[0] + position[0]), abs(dir[1] + position[1]))


                        noise_distance = self.getCurrentObservation().getAgentDistances()[opponent]
                        # distance is 7 as we need to include there possible movie taking them outside of our ping area
                        max_distance = noise_distance + 7
                        min_distance = noise_distance - 7

                        myPos = gameState.getAgentState(self.index).getPosition()
                        distance = util.manhattanDistance(myPos, new_position)

                        # Check possible locations that fit inside the noise reading minus the area we can see
                        # add the visula range of my team mate as well
                        if distance >= min_distance and  distance <= max_distance and distance > 4:
                            self.distributions[opponent][new_position] = 1
                        else:
                            self.distributions[opponent].pop(new_position, None)

            # If we kill them reset distributions
        if not self.distributions[opponent]:
            self.distributions[opponent][self.enemiesStartingPos[opponent]] = 1
            self.updateEnemyDistributions(gameState)

        for (opponent, distribution) in self.distributions.items():
            if(opponent == 1):
                self.debugDraw(distribution.keys(), [1, 0, 0], True)
            if(opponent == 3):
                self.debugDraw(distribution.keys(), [0, 0, 1], True)




class ApproximateQAgent(DummyAgent):

    def __init__(self, index, epsilon=0.2, gamma=0.2, alpha=0.2, numTraining=0, saveWeights=False, loadWeights=False,
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
                print(self.weights)
        else:
            self.weights = util.Counter()

        # IF its the comp use these weights
        if True:
            self.weights = {'reverse': -3.1193152508543696, 'stop': -0.02860773042212779,
                            'enemyPacManDistance': -0.19993697043215167,
                            'scoredPoints': 20.988673473991977, 'distanceToFood': 0.6040656773252039,
                            'foodICanReturn': 4.650475210934178,
                            'foodLeft': 1.8896938596362576, 'ghostDistance': 2.422286242755341,
                            'foodEaten': 0.43581990540752097, 'distanceToEnemyPacMan': -100}

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
        if self.episodesSoFar == 0 and self.numTraining != 0:
            print 'Beginning %d episodes of Training' % (self.numTraining)

    def computeActionFromQValues(self, state):
        legal_actions = state.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)

        random.shuffle(legal_actions)

        successor_q_values = {action: self.getQValue(state, action) for action in legal_actions}
        max_action = max(successor_q_values.iterkeys(), key=(lambda key: successor_q_values[key]))

        # print "successor_q_values", successor_q_values
        return max_action if legal_actions else None

    def computeValueFromQValues(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        successor_q_values = [self.getQValue(gameState, action) for action in legal_actions]
        return max(successor_q_values) if legal_actions else 0.0

    def findOptimalAction(self, gameState):
        # Pick Action

        legalActions = gameState.getLegalActions(self.index)

        if self.numTraining > 0:
            random_action = util.flipCoin(self.epsilon)
        else:
            random_action = False

        action = self.computeActionFromQValues(gameState)

        return random.choice(legalActions) if random_action else action

    def chooseAction(self, gameState):
        if not self.teamsRegistered:
            self.setInitialDistributions(gameState)
            self.teamsRegistered = True
        else:
            self.updateEnemyDistributions(gameState)

        action = self.findOptimalAction(gameState)

        # reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
        # reward = self.getScore(gameState) - self.getScore(self.lastState)
        # self.update(self.lastState, action, gameState, reward)

        # Was getting invalid moves as it was trying to use this states action on lastGame state during  observeTransition
        if self.numTraining > 0:
            self.observation(gameState)
        # self.observeTransition(self.lastState, action, gameState, reward)

        self.lastState = gameState
        self.lastAction = action

        #self.debugBelieveSystem()

        return action

    def getWeights(self):
        self.weights['distanceToEnemyPacMan'] = -100

        return self.weights

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return Counter(self.weights) * Counter(features)

    # TODO this can be better

    def getCustomScore(self, state):
        foodRemaining = len(self.getFood(state).asList())
        defend = len(self.getFoodYouAreDefending(state).asList())

        score = self.getScore(state) - self.getScore(self.getPreviousObservation())

        pickedUpFood = 1 if state.getAgentState(
            self.index).numCarrying - self.getPreviousObservation().getAgentState(self.index).numCarrying > 0 else 0
        timeLeft = state.data.timeleft / 100.0

        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # if len(ghost) > 0:
        #     dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghost]

        ghostInVision = -len(ghost)

        custom_score = ((10 * score + timeLeft) + pickedUpFood) + ghostInVision

        return custom_score

    def getFeatures(self, state, action):
        features = util.Counter()

        successor = self.getSuccessor(state, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()

        # TODO need to add check for if blue team as we want to lose points
        features['scoredPoints'] = self.getScore(successor) / 100.0 if self.getScore(successor) - self.getScore(
            state) > 0 else 0
        features['foodEaten'] = 1.0 if successor.getAgentState(self.index).numCarrying - state.getAgentState(
            self.index).numCarrying > 0 else 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        enemyPacMan = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['ghostDistance'] = (min(dists) / 100.0)

        if len(enemyPacMan) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemyPacMan]
            features['enemyPacManDistance'] = (min(dists) / 100.0)

        if len(foodList) > 0 and features['scoredPoints'] == 0 and features[
            'foodEaten'] == 0:  # This should always be True,  but better safe than sorry
            # myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -(minDistance / 100.0)

        # TODO add distance to friendly if can see don't move towards it unless it can see a ghost
        # TODO add if we have 5 food return
        if state.getAgentState(self.index).isPacman and features['distanceToFood'] < -2 / 100.0:
            middle = self.getFood(state).width / 2
            borderLine = [(middle, y) for y in range(self.getFood(state).height) if
                          not state.hasWall(middle, y)]

            if not self.red:
                xRange = range(middle)
            else:
                xRange = range(middle, self.getFood(state).width)

            for x in xRange:
                for y in range(self.getFood(state).height):
                    if not self.getFood(state)[x][y]:
                        borderDistances = min(
                            self.getMazeDistance(myPos, borderPos) for borderPos in borderLine)

                        features['foodICanReturn'] = ((-borderDistances * successor.getAgentState(
                            self.index).numCarrying) / 100.0)

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

        if not self.lastState is None:
            reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
            self.observeTransition(self.lastState, self.lastAction, gameState, reward)

        return gameState

    def terminal(self, state):
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

#        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
#            print 'Reinforcement Learning Status:'
#            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
#            if self.episodesSoFar <= self.numTraining:
#                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
#                print '\tCompleted %d out of %d training episodes' % (
#                    self.episodesSoFar, self.numTraining)
#                print '\tAverage Rewards over all training: %.2f' % (
#                    trainAvg)
#            else:
#                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
#                print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
#                print '\tAverage Rewards over testing: %.2f' % testAvg
#            print '\tAverage Rewards for last %d episodes: %.2f' % (
#                NUM_EPS_UPDATE, windowAvg)
#            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
#            self.lastWindowAccumRewards = 0.0
#            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))

    def final(self, state):
        # call the super-class final method
        self.terminal(state)

        if self.saveWeights:
            with open(self.savePath, 'wb') as f:
                pickle.dump(self.weights, f, pickle.HIGHEST_PROTOCOL)
        if self.episodesSoFar == self.numTraining:
            print("episode equals numTraining, begin testing afterwards")


class DefensiveAgent(DummyAgent):

    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)
        self.teamsInitialPosition[self.index] = gameState.getAgentState(self.index).getPosition()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        if not self.teamsRegistered:
            self.setInitialDistributions(gameState)
            self.teamsRegistered = True
        else:
            self.updateEnemyDistributions(gameState, True)

        actions = gameState.getLegalActions(self.index)

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

        # TODO I think distances are in order index 0, 1, 2, 3

        # self.debugDraw((agentDistances[0], ), [1,0,0])
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['defendFood'] = len(self.getFoodYouAreDefending(successor).asList())

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        ## go on the attack
        # print features['numInvaders']
        # if features['numInvaders'] == 0 :
        #     successor = self.getSuccessor(gameState, action)
        #     foodList = self.getFood(successor).asList()
        #     features['successorScore'] = -len(foodList)  # self.getScore(successor)
        #
        #     # Compute distance to the nearest food
        #
        #     if len(foodList) > 0:  # This should always be True,  but better safe than sorry
        #         myPos = successor.getAgentState(self.index).getPosition()
        #         minDistance = -min([self.getMazeDistance(myPos, food) for food in foodList])
        #         features['distanceToFood'] = minDistance
        #
        #
        # else:
        if not myState.isPacman and features['numInvaders'] == 0:
            grid = gameState.getWalls()
            halfway = grid.width / 2
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
                        # print (-borderDistances * successor.getAgentState(self.index).numCarrying) / 100.0
                        features['boarderDistance'] = (borderDistances / 100.0)

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'boarderDistance': -2, 'defendFood': 10}
