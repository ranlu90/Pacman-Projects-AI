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
from game import Directions
import game
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

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

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




class ApproximateQAgent(DummyAgent):

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


    def __init__(self, index, epsilon=0.2, gamma=0.2, alpha=0.2, numTraining=95, **args):

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

        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0



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
        CaptureAgent.registerInitialState(self, gameState)
        #self.lastState = gameState
        if self.episodesSoFar == 0:
            print 'Beginning %d episodes of Training' % (self.numTraining)

    def computeActionFromQValues(self, state):
        legal_actions = state.getLegalActions(self.index)
        successor_q_values = {action: self.getQValue(state, action) for action in legal_actions}
        max_action = max(successor_q_values.iterkeys(), key=(lambda key: successor_q_values[key]))

        #print "successor_q_values", successor_q_values
        return max_action if legal_actions else None

    def computeValueFromQValues(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        successor_q_values = [self.getQValue(gameState, action) for action in legal_actions]
        return max(successor_q_values) if legal_actions else 0.0

    def findOptimalAction(self, gameState):
        # Pick Action

        legalActions = gameState.getLegalActions(self.index)
        random_action = util.flipCoin(self.epsilon)
        action = self.computeActionFromQValues(gameState)
        #print self.epsilon
        #print random_action
        return random.choice(legalActions) if random_action else action

    def chooseAction(self, gameState):
        action = self.findOptimalAction(gameState)

        #reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
        #reward = self.getScore(gameState) - self.getScore(self.lastState)
        #self.update(self.lastState, action, gameState, reward)

        # Was getting invalid moves as it was trying to use this states action on lastGame state during  observeTransition
        self.observation(gameState)
        #self.observeTransition(self.lastState, action, gameState, reward)


        self.lastState = gameState
        self.lastAction = action

        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return Counter(self.weights) * Counter(features)

    #TOD this can be better
    def getCustomScore(self, state):
        food = len(self.getFood(state).asList())
        defend = len(self.getFoodYouAreDefending(state).asList())
        score = self.getScore(state)
        time = state.data.timeleft * 0.01

        custom_score = (10 * score - food + defend + time)

        return custom_score

    def getFeatures(self, state, action):
        features = util.Counter()

        successor = self.getSuccessor(state, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        #features['foodLeft'] = -len(foodList) / 100.0

        #TODO need to add check for if blue team as we want to lose points
        features['scoredPoints'] = self.getScore(successor) / 100.0 if self.getScore(successor) - self.getScore(state) > 0 else 0
        features['foodEaten'] = 1.0 if successor.getAgentState(self.index).numCarrying - state.getAgentState(self.index).numCarrying > 0 else 0
        #features['timeLeft'] =  state.data.timeleft /10000.0
        # Compute distance to the nearest food

        # Doesnt seem to do much
        # features['killedWithFood'] = (state.getAgentState(self.index).numCarrying - successor.getAgentState(self.index).numCarrying)

        #features['distanceToGhost']

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghost = [a for a in enemies if  not a.isPacman and a.getPosition() != None]
        if len(ghost) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghost]
            features['ghostDistance'] = min(dists) / 100.0
        # features['numInvaders'] = len(invaders)
        # if len(invaders) > 0:
        #     dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        #     features['invaderDistance'] = min(dists)
        #
        #myPos = successor.getAgentState(self.index).getPosition()
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            #myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance / 100.0


        # features['distanceToBoarder'] = state.data.borderDistances[myPos] if myPos in state.data.borderDistances else 0
        if state.getAgentState(self.index).isPacman:
            halfway = self.getFood(state).width / 2
            borderPositions = [(halfway, y) for y in range(self.getFood(state).height) if not state.hasWall(halfway, y)]

            grid = state.getWalls()
            halfway = grid.width / 2
            if not self.red:
                xrange = range(halfway)
            else:
                xrange = range(halfway, grid.width)

            for x in xrange:
                for y in range(grid.height):
                    if not grid[x][y]:
                        borderDistances = min(
                            self.getMazeDistance(myPos, borderPos) for borderPos in borderPositions)
                        #print (-borderDistances * successor.getAgentState(self.index).numCarrying) / 100.0
                        features['riskOfLoosingFood'] = (-borderDistances * successor.getAgentState(self.index).numCarrying) / 1000.0

        #
        if self.distancer.getDistance(myPos, state.getAgentState(self.index).getPosition()) > 1.0:

            print self.distancer.getDistance(myPos, state.getAgentState(self.index).getPosition())
            features['died'] = -1.0
            features['riskOfLoosingFood'] = -1.0
            #features['died'] = -100 if myPos - state.getAgentState(self.index).getPosition() > 2 else 0
        #print - 100 if myPos - state.getAgentState(self.index).getPosition() > 2 else 0

        if action == Directions.STOP: features['stop'] = -1.0
        return features




        # features['foodDefending'] = len(self.getFoodYouAreDefending(state).asList())
        # features['score'] = self.getScore(state)
        # self.getCapsules(state)
        # self.getOpponents(state)
        # self.getCurrentObservas)



    def update(self, gameState, action, nextState, reward):

        features = self.getFeatures(gameState, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(gameState,
                                                                                                         action)
        self.weights = {k: self.weights.get(k, 0) + self.alpha * difference * features.get(k, 0) for k in
                        set(self.weights) | set(features)}
        #print self.weights
        #print features

    def observeTransition(self, state, action, nextState, deltaReward):


        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def observation(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """

        if not self.lastState is None:
            reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
            #reward = gameState.getScore() - self.lastState.getScore()
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

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print 'Reinforcement Learning Status:'
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print '\tCompleted %d out of %d training episodes' % (
                    self.episodesSoFar, self.numTraining)
                print '\tAverage Rewards over all training: %.2f' % (
                    trainAvg)
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
                print '\tAverage Rewards over testing: %.2f' % testAvg
            print '\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, windowAvg)
            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))

    def final(self, state):
        # call the super-class final method
        self.terminal(state)
        print(self.weights)
        if self.episodesSoFar == self.numTraining:
            print(self.weights)


class DefensiveAgent(DummyAgent):



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

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

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



