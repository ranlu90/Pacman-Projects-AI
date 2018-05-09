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
               first='ApproximateQAgent', second='ApproximateQAgent', numTraining=0):
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
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)


class ApproximateQAgent(CaptureAgent):

    def __init__(self, index, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        self.lastState = gameState
        if self.episodesSoFar == 0:
            print 'Beginning %d episodes of Training' % (self.numTraining)

    def computeActionFromQValues(self, state):
        legal_actions = state.getLegalActions(self.index)
        successor_q_values = {action: self.getQValue(state, action) for action in legal_actions}
        max_action = max(successor_q_values.iterkeys(), key=(lambda key: successor_q_values[key]))
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
        return random.choice(legalActions) if random_action else action

    def chooseAction(self, gameState):
        action = self.findOptimalAction(gameState)
        reward = self.getCustomScore(gameState) - self.getCustomScore(self.lastState)
        self.update(self.lastState, action, gameState, reward)
        self.lastState = gameState
        self.lastAction = action
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return Counter(self.weights) * Counter(features)

    def getCustomScore(self, state):
        food = len(self.getFood(state).asList())
        defend = len(self.getFoodYouAreDefending(state).asList())
        score = self.getScore(state)

        custom_score = 10*score - food + defend

        return custom_score

    def getFeatures(self, state, action):
        features = util.Counter()

        features['food'] = len(self.getFood(state).asList())
        features['defending'] = len(self.getFoodYouAreDefending(state).asList())
        features['score'] = self.getScore(state)
        # self.getCapsules(state)
        # self.getOpponents(state)
        # self.getCurrentObservation()

        return features

    def update(self, gameState, action, nextState, reward):
        features = self.getFeatures(gameState, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(gameState,
                                                                                                         action)
        self.weights = {k: self.weights.get(k, 0) + self.alpha * difference * features.get(k, 0) for k in
                        set(self.weights) | set(features)}

        print self.weights

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

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
        if self.episodesSoFar == self.numTraining:
            print(self.weights)
