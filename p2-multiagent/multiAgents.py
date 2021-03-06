# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        expectedValue = 0
        foodList = currentGameState.getFood()
        # If the successor position after one Pacman move is in the food list, increments the value by 100
        if foodList[newPos[0]][newPos[1]]:
            expectedValue += 100

        # Declare both distances to be infinite values as the comparision are close
        minFoodDistance = float('inf')
        minGhostDistance = float('inf')

        # Search all foods in the successor state to find the minimum distance
        for foodPosition in newFood.asList():
            foodDistance = manhattanDistance(newPos, foodPosition)
            minFoodDistance = min(minFoodDistance, foodDistance)

        # Search all ghosts to find the minimum ghost distance
        for ghostPosition in successorGameState.getGhostPositions():
            ghostDistance = manhattanDistance(newPos, ghostPosition)
            minGhostDistance = min(minGhostDistance, ghostDistance)

        # The game ends if Pacman gets hit by ghosts, hence decreases the value
        # to get a smaller probability of Pacman's move
        # Assume 2000 is the upper bound Pacman can get after endgame
        if minGhostDistance < 2:
            expectedValue -= 2000

        walls = currentGameState.getWalls()
        # Use the reciprocal of total length as the weighted factor
        length = walls.height - 2 + walls.width - 2

        # Use the score if Pacman will eat a food, the reciprocal of distance to food and
        # minGhostDistance divided total length as overall value
        expectedValue = expectedValue + 1.0 / minFoodDistance + minGhostDistance / length
        return expectedValue


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # Go through the tree and get the utility of terminal state for Pacman
    def getMaxUtility(self, gameState, depth):
        utility = float('-inf')
        # If the search has reached any leaves or game ends return the utility
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # Use recursive getMaxUtility function with new agent ghost at index 1
        # Get the maximum utility of successor state
        for action in gameState.getLegalActions(0):
            utility = max(utility, self.getMinUtility(gameState.generateSuccessor(0, action), 1, depth))
        return utility

    # Go through the tree and get the utility of terminal state for all ghost agents
    def getMinUtility(self, gameState, agentIndex, depth):
        utility = float('inf')
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # if the ghost is the last one in all ghosts, go through successor states to get minimum utility
        if agentIndex == gameState.getNumAgents() - 1:
            # Use recursive getMaxUtility function of Pacman with depth + 1
            # Get the minimum utility of successor state
            for action in gameState.getLegalActions(agentIndex):
                utility = min(utility, self.getMaxUtility(gameState.generateSuccessor(agentIndex, action), depth + 1))
        # If the ghost isn't the last ghost, loop through getMinUtility function
        # and get minimum utility of all ghosts
        else:
            for action in gameState.getLegalActions(agentIndex):
                utility = min(utility, self.getMinUtility(gameState.generateSuccessor(agentIndex, action),
                                                     agentIndex + 1, depth))
        return utility

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        # The initial move before receiving any successor utilities
        initialMove = Directions.STOP
        maxValue = float('-inf')

        # Get the maximum value from all Min players
        for action in gameState.getLegalActions(0):
            # Get the minimum utility received from ghosts at index 1
            value = self.getMinUtility(gameState.generateSuccessor(0, action), 1, 0)
            if value > maxValue:
                maxValue = value
                initialMove = action

        return initialMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    # Get the maximum utility of Pacman
    def getMaxUtility(self, gameState, depth, alpha, beta):

        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # The default utility is negative infinite
        utility = float('-inf')

        for action in gameState.getLegalActions(0):
            utility = max(utility, self.getMinUtility(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
            # Cut off the branch if it's already greater than beta
            if utility > beta:
                return utility
            alpha = max(alpha, utility)
        return utility

    # Get the minimum utility of all ghosts
    def getMinUtility(self, gameState, agentIndex, depth, alpha, beta):

        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # The default utility is positive infinite
        utility = float('inf')

        # If the agent is the last one in all ghosts
        if agentIndex == gameState.getNumAgents() - 1:
            for action in gameState.getLegalActions(agentIndex):
                utility = min(utility, self.getMaxUtility(gameState.generateSuccessor(agentIndex, action),
                                                          depth + 1, alpha, beta))
                # Cut off the branch if it's already smaller than alpha
                if utility < alpha:
                    return utility
                beta = min(beta, utility)

        # The agent isn't the last one in all ghosts, loop through GetMinUtility to get minimum utility of all ghosts
        else:
            for action in gameState.getLegalActions(agentIndex):
                utility = min(utility, self.getMinUtility(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                          depth, alpha, beta))
                if utility < alpha:
                    return utility
                beta = min(beta, utility)
        return utility

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        initialMove = Directions.STOP

        maxValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            value = self.getMinUtility(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if value > maxValue:
                maxValue = value
                initialMove = action
            # Track alpha after expanding the fist min player and used for the comparision with others on the same level
            alpha = max(alpha, maxValue)

        return initialMove


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Go through the tree and get the utility of terminal state for Pacman
    def getMaxUtility(self, gameState, depth):
        utility = float('-inf')
        # If the search has reached any leaves or game ends return the utility
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # Use recursive getMaxUtility function with new agent ghost at index 1
        # Get the maximum utility of successor state
        for action in gameState.getLegalActions(0):
            utility = max(utility, self.getExpectiUtility(gameState.generateSuccessor(0, action), 1, depth))
        return utility

    # Go through the tree and get the utility of terminal state for all ghost agents
    def getExpectiUtility(self, gameState, agentIndex, depth):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # if the ghost is the last one in all ghosts, go through successor states to get minimum utility
        if agentIndex == gameState.getNumAgents() - 1:
            # Use recursive getMaxUtility function of Pacman with depth + 1
            # Get the minimum utility of successor state
            combinedUtility = 0.0
            for action in gameState.getLegalActions(agentIndex):
                combinedUtility += float(self.getMaxUtility(gameState.generateSuccessor(agentIndex, action), depth + 1))
            utility = combinedUtility / len(gameState.getLegalActions(agentIndex))
        # If the ghost isn't the last ghost, loop through getExpectiUtility function
        # and get average utility of all ghosts
        else:
            combinedUtility = 0.0
            for action in gameState.getLegalActions(agentIndex):
                combinedUtility +=  self.getExpectiUtility(gameState.generateSuccessor(agentIndex, action),
                                                     agentIndex + 1, depth)
            utility = combinedUtility / len(gameState.getLegalActions(agentIndex))

        return utility


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # The initial move before receiving any successor utilities
        initialMove = Directions.STOP
        maxValue = float('-inf')

        # Get the maximum value from all Min players
        for action in gameState.getLegalActions(0):
            # Get the  epectiUtility received from ghosts at index 1
            value = self.getExpectiUtility(gameState.generateSuccessor(0, action), 1, 0)
            if value > maxValue:
                maxValue = value
                initialMove = action

        return initialMove



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
        - keep score consistent
        - maximize distance to ghost
        - minimize distance to food
        - minimize distance to capsule
    """
    lastScore = currentGameState.getScore()
    pacmanState = currentGameState.getPacmanState()

    if currentGameState.isWin():
        return lastScore

    position = pacmanState.getPosition()

    ghostStates = currentGameState.getGhostStates()
    distanceGhosts = min(list(map(lambda x: manhattanDistance(position, x.getPosition()), ghostStates)))

    food = currentGameState.getFood().asList()
    if food:
        distanceFood = distanceGhosts / min(list(map(lambda x: util.manhattanDistance(position, x), food)))
    else:
        distanceFood = 0

    capsules = currentGameState.getCapsules()
    if capsules:
        distanceCapsules =  distanceGhosts / min(list(map(lambda x: util.manhattanDistance(position, x), capsules)))
    else:
        distanceCapsules = 0

    return lastScore + distanceFood + distanceCapsules


# Abbreviation
better = betterEvaluationFunction
