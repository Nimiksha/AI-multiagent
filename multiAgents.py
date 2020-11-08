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


import random
import util

from game import Agent
from util import manhattanDistance


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        "*** YOUR CODE HERE ***"
        # get the possible position of the ghost in the successor state
        ghostPosition = successorGameState.getGhostPositions()
        # based on the ghost position found, calculate distance between pacman and ghost
        ghostDistance = [manhattanDistance(newPos, newGhostPos) for newGhostPos in ghostPosition]
        # calculate distance between pacman and food position in the next state
        foodDistance = [manhattanDistance(food, newPos) for food in newFood.asList()]
        # initialize score based on the number of food dots available in the state
        score = -len(newFood.asList())
        # Updating score based on the distance to the available food
        if len(foodDistance) > 0:
            # used reciprocal for features as suggested in the question
            # update score (+ve) based on the closest food dot
            score += 1. / (min(foodDistance) + 1) + 1. / (sum(foodDistance) + 1)
        else:
            score += 1.

        # update score based on the distance between ghost and pacman
        if len(ghostDistance) > 0:
            # update score (-ve) based on the distance to the ghost (pacman loses points as it gets closer to the ghost)
            score += -2. / (min(ghostDistance) + 1)
        else:
            score += -1.
        # update score based on the sum of moves the ghost will remain scared because of the power pellet
        score += 1. / (sum(newScaredTimes) + 1)

        return score
        # return successorGameState.getScore()


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # minimax function to perform appropriate minimizing and maximizing functions, based on the agent
        def minimax(state, agentIndex, depth):

            # checks to see that the game state is terminal or not
            # the last check sees if there is any more depth to explore and if the agent exploring is pacman or not
            if state.isWin() or state.isLose() or (depth == self.depth and agentIndex % state.getNumAgents() == 0):
                return self.evaluationFunction(state), None

            # check if next agent is pacman (agentIndex for pacman is 0) and if it is, maximize the value
            elif agentIndex % state.getNumAgents() == 0:
                agent = 0  # set to 0 since agent is pacman
                availableLegalActions = state.getLegalActions(agent)
                value = -999999     # arbitrarily set large negative value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # newValue and newAction state is the successor state, we add one to the agent to transfer
                        # control from pacman to ghost, and add 1 to depth since we've move on to the next layer
                        newValue, nextAction = minimax(state.generateSuccessor(agent, legalAction), agent + 1,
                                                       depth + 1)
                        # if the new value is maximum, update the value and action
                        if newValue > value:
                            value = newValue
                            action = legalAction
                    return value, action
                else:
                    return self.evaluationFunction(state), None

            # else if the next agent is >= 1, it's a ghost, minimize the value
            else:
                agent = agentIndex % state.getNumAgents()  # get the index number of the agent
                availableLegalActions = state.getLegalActions(agent)
                value = 999999       # arbitrarily set large positive value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # newValue and newAction state is the successor state and we add one to the agent to transfer
                        # control from oen agent to another
                        newValue, nextAction = minimax(state.generateSuccessor(agent, legalAction), agent + 1, depth)
                        # if the new value is less than the existing value (since we are minimizing),
                        # update the value and action
                        if newValue < value:
                            value = newValue
                            action = legalAction
                    return value, action
                else:
                    return self.evaluationFunction(state), None

        return minimax(gameState, 0, 0)[1]
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # alphaBetaMinimax function to perform appropriate and necessary minimizing and maximizing functions,
        # based on the agent (pretty similar to minimax)
        def alphabetaminimax(state, agentIndex, depth, alpha, beta):
            # checks to see that the game state is terminal or not
            # the last check sees if there is any more depth to explore and if the agent exploring is pacman or not
            if state.isWin() or state.isLose() or (depth == self.depth and agentIndex % state.getNumAgents() == 0):
                return self.evaluationFunction(state), None

            # check if next agent is pacman (agentIndex for pacman is 0) and if it is, maximize the value
            elif agentIndex % state.getNumAgents() == 0:
                agent = 0  # set to 0 since agent is pacman
                availableLegalActions = state.getLegalActions(agent)
                value = -99999        # arbitrarily set large negative value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # newValue and newAction state is the successor state, we add one to the agent to transfer
                        # control from pacman to ghost, and add 1 to depth since we've move on to the next layer
                        newValue, nextAction = alphabetaminimax(state.generateSuccessor(agent, legalAction), agent + 1,
                                                                depth + 1, alpha, beta)
                        # if the new value is maximum, update the value and action
                        if newValue > value:
                            value = newValue
                            action = legalAction
                            # if the newValue is greater than the beta value, then prune and return the value and action
                            if newValue > beta:
                                return value, action
                            # update the value of alpha given alpha is max's best option on path to root
                            alpha = max(alpha, value)
                    return value, action
                else:
                    return self.evaluationFunction(state), None

            # else if the next agent is >= 1, it's a ghost, minimize the value
            else:
                agent = agentIndex % state.getNumAgents()  # get the index number of the agent
                availableLegalActions = state.getLegalActions(agent)
                value = 99999       # arbitrarily set large positive value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # newValue and newAction state is the successor state and we add one to the agent to transfer
                        # control from oen agent to another
                        newValue, newAction = alphabetaminimax(state.generateSuccessor(agent, legalAction), agent + 1,
                                                               depth, alpha, beta)
                        # if the new value is less than the existing value (since we are minimizing),
                        # update the value and action
                        if newValue < value:
                            value = newValue
                            action = legalAction
                            # if the newValue is less than the alpha value, then prune and return the value and action
                            if newValue < alpha:
                                return value, action
                            # update the value of beta given beta is min's best option on path to root
                            beta = min(value, beta)
                    return value, action
                else:
                    return self.evaluationFunction(state), None

        return alphabetaminimax(gameState, 0, 0, -99999, 99999)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # expectimax function
        def expectimax(state, agentIndex, depth):

            # checks to see that the game state is terminal or not
            # the last check sees if there is any more depth to explore and if the agent exploring is pacman or not
            if state.isWin() or state.isLose() or (depth == self.depth and agentIndex % state.getNumAgents() == 0):
                return self.evaluationFunction(state), None

            # check if next agent is pacman (agentIndex for pacman is 0) and if it is, maximize the value
            elif agentIndex % state.getNumAgents() == 0:
                agent = 0  # set to 0 since agent is pacman
                availableLegalActions = state.getLegalActions(agent)
                value = -99999       # arbitrarily set large negative value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # newValue and newAction state is the successor state, we add one to the agent to transfer
                        # control from pacman to ghost, and add 1 to depth since we've move on to the next layer
                        newValue, nextAction = expectimax(state.generateSuccessor(agent, legalAction), agent + 1,
                                                          depth + 1)
                        # if the new value is maximum, update the value and action pair
                        if newValue > value:
                            value = newValue
                            action = legalAction
                    return value, action
                else:
                    return self.evaluationFunction(state), None

            # else if the next agent is >= 1, it's a ghost, find the expected value
            else:
                agent = agentIndex % state.getNumAgents()  # get the index number of the agent
                availableLegalActions = state.getLegalActions(agent)
                value = 0        # the value is set to 0 instead of an arbitrarily large value
                action = None

                # while there are legal actions available for the agent
                if len(availableLegalActions) != 0:
                    # iterate through the list of actions and get the value and action for the successor
                    for legalAction in availableLegalActions:
                        # update the value by recursively calling the expectimax()
                        value += expectimax(state.generateSuccessor(agent, legalAction), agent + 1, depth)[0]
                    # return value and action.
                    # the value returned by chance node is the average of all available actions
                    return (value / len(availableLegalActions)), action
                else:
                    return self.evaluationFunction(state), None

        return expectimax(gameState, 0, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # information extracted from game state
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    # calculate distance between pacman and food position in the game state
    foodDistance = [manhattanDistance(food, newPos) for food in newFood.asList()]
    # get list of remaining power pellets in the game
    numberOfPowerPellets = currentGameState.getCapsules()
    # calculate distance between pacman and power pellets
    capsuleDistance = [manhattanDistance(newPos, pellet) for pellet in numberOfPowerPellets]

    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # get the possible position of the ghost in the game state
    ghostPosition = currentGameState.getGhostPositions()
    # based on the ghost position found, calculate distance between pacman and ghost
    ghostDistance = [manhattanDistance(newPos, newGhostPos) for newGhostPos in ghostPosition]

    # initialize score based on getScore()
    score = currentGameState.getScore()

    if len(foodDistance) > 0:
        # used reciprocal for features as suggested in the question
        # update score (+ve) based on the closest food dot
        score += 1. / (min(foodDistance) + 1) + 1. / (sum(foodDistance) + 1)
    else:
        score += 1.

        # update score based on the distance between ghost and pacman
        if len(ghostDistance) > 0:
            # update score (-ve) based on the distance to the ghost (pacman loses points as it gets closer to the ghost)
            score += -2. / (min(ghostDistance) + 1)
        else:
            score += -1.

        # update score based on the sum of moves the ghost will remain scared because of the power pellet
        score += 1. / (sum(newScaredTimes) + 1)

    # if there are power pellets available in the game, update the score based on the
    # reciprocal of the minimum distance to the remaining pellets.
    if numberOfPowerPellets:
        score += 1. / min(capsuleDistance)

    return score

# Abbreviation
better = betterEvaluationFunction
