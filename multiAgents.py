# multiAgents.py
# --------------


from util import manhattanDistance
from game import Directions
import random, util
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        
        if successorGameState.isWin():
            return math.inf
        elif successorGameState.isLose():
            return -math.inf
        
        
        foodDist = [manhattanDistance(successorGameState.getPacmanPosition(), p) for p in successorGameState.getFood().asList()]
        
        evalu = 0
        
       
        if action == Directions.STOP:
            
            evalu -= 10

        if len(successorGameState.getFood().asList()) < len(currentGameState.getFood().asList()):
            evalu += 150
               
            
            
        return int(evalu - min(foodDist))

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        depth = self.depth #tope de expansión
        maxAgentes = gameState.getNumAgents() #numero de agentes
        moves = gameState.getLegalActions(0) #movimientos posibles
        states = [gameState.generateSuccessor(0,move) for move in moves] #estados de esos movimientos
        
        move = moves[0]
        m = -math.inf
        for i in range(len(moves)):
            p = self.value(states[i], depth, 1, maxAgentes)
            if p > m:
                m = p
                move = moves[i]
        return move
        
    def value(self, state, depth, agent, maxAgents):
        
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        if agent:
            temp = math.inf
        else:
            temp = -math.inf
        
        states = [state.generateSuccessor(agent, move) for move in state.getLegalActions(agent)]
        
        for succ_state in states:
            if agent:
                if agent+1 == maxAgents:
                    temp = min(temp, self.value(succ_state, depth-1, 0, maxAgents))
                else:
                    temp = min(temp, self.value(succ_state, depth, agent+1, maxAgents))
            else:
                temp = max(temp, self.value(succ_state, depth, agent+1, maxAgents))
                
        return temp

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth #tope de expansión
        maxAgentes = gameState.getNumAgents() #numero de agentes
        moves = gameState.getLegalActions(0) #movimientos posibles
        states = [gameState.generateSuccessor(0,move) for move in moves] #estados de esos movimientos
        
        move = moves[0]
        m = -math.inf
        alpha = -math.inf
        beta = math.inf
        for i in range(len(moves)):
            p = self.value(states[i], self.depth, 1, maxAgentes, alpha, beta)
            
            if p > m:
                m = p
                move = moves[i]
                
            if m > beta:
                return move
            alpha = max(m,alpha)
        return move


    def value(self, state, depth, agent, maxAgents, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        if agent:
            v = math.inf
        else:
            v = -math.inf
        
        moves = [ move for move in state.getLegalActions(agent)]

        for move in moves:
            if agent:
                if agent+1 == maxAgents:
                    v = min(v, self.value(state.generateSuccessor(agent,move), depth-1, 0, maxAgents,alpha, beta))
                    
                else:
                    v = min(v, self.value(state.generateSuccessor(agent,move), depth, agent+1, maxAgents,alpha, beta))
                    
                if v < alpha:
                    return v
                beta = min(v, beta)
                
            else:
                v = max(v, self.value(state.generateSuccessor(agent,move), depth, agent+1, maxAgents, alpha, beta))
                if v > beta:
                    return v
                alpha = max(v,alpha)       
        return v
    

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
        depth = self.depth #tope de expansión
        maxAgentes = gameState.getNumAgents() #numero de agentes
        moves = gameState.getLegalActions(0) #movimientos posibles
        states = [gameState.generateSuccessor(0,move) for move in moves] #estados de esos movimientos
        
        move = moves[0]
        m = -math.inf
        for i in range(len(moves)):
            p = self.value(states[i], depth, 1, maxAgentes)
            if p > m:
                m = p
                move = moves[i]
        return move
       
    def value(self, state, depth, agent, maxAgents):
        
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
    
        if agent:
            temp = 0
        else:
            temp = -math.inf
        
        states = [state.generateSuccessor(agent, move) for move in state.getLegalActions(agent)]
        
        for succ_state in states:
            if agent:
                if agent+1 == maxAgents:
                    temp += 1/len(states) * self.value(succ_state, depth-1, 0, maxAgents)
                else:
                    temp += 1/len(states) * self.value(succ_state, depth, agent+1, maxAgents)
            else:
                temp = max(temp, self.value(succ_state, depth, agent+1, maxAgents))
                
        return temp
        
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #datos extraibles del estado
    p = currentGameState.getPacmanPosition() 
    f = currentGameState.getFood() 
    g = currentGameState.getGhostStates() 
    ncaps = len(currentGameState.getCapsules())
    #datos derivados de datos extraidos
    fd = [manhattanDistance(p, x) for x in f.asList()]
    gp = [x.getPosition() for x in g]
    gd = [manhattanDistance(p, x) for x in gp]
    sc = [ghostState.scaredTimer for ghostState in g] 
    nf = len(fd) 
    ssc = sum(sc)
    sgd = sum (gd)
    
    
    reciprocalfoodDistance = 1.0 / (sum(fd) + 0.00001)
        
    ret = currentGameState.getScore()  + 1 / (sum(fd) + 0.00001) + nf

    if ssc > 0:    
        
        ind = gd.index(min(gd))
        
        if sc[ind] > 0:
            ret +=   ssc + ncaps - sgd
        else:
            ret +=   -ssc + ncaps + sgd
    else :
        
        ret +=  sgd - ncaps 
        
    return ret



# Abbreviation
better = betterEvaluationFunction
