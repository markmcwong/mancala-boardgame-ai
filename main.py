from Simulator import Simulator
from agents import *

# Define the two agents that will play the game.
p1 = DeepQLearningAgent()
p2 = MinimaxAgent(9)
simulator = Simulator(p1, p2)
simulator.start()
