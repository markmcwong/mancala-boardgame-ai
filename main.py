from Simulator import Simulator
from Agent import *

# Start a simulator that lets me play with an AI that chooses a random action.
p1 = HumanAgent()
p2 = MinimaxAgent(9)
simulator = Simulator(p1, p2)
simulator.start()
