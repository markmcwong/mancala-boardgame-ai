from Simulator import Simulator
from Agent import *

# Start a simulator that lets me play with an AI that chooses a random action.
me = HumanAgent()
ai = GreedyAgent()
simulator = Simulator(me, ai)
simulator.start()
