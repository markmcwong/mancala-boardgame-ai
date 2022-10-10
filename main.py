from Simulator import Simulator
from Agent import HumanAgent, RandomAgent

# Start a simulator that lets me play with an AI that chooses a random action.
me = HumanAgent()
ai = RandomAgent()
simulator = Simulator(me, ai)
simulator.start()
