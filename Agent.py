import random

# The abstract class for other classes to derive from.
class Agent:
	def policy(self):
		pass

# An agent that has a policy that simply selects a random move.
class RandomAgent(Agent):
	def policy(self, game):
		return random.randint(0, 5)

# An agent that determines its best action from an actual person.
class HumanAgent(Agent):
	def policy(self, game):
		move = input("Choose a house (0 to 5): ")
		return int(move)
