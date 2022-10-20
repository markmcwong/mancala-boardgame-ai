from .Agent import Agent
import random

# An agent that has a policy that simply selects a random action.
class RandomAgent(Agent):
	def policy(self, game):
		return random.choice(game.actions())
