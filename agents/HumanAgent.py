from .Agent import Agent

# An agent that determines its best action from an actual person.
class HumanAgent(Agent):
	def policy(self, game):
		action = input("Choose a house (0 to 5): ")
		return int(action)
