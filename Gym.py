from Game import Game

# A class to train an agent (Agent X) against another agent (Agent Y).
class Gym():
	def __init__(self, agent_x, agent_y):
		self.game = Game()
		self.agent_x = agent_x
		self.agent_y = agent_y

	def start(self):
		while not self.game.is_over():
			game = self.game

			action = -1
			if game.turn == 'x':
				action = self.agent_x.policy(game)
				next_game = game.action(action)
				self.agent_x.learn(game, action, next_game)
				self.game = next_game
			else:
				action = self.agent_y.policy(game)
				self.game = game.action(action)
