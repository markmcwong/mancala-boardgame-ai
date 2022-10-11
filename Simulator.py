from Game import Game

class Simulator:
	def __init__(self, agent_x, agent_y):
		self.game = Game()
		self.agent_x = agent_x
		self.agent_y = agent_y

	def start(self):
		while not self.game.is_over():
			game = self.game
			print()
			print(game.ascii())
			print(f"Turn: {game.turn.upper()}")

			move = -1
			if game.turn == 'x':
				move = self.agent_x.policy(game)
			else:
				move = self.agent_y.policy(game)

			self.game = game.move(move)
			print(f"Move chosen: {move}")
