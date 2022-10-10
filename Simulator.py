from Game import Game

class Simulator:
	def __init__(self, agent_x, agent_y):
		self.game = Game()
		self.agent_x = agent_x
		self.agent_y = agent_y

	def start(self):
		game = self.game
		while not game.is_game_over():
			print()
			print(game.ascii())
			print(f"Turn: {game.turn.upper()}")

			move = -1
			if game.turn == 'x':
				move = self.agent_x.policy(game.board)
			else:
				move = self.agent_y.policy(game.board)

			game.move(move)
			print(f"Move chosen: {move}")
