from .Agent import Agent

# An agent that searches through the game tree over a certain depth with alpha-beta pruning.
class MinimaxAgent(Agent):
	def __init__(self, depth):
		self.depth = depth

	def ab(self, game, depth, a, b):
		turn = game.turn
		best_action = None

		if depth == 0 or game.is_over():
			return (game.score('x') - game.score('y'), best_action)

		value = 0
		if turn == 'x':
			value = -99
			for action in game.actions():
				next_game = game.action(action)
				next_value = self.ab(next_game, depth - 1, a, b)[0]
				if value < next_value:
					value = next_value
					best_action = action
				if value >= b: break
				a = max(a, value)
		else:
			value = 99
			for action in game.actions():
				next_game = game.action(action)
				next_value = self.ab(next_game, depth - 1, a, b)[0]
				if value > next_value:
					value = next_value
					best_action = action
				if value <= a: break
				b = min(b, value)

		return (value, best_action)

	def policy(self, game):
		return self.ab(game, self.depth, -99, 99)[1]
