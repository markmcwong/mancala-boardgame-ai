from .Agent import Agent
from math import log, sqrt

# An agent that performs the Monte Carlo Tree Search algorithm.
class MCTSAgent(Agent):
	def __init__(self, playouts):
		self.balancing_factor = sqrt(2)
		self.dag_trials = dict()
		self.dag_wins = dict()
		self.playouts = playouts

	def encode(self, game):
		board = game.board[0:6] + game.board[7:13]
		return "".join([chr(b + 48) for b in board])

	def simulate(self, game):
		encoded_game = self.encode(game)
		if encoded_game not in self.dag_trials:
			self.dag_trials[encoded_game] = [1] * 6
			self.dag_wins[encoded_game] = [1] * 6

		if game.is_over():
			winner = None
			if game.score('x') > game.score('y'): winner = 'x'
			if game.score('x') < game.score('y'): winner = 'y'

			return winner

		best_action = None
		best_value = float("-inf")
		for action in game.actions():
			wins = self.dag_wins[encoded_game][action]
			trials = self.dag_trials[encoded_game][action]
			q_n_a = 0 if trials == 0 else wins / trials

			value = q_n_a + self.balancing_factor * sqrt(log(sum(self.dag_trials[encoded_game])) / trials)
			if value > best_value:
				best_action = action
				best_value = value

		winner = self.simulate(game.action(best_action))
		self.dag_trials[encoded_game][best_action] += 1
		if winner == game.turn:
			self.dag_wins[encoded_game][best_action] += 1

		return winner

	def policy(self, game):
		self.dag_trials = dict()
		self.dag_wins = dict()

		for _ in range(self.playouts):
			self.simulate(game)

		encoded_game = self.encode(game)
		best_action = None
		best_value = float("-inf")
		for action in game.actions():
			wins = self.dag_wins[encoded_game][action]
			trials = self.dag_trials[encoded_game][action]
			q_n_a = 0 if trials == 0 else wins / trials

			value = q_n_a + self.balancing_factor * sqrt(log(sum(self.dag_trials[encoded_game])) / trials)
			if value > best_value:
				best_action = action
				best_value = value

		return best_action
