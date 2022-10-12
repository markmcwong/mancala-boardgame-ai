import random

# The abstract class for other classes to derive from.
class Agent:
	def policy(self):
		pass

# An agent that has a policy that simply selects a random action.
class RandomAgent(Agent):
	def policy(self, game):
		return random.choice(game.actions())

# An agent that determines its best action from an actual person.
class HumanAgent(Agent):
	def policy(self, game):
		action = input("Choose a house (0 to 5): ")
		return int(action)

# An agent that greedily chooses the best action every turn.
class GreedyAgent(Agent):
	def policy(self, game):
		max_score = -1
		best_action = -1

		turn = game.turn
		for action in game.actions():
			if turn == 'x' and game.board[action] == 0: continue;
			if turn == 'y' and game.board[action + 7] == 0: continue;

			next_game = game.action(action)
			if next_game.turn == turn: return action

			next_score = next_game.score()
			if next_score <= max_score: continue

			max_score = next_score
			best_action = action

		return best_action

# An agent that searches through the game tree over a certain depth with alpha-beta pruning.
class MinimaxAgent(Agent):
	def __init__(self, depth):
		self.depth = depth

	def ab(self, game, depth, a, b):
		turn = game.turn
		if depth == 0 or game.is_over():
			return (game.score('x') - game.score('y'), None)

		value = 0
		if turn == 'x':
			value = -99
			for action in game.actions():
				next_game = game.action(action)
				value = max(value, self.ab(next_game, depth - 1, a, b)[0])
				if value >= b: break
				a = max(a, value)
		else:
			value = 99
			for action in game.actions():
				next_game = game.action(action)
				value = min(value, self.ab(next_game, depth - 1, a, b)[0])
				if value <= a: break
				b = min(b, value)

		return (value, action)

	def policy(self, game):
		return self.ab(game, self.depth, -99, 99)[1]
