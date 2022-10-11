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

# An agent that greedily chooses the best action every turn.
class GreedyAgent(Agent):
	def policy(self, game):
		max_score = -1
		best_action = -1

		turn = game.turn
		for move in range(6):
			if turn == 'x' and game.board[move] == 0: continue;
			if turn == 'y' and game.board[move + 7] == 0: continue;

			next_game = game.move(move)
			if next_game.turn == turn: return move

			next_score = next_game.score(turn)
			if next_score <= max_score: continue

			max_score = next_score
			best_action = move

		return best_action
