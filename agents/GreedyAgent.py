from .Agent import Agent

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
