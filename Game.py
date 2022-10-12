class Game:
	# There are two players in a game, X and Y.
	# At the start, each player has 6 houses with 4 seeds each and an empty store.
	# Index 0 to 6 is owned by X, 7 to 13 is owned by Y.
	# Index 6 and 13 are X's store and Y's store respectively.
	# `self.turn` can be either 'x' or 'y' and we assume X goes first.
	def __init__(self, board = [4, 4, 4, 4, 4, 4, 0] * 2, turn = 'x'):
		self.board = board
		self.turn = turn

	# The criteria for the game to end is if either player's house has no more seeds.
	def is_over(self):
		return sum(self.board[0:6]) * sum(self.board[7:13]) == 0

	# `i` can take the value of 0, 1, 2, 3, 4, 5.
	# Each value represents the index of the current player's house.
	# This method returns a new state of the Game object after the action is played.
	def action(self, i):
		board = list(self.board)
		turn = self.turn

		if turn == 'y': i += 7

		seeds = board[i]
		if self.is_over() or seeds == 0: return self

		board[i] = 0
		while seeds > 0:
			i = (i + 1) % 14
			if turn == 'x' and i == 13: i = 0
			if turn == 'y' and i == 6: i = 7

			board[i] += 1
			seeds -= 1

		if turn == 'x':
			if 0 <= i and i <= 5 and board[i] == 1 and board[12 - i] > 0:
				board[6] += board[i] + board[12 - i]
				board[i] = 0
				board[12 - i] = 0
				turn = 'y'
			elif i == 6:
				turn = 'x'
			else:
				turn = 'y'
		else:
			if 7 <= i and i <= 12 and board[i] == 1 and board[12 - i] > 0:
				board[13] += board[i] + board[12 - i]
				board[i] = 0
				board[12 - i] = 0
				turn = 'x'
			elif i == 13:
				turn = 'y'
			else:
				turn = 'x'

		return Game(board, turn)

	def actions(self, turn):
		if self.is_over(): return []

		if turn == 'x':
			return [i for i in range(6) if self.board[i] > 0]
		else:
			return [i for i in range(6) if self.board[i + 7] > 0]

	def score(self, turn):
		return self.board[6] if turn == 'x' else self.board[13]

	# Returns the ASCII art representation of the board
	def ascii(self):
		scores = [str(score).zfill(2) for score in self.board]
		return (
			"┌────┬────┬────┬────┬────┬────┬────┬────┐\n" +
			"│    │ {} │ {} │ {} │ {} │ {} │ {} │    │\n".format(*scores[12:6:-1]) +
			"│ {} ├────┼────┼────┼────┼────┼────┤ {} │\n".format(scores[13], scores[6]) +
			"│    │ {} │ {} │ {} │ {} │ {} │ {} │    │\n".format(*scores[0:6]) +
			"└────┴────┴────┴────┴────┴────┴────┴────┘"
		)
