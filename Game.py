class Game:
	# There are two players in a game, X and Y.
	# At the start, each player has 6 houses with 4 seeds each and an empty store.
	# Index 0 to 6 is owned by X, 7 to 13 is owned by Y.
	# Index 6 and 13 are X's store and Y's store respectively.
	# `self.turn` can be either 'x' or 'y' and we assume X goes first.
	def __init__(self):
		self.board = [4, 4, 4, 4, 4, 4, 0] * 2
		self.turn = 'x'

	# The criteria for the game to end is if either player's house has no more seeds.
	def is_game_over(self):
		return sum(self.board[0:6]) * sum(self.board[7:13]) == 0

	# `i` can take the value of 0, 1, 2, 3, 4, 5.
	# Each value represents the index of the current player's house.
	# This method returns either 'x' or 'y', indicating the next player's turn.
	def move(self, i):
		if self.turn == 'y': i += 7

		seeds = self.board[i]
		if self.is_game_over() or seeds == 0:
			return self.turn

		self.board[i] = 0
		while seeds > 0:
			i = (i + 1) % 14
			if self.turn == 'x' and i == 13: i = 0
			if self.turn == 'y' and i == 6: i = 7

			self.board[i] += 1
			seeds -= 1

		if self.turn == 'x':
			if 0 <= i and i <= 5 and self.board[i] == 1 and self.board[12 - i] > 0:
				self.board[6] += self.board[i] + self.board[12 - i]
				self.board[i] = 0
				self.board[12 - i] = 0
				self.turn = 'y'
			elif i == 6:
				self.turn = 'x'
			else:
				self.turn = 'y'
		else:
			if 7 <= i and i <= 12 and self.board[i] == 1 and self.board[12 - i] > 0:
				self.board[13] += self.board[i] + self.board[12 - i]
				self.board[i] = 0
				self.board[12 - i] = 0
				self.turn = 'x'
			elif i == 13:
				self.turn = 'y'
			else:
				self.turn = 'x'

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
