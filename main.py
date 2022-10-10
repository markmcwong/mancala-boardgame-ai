from Game import Game
import random

game = Game()

while not game.is_game_over():
	print(game.ascii())
	print(f"Turn: {game.turn.upper()}")
	if game.turn == 'y':
		game.move(random.randint(0, 5))
	else:
		move = input("Move? (0 to 5) ")
		game.move(int(move))
