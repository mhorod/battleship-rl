from player import *
from game import *

def compare_placer_with_shooter(placer: Placer, shooter: Shoter, matches: int = 100) -> list:
    game_lengths = []
    for i in range(matches):
        ship_board = placer.place_ships()
        shoot_board = Board()
        ship_count = ship_board.count(Tile.SHIP)
        game_length = 0
        while ship_count > 0:
            game_length += 1
            pos = shooter.shoot(shoot_board)
            if ship_board[pos] == Tile.SHIP:
                shoot_board[pos] = Tile.HIT
                ship_board[pos] = Tile.HIT
                ship_count -= 1
            else:
                shoot_board[pos] = Tile.HIT
                ship_board[pos] = Tile.MISS

        game_lengths.append(game_length)

    return game_lengths
