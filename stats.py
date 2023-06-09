from player import *
from game import *

def compare_placer_with_shooter(placer: Placer, shooter: Shooter, matches: int = 100) -> list:
    game_lengths = []
    for i in range(matches):
        ship_board = placer.place_ships()
        shoot_board = Board(ship_board)
        ship_count = ship_board.count_ship_tiles()
        game_length = 0
        while ship_count > 0:
            game_length += 1
            pos = shooter.shoot(shoot_board)
            result = shoot_board.shoot(pos)
            if result == ShotResult.HIT or result == ShotResult.SUNK:
                ship_count -= 1
        game_lengths.append(game_length)

    return game_lengths
