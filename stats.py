from player import *
from game import *

from tqdm import tqdm

def compare_placer_with_shooter(placer: Placer, shooter: Shooter, matches: int = 100) -> list:
    game_lengths = []
    boards = [Board(placer.place_ships()) for _ in range(matches)]
    ship_counts = [board.count_ship_tiles() for board in boards]
    game_length = 0

    while len(boards) > 0:
        game_length += 1
        shots = shooter.shoot_many(boards)
        indices_to_remove = []
        for i, (board, shot) in enumerate(zip(boards, shots)):
            result = board.shoot(shot)
            if result == ShotResult.HIT or result == ShotResult.SUNK:
                ship_counts[i] -= 1
                if ship_counts[i] == 0:
                    indices_to_remove.append(i)
        
        for i in indices_to_remove[::-1]:
            game_lengths.append(game_length)
            boards.pop(i)
            ship_counts.pop(i)

    return game_lengths    