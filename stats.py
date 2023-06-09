from player import *
from game import *

from tqdm import tqdm

def shoot(pos, ship_board, shoot_board):
    tile = ship_board[pos]
    if tile == Tile.SHIP:
        shoot_board[pos] = Tile.HIT
        return True
    else:
        shoot_board[pos] = Tile.MISS
        return False

def compare_placer_with_shooter(placer: Placer, shooter: Shooter, matches: int = 100) -> list:
    game_lengths = []
    ship_boards = [placer.place_ships() for _ in range(matches)]
    shoot_boards = [Board() for _ in range(matches)]
    ship_counts = [ship_board.count(Tile.SHIP) for ship_board in ship_boards]
    game_length = 0

    while len(ship_boards) > 0:
        game_length += 1
        shots = shooter.shoot_many(shoot_boards)
        indices_to_remove = []
        for i, (ship_board, shoot_board, shot) in enumerate(zip(ship_boards, shoot_boards, shots)):
            result = shoot(shot, ship_board, shoot_board)
            if result:
                ship_counts[i] -= 1
                if ship_counts[i] == 0:
                    indices_to_remove.append(i)
        
        for i in indices_to_remove[::-1]:
            game_lengths.append(game_length)
            ship_boards.pop(i)
            shoot_boards.pop(i)
            ship_counts.pop(i)

    return game_lengths    