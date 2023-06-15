import random

from game import *

def make_dataset(placer, shooter, games):
    xs = []
    ys = []
    boards = [Board(placer.place_ships()) for _ in range(games)]
    ship_counts = [board.count_ship_tiles() for board in boards]
    tiles = boards[0].config.size * boards[0].config.size
    min_moment = min(ship_counts) - 1
    moments_to_extract = [random.randint(min_moment, tiles) for _ in range(games)]
    game_length = 0

    while len(boards) > 0:
        game_length += 1
        indices_to_remove = []
        for i, (board, moment) in enumerate(zip(boards, moments_to_extract)):
            if game_length == moment:
                x, y = board_to_sample(boards[i])
                xs.append(x)
                ys.append(y)
                indices_to_remove.append(i)

        for i in indices_to_remove[::-1]:
            boards.pop(i)
            ship_counts.pop(i)
            moments_to_extract.pop(i)

        if len(boards) == 0:
            break

        shots = shooter.shoot_many(boards)
        indices_to_remove = []
        for i, (board, shot) in enumerate(zip(boards, shots)):
            result = board.shoot(shot)
            if result == ShotResult.HIT or result == ShotResult.SUNK:
                ship_counts[i] -= 1
                if ship_counts[i] == 0:
                    indices_to_remove.append(i)
        
        for i in indices_to_remove[::-1]:
            x, y = board_to_sample(board)
            xs.append(x)
            ys.append(y)

            boards.pop(i)
            ship_counts.pop(i)
            moments_to_extract.pop(i)

    return xs, ys

def board_to_sample(board):
    x = board.get_repr()
    y = board.get_ship_repr()
    for pos in board_positions(board.config):
        if board[pos] != Tile.EMPTY:
            y[pos] = 0
    return x, y
