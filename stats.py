import matplotlib.pyplot as plt

from player import *
from game import *
from dataset import *
from random_player import *
from prediction_shooter import *


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


def plot_predictions(shooter: PredictionShooter):
    board = Board(RandomPlacer().place_ships())
    while board.count_ship_tiles() > 0:
        fig, ax = plt.subplots(1, 3)
        x, y = board_to_sample(board)

        xx = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                xx[row, col] = np.argmax(x[row, col])

        ax[0].matshow(xx, vmin=0, vmax=3)
        ax[1].matshow(y, vmin=0, vmax=1)

        ax[2].matshow(shooter.predict_raw(board), vmin=0, vmax=1)
        plt.show()
        board.shoot(shooter.shoot(board))
