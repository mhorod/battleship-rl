import random

import numpy as np
from matplotlib import pyplot as plt

from game import *
from player import *
from random_player import *
from stats import *
from dataset import *

class HeatmapShooter(Shooter):
    def __init__(self, simulations):
        self.simulations = simulations

    def shoot(self, board) -> tuple:
        pos =  self.select_best_pos(board, self.predict(board))
        return pos

    def select_best_pos(self, board, predictions):
        tried = 0
        while tried < BOARD_SIZE * BOARD_SIZE:
            pos = np.unravel_index(np.argmax(predictions), predictions.shape)
            if board[pos] == Tile.EMPTY:
                return pos
            predictions[pos] = -1
            tried += 1

    def predict(self, board):
        values = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for _ in range(self.simulations):
            values += self.simulate(board)
        return values / self.simulations
        
    def simulate(self, board):
        avoid = set()
        good = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] in (Tile.SUNK, Tile.MISS):
                    avoid.add((r, c))
                else:
                    good.append((r, c))
        random.shuffle(good)

        ship_lengths = self.missing_ships(board)
        random.shuffle(ship_lengths)

        return self.random_placement(board, good, avoid, ship_lengths)

    def missing_ships(self, board):
        ship_lengths = []
        for ship in SHIPS:
            ship_lengths += [ship] * SHIPS[ship]
        for sunken in board.ship_board.sunken_ships():
            ship_lengths.remove(sunken)
        return ship_lengths



    
    def random_placement(self, board, good, avoid, ship_lengths):
        for pos in good:
            for ship_length in ship_lengths:
                if self.can_place_ship(board, avoid, pos, ship_length, Orientation.HORIZONTAL):
                    return self.place_ship(board, avoid, pos, ship_length, Orientation.HORIZONTAL)
                elif self.can_place_ship(board, avoid, pos, ship_length, Orientation.VERTICAL):
                    return self.place_ship(board, avoid, pos, ship_length, Orientation.VERTICAL)
        return np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    def can_place_ship(self, board, avoid, pos, length, orientation):
        if orientation == Orientation.HORIZONTAL:
            r, c = pos[0], pos[1]
            if board[r, c - 1] in (Tile.HIT, Tile.SUNK) or board[r, c + length] in (Tile.HIT, Tile.SUNK):
                return False

            for i in range(length):
                r, c = pos[0], pos[1] + i
                if board[r, c] is None or (r, c) in avoid:
                    return False
                if board[r - 1, c] in (Tile.HIT, Tile.SUNK) or board[r + 1, c] in (Tile.HIT, Tile.SUNK):
                    return False

        elif orientation == Orientation.VERTICAL:
            r, c = pos[0], pos[1]

            if board[r - 1, c] in (Tile.HIT, Tile.SUNK) or board[r + length, c] in (Tile.HIT, Tile.SUNK):
                return False

            for i in range(length):
                r, c = pos[0] + i, pos[1]
                if board[r, c] is None or (r, c) in avoid:
                    return False
                if board[r, c - 1] in (Tile.HIT, Tile.SUNK) or board[r, c + 1] in (Tile.HIT, Tile.SUNK):
                    return False

        return True

    def place_ship(self, board, avoid, pos, length, orientation):
        values = np.zeros((BOARD_SIZE, BOARD_SIZE))
        positions = []
        if orientation == Orientation.HORIZONTAL:
            for i in range(length):
                positions.append((pos[0], pos[1] + i))
        elif orientation == Orientation.VERTICAL:
            for i in range(length):
                positions.append((pos[0] + i, pos[1]))

        hit_bonus = 5 * sum(board[pos] == Tile.HIT for pos in positions)

        for pos in positions:
            if board[pos] == Tile.EMPTY:
                values[pos] += 1 + hit_bonus

        return values


def plot_predictions(shooter):
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

        ax[2].matshow(shooter.predict(board))
        plt.show()
        board.shoot(shooter.shoot(board))

shooter = HeatmapShooter(100)
#plot_predictions(shooter)
game_lengths = compare_placer_with_shooter(RandomPlacer(), shooter, 10)
print(f"Heatmap shooter: {np.mean(game_lengths)} " + u"\u00B1" + f" {np.std(game_lengths)}")