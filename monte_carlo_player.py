import random

import numpy as np
from matplotlib import pyplot as plt

from game import *
from player import *
from random_player import *
from stats import *
from dataset import *

class MonteCarloPredictor(ShipPredictor):
    '''
    Predicts ship placement using Monte Carlo simulations.
    Performs a number of random ship placements that are consistent with the board
    and counts how often a ship is placed on a given tile.
    '''
    def __init__(self, simulations):
        self.simulations = simulations

    def predict_ships(self, board):
        values = np.zeros(board.config.dimensions)
        for _ in range(self.simulations):
            values += self.simulate(board)
        return values / self.simulations
        
    def simulate(self, board):
        avoid = set()
        good = []
        for pos in board_positions(board.config):
            if board[pos] in (Tile.SUNK, Tile.MISS):
                avoid.add(pos)
            else:
                good.append(pos)
        random.shuffle(good)

        ship_lengths = self.missing_ships(board)
        random.shuffle(ship_lengths)

        return self.random_placement(board, good, avoid, ship_lengths)

    def missing_ships(self, board):
        ship_lengths = []
        for ship in board.config.ships:
            ship_lengths += [ship.length] * ship.count
        for sunken in board.ship_board.sunken_ships():
            ship_lengths.remove(sunken)
        return ship_lengths


    def random_placement(self, board, good, avoid, ship_lengths):
        for pos in good:
            for ship_length in ship_lengths:
                if self.can_place_ship(board, avoid, pos, ship_length, Orientation.HORIZONTAL):
                    return self.place_ship(board, pos, ship_length, Orientation.HORIZONTAL)
                elif self.can_place_ship(board, avoid, pos, ship_length, Orientation.VERTICAL):
                    return self.place_ship(board, pos, ship_length, Orientation.VERTICAL)
        return np.zeros(board.config.dimensions)
    
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

    def place_ship(self, board, pos, length, orientation):
        values = np.zeros(board.config.dimensions)
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


class MonteCarloShooter(PredictionShooter):
    def __init__(self, simulations):
        super().__init__(MonteCarloPredictor(simulations))