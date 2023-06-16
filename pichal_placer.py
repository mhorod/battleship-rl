from game import *
from player import *

import random
import numpy as np


class PichalPlacer(Placer):
    def __init__(self, board_config: BoardConfig):
        self.board_config = board_config
        self.weights = np.ones(board_config.dimensions)

    def place_ships(self) -> ShipBoard:
        board = ShipBoard(self.board_config)
        ships_to_place = []
        for ship in self.board_config.ships:
            ships_to_place.extend([ship.length] * ship.count)
        self.place_ships_with_backtracking(board, ships_to_place)
        return board
    
    def generate_shot_sequences(self, shooter: Shooter):
        boards = [Board(self.place_ships()) for _ in range(30)]
        shot_sequences = [[] for _ in range(100)]
        result = []
        while boards:
            shots = shooter.shoot_many(boards)
            to_remove = []
            for i in range(len(boards)):
                boards[i].shoot(shots[i])
                shot_sequences[i].append(shots[i])
                if boards[i].count_ship_tiles() == 0:
                    to_remove.append(i)
            for i in to_remove[::-1]:
                del boards[i]
                result.append(shot_sequences[i])
                del shot_sequences[i]
        return result
    
    def train(self, shooter: Shooter):
        shot_sequences = self.generate_shot_sequences(shooter)
        new_weights = np.zeros(self.board_config.dimensions)
        lens = []
        for sequence in shot_sequences:
            lens.append(len(sequence))
            for i in range(len(sequence)):
                new_weights[sequence[i]] += 1 
        self.weights = self.weights * 0.3 + new_weights * 0.7
        return np.average(lens)

    def place_ships_with_backtracking(self, board: ShipBoard, ships):
        if not ships:
            return True

        possible_ships = []
        ship_weights = []
        for pos in list(board_positions(self.board_config)):
            for orientation in Orientation:
                ship = Ship(pos, ships[0], orientation)
                if board.can_place_ship(ship):
                    possible_ships.append(ship)
                    ship_weights.append(np.product(self.weights[ship.get_tiles()]))
        ship_weights = 1 / (np.array(ship_weights) + 0.01)
        ship_weights /= np.sum(ship_weights)
        while True:
            ship_index = np.random.choice(len(possible_ships), p = ship_weights)
            board.place_ship(possible_ships[ship_index])
            if self.place_ships_with_backtracking(board, ships[1:]):
                return True
            board.remove_ship(possible_ships[ship_index])