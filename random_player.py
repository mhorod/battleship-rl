import random

import matplotlib.pyplot as plt

from player import *
from game import *


class RandomPlacer(Placer):
    def __init__(self, board_config):
        self.board_config = board_config

    def place_ships(self) -> ShipBoard:
        board = ShipBoard(self.board_config)
        ships_to_place = []
        for ship in self.board_config.ships:
            ships_to_place.extend([ship.length] * ship.count)
        self.place_ships_with_backtracking(board, ships_to_place)
        return board

    def place_ships_with_backtracking(self, board, ships):
        if not ships:
            return True

        positions = list(board_positions(self.board_config))
        random.shuffle(positions)
        for pos in positions:
            possible_ships = [ship for ship in [
                Ship(pos, ships[0], Orientation.HORIZONTAL),
                Ship(pos, ships[0], Orientation.VERTICAL)
            ] if board.can_place_ship(ship)]

            if not possible_ships:
                continue

            ship = random.choice(possible_ships)

            board.place_ship(ship)
            if self.place_ships_with_backtracking(board, ships[1:]):
                return True
            board.remove_ship(ship)

        return False


class RandomShooter(Shooter):
    def shoot(self, board) -> tuple:
        while (True):
            pos = (random.randint(0, board.config.size - 1),
                   random.randint(0, board.config.size - 1))
            if board[pos] == Tile.EMPTY:
                return pos


class RandomPlayer(RandomPlacer, RandomShooter):
    def __init__(self, board_config):
        super().__init__(board_config)