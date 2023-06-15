from enum import IntEnum, Enum, auto
from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class ShipConfig:
    length: int
    count: int

@dataclass
class BoardConfig:
    size: int
    ships: List[ShipConfig]
    
    @property
    def dimensions(self):
        return (self.size, self.size)

DEFAULT_BOARD_CONFIG = BoardConfig(
    10,
    [
        ShipConfig(length=5, count=1),
        ShipConfig(length=4, count=1),
        ShipConfig(length=3, count=2),
        ShipConfig(length=2, count=1),
    ]
)

TINY_BOARD_CONFIG = BoardConfig(
    5,
    [
        ShipConfig(length=3, count=1),
        ShipConfig(length=2, count=1),
    ]
)

class Tile(IntEnum):
    EMPTY = 0
    MISS = 1
    HIT = 2
    SUNK = 3


class Orientation(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class ShotResult(Enum):
    MISS = auto()
    HIT = auto()
    SUNK = auto()
    ILLEGAL = auto()


class Ship:
    def __init__(self, pos, length, orientation):
        self.pos = pos
        self.length = length
        self.orientation = orientation
        self.hits = np.zeros(length)

    def shoot(self, pos):
        rel_pos = (pos[0] - self.pos[0], pos[1] - self.pos[1])
        if rel_pos[1 - self.orientation] == 0 and 0 <= rel_pos[self.orientation] < self.length:
            self.hits[rel_pos[self.orientation]] = 1
            return ShotResult.SUNK if np.all(self.hits) else ShotResult.HIT
        return ShotResult.MISS

    def clone(self):
        return Ship(self.pos, self.length, self.orientation)

    def get_tiles(self):
        if self.orientation == Orientation.HORIZONTAL:
            return [(self.pos[0] + i, self.pos[1]) for i in range(self.length)]
        elif self.orientation == Orientation.VERTICAL:
            return [(self.pos[0], self.pos[1] + i) for i in range(self.length)]


def board_positions(config):
    rows, columns = config.dimensions
    for r in range(rows):
        for c in range(columns):
            yield (r, c)

class ShipBoard:
    def __init__(self, config):
        self.config = config
        self.ships = {pos : None for pos in board_positions(self.config)}

    def clone(self):
        result = ShipBoard()
        ships = {}
        for pos in board_positions(self.config):
            if self.ships[pos] not in ships and self.ships[pos] is not None:
                ships[self.ships[pos]] = self.ships[pos].clone()

        for i, j in board_positions(self.config):
            if self.ships[pos] is not None:
                result.ships[pos] = ships[self.ships[pos]]
        return result

    def get_repr(self):
        result = np.zeros(self.config.dimensions, dtype=np.int)
        for pos in board_positions(self.config):
            if self.ships[pos] is not None:
                result[pos] = 1
        return result

    def place_ship(self, ship):
        for pos in ship.get_tiles():
            self.ships[pos] = ship

    def remove_ship(self, ship):
        for pos in ship.get_tiles():
            self.ships[pos] = None

    def get_ship(self, pos):
        return self.ships[pos]

    def count_ship_tiles(self):
        return sum(1 for pos in board_positions(self.config) if self.ships[pos] is not None)

    def can_place_ship(self, ship):
        for (x, y) in ship.get_tiles():
            if x < 0 or x >= self.config.size or y < 0 or y >= self.config.size:
                return False
            if self.ships[x, y] is not None:
                return False
            if x > 0 and self.ships[x - 1, y] is not None:
                return False
            if y > 0 and self.ships[x, y - 1] is not None:
                return False
            if x < self.config.size - 1 and self.ships[x + 1, y] is not None:
                return False
            if y < self.config.size - 1 and self.ships[x, y + 1] is not None:
                return False
        return True

    def shoot(self, pos):
        if self.ships[pos] is None:
            return ShotResult.MISS
        return self.ships[pos].shoot(pos)

    def sunken_ships(self):
        sunken = set()
        for pos in board_positions(self.config.size):
            ship = self.ships[pos]
            if ship is not None and np.all(ship.hits):
                sunken.add(ship)
        return [ship.length for ship in sunken]

    def __getitem__(self, pos):
        return self.ships[pos]


class Board:
    def __init__(self, ship_board):
        self.ship_board = ship_board
        self.config = ship_board.config
        self.repr = np.zeros((self.config.size, self.config.size, len(Tile)), dtype=np.int)
        self.repr[:, :, Tile.EMPTY] = 1
        self.ship_tiles = ship_board.count_ship_tiles()

    def get_repr(self):
        return self.repr

    def get_ship_repr(self):
        return self.ship_board.get_repr()

    def __getitem__(self, pos):
        if pos[0] < 0 or pos[0] >= self.config.size or pos[1] < 0 or pos[1] >= self.config.size:
            return None
        return Tile(np.argmax(self.repr[pos[0], pos[1]]))

    def __setitem__(self, pos, tile):
        self.repr[pos[0], pos[1], :] = 0
        self.repr[pos[0], pos[1], tile] = 1

    def count(self, tile):
        return np.sum(self.repr[:, :, tile])

    def count_ship_tiles(self):
        return self.ship_tiles

    def hash(self):
        return hash(self.repr.tobytes())

    def __eq__(self, other):
        return self.hash() == other.hash() and np.all(self.repr == other.repr)

    def clone(self):
        new_board = Board(self.ship_board.clone())
        new_board.repr = np.copy(self.repr)
        return new_board

    def shoot(self, pos):
        if self[pos] != Tile.EMPTY:
            return ShotResult.ILLEGAL
        result = self.ship_board.shoot(pos)
        if result == ShotResult.MISS:
            self[pos] = Tile.MISS
        elif result == ShotResult.HIT:
            self.ship_tiles -= 1
            self[pos] = Tile.HIT
        elif result == ShotResult.SUNK:
            self.ship_tiles -= 1
            for ship_tile in self.ship_board.get_ship(pos).get_tiles():
                self[ship_tile] = Tile.SUNK

        return result
