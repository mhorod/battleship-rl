from enum import IntEnum, Enum, auto
import numpy as np

CARRIER = 5
BATTLESHIP = 4
CRUISER = 3
DESTROYER = 2
SUBMARINE = 1

SHIPS = {
    CARRIER: 1,
    BATTLESHIP: 1,
    CRUISER: 2,
    DESTROYER: 1,
    SUBMARINE: 0,
}

BOARD_SIZE = 10


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

    def get_tiles(self):
        if self.orientation == Orientation.HORIZONTAL:
            return [(self.pos[0] + i, self.pos[1]) for i in range(self.length)]
        elif self.orientation == Orientation.VERTICAL:
            return [(self.pos[0], self.pos[1] + i) for i in range(self.length)]


class ShipBoard:
    def __init__(self):
        self.ships = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    def get_repr(self):
        result = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.ships[x][y] is not None:
                    result[x][y] = 1
        return result

    def place_ship(self, ship):
        for (x, y) in ship.get_tiles():
            self.ships[x][y] = ship

    def remove_ship(self, ship):
        for (x, y) in ship.get_tiles():
            self.ships[x][y] = None

    def get_ship(self, pos):
        return self.ships[pos[0]][pos[1]]

    def count_ship_tiles(self):
        result = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.ships[x][y] is not None:
                    result += 1
        return result

    def can_place_ship(self, ship):
        for (x, y) in ship.get_tiles():
            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                return False
            if self.ships[x][y] is not None:
                return False
            if x > 0 and self.ships[x - 1][y] is not None:
                return False
            if y > 0 and self.ships[x][y - 1] is not None:
                return False
            if x < BOARD_SIZE - 1 and self.ships[x + 1][y] is not None:
                return False
            if y < BOARD_SIZE - 1 and self.ships[x][y + 1] is not None:
                return False
        return True

    def shoot(self, pos):
        if self.ships[pos[0]][pos[1]] is None:
            return ShotResult.MISS
        return self.ships[pos[0]][pos[1]].shoot(pos)

    def sunken_ships(self):
        sunken = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                ship = self.ships[r][c]
                if ship is not None and np.all(ship.hits):
                    sunken.add(ship)
        return [ship.length for ship in sunken]

    def __getitem__(self, pos):
        return self.ships[pos[0]][pos[1]]


class Board:
    def __init__(self, ship_board):
        self.ship_board = ship_board
        self.repr = np.zeros((BOARD_SIZE, BOARD_SIZE, len(Tile)), dtype=np.int)
        self.repr[:, :, Tile.EMPTY] = 1
        self.ship_tiles = ship_board.count_ship_tiles()

    def get_repr(self):
        return self.repr

    def get_ship_repr(self):
        return self.ship_board.get_repr()

    def __getitem__(self, pos):
        if pos[0] < 0 or pos[0] >= BOARD_SIZE or pos[1] < 0 or pos[1] >= BOARD_SIZE:
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
        new_board = Board(self.ship_board)
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
