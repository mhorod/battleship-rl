from enum import Enum, auto

CARRIER = 5
BATTLESHIP = 4
CRUISER = 3
DESTROYER = 2
SUBMARINE = 1

SHIPS = {
    CARRIER: 2,
    BATTLESHIP: 2,
    CRUISER: 3,
    DESTROYER: 3,
    SUBMARINE: 0,
}

BOARD_SIZE = 10

class Tile(Enum):
    EMPTY = auto()
    MISS = auto()
    HIT = auto()
    SHIP = auto()


class Board:
    def __init__(self):
        self.tiles = [[Tile.EMPTY for _ in range(BOARD_SIZE)]
                      for _ in range(BOARD_SIZE)]
        
    def __getitem__(self, pos):
        if pos[0] < 0 or pos[0] >= BOARD_SIZE or pos[1] < 0 or pos[1] >= BOARD_SIZE:
            return Tile.EMPTY
        return self.tiles[pos[0]][pos[1]]
    
    def __setitem__(self, pos, tile):
        self.tiles[pos[0]][pos[1]] = tile

    def count(self, tile):
        return sum([1 for row in self.tiles for t in row if t == tile])