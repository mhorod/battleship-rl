import random

from player import *
from game import *



class RandomPlacer(Placer):
    def place_ships(self) -> Board:
        board = ShipBoard()
        ships_to_place = []
        for ship, count in SHIPS.items():
            ships_to_place += [ship] * count
        random.shuffle(ships_to_place)
        self.place_ships_with_backtracking(board, ships_to_place)
        return board

    def place_ships_with_backtracking(self, board, ships):
        if not ships:
            return True

        positions = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
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
        while(True):
            pos = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            if board[pos] == Tile.EMPTY:
                return pos

class RandomPlayer(RandomPlacer, RandomShooter):
    pass