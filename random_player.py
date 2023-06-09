import random

from player import *
from game import *



class RandomPlacer(Placer):
    def place_ships(self) -> Board:
        board = Board()
        positions = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
        random.shuffle(positions)
        ships_to_place = []
        for ship, count in SHIPS.items():
            ships_to_place += [ship] * count
        random.shuffle(ships_to_place)
        self.place_ships_with_backtracking(board, ships_to_place, positions)
        return board

    def place_ships_with_backtracking(self, board, ships, positions):
        if not ships:
            return True

        ship = ships[0]
        for pos in positions:
            actions = [
                (
                    lambda: self.place_ship_horizontally(board, ship, pos),
                    lambda: self.remove_ship_horizontally(board, ship, pos)
                ),
                ( 
                    lambda: self.place_ship_vertically(board, ship, pos),
                    lambda: self.remove_ship_vertically(board, ship, pos)
                )
            ]

            h = self.can_place_ship_horizontally(board, ship, pos)
            v = self.can_place_ship_vertically(board, ship, pos)

            if h and v:
                action = random.choice(actions)
            elif h:
                action = actions[0]
            elif v:
                action = actions[1]
            else:
                continue

            place_ship, remove_ship = action
            place_ship()
            if self.place_ships_with_backtracking(board, ships[1:], positions):
                return True
            remove_ship()

        return False    

    def can_place_ship_horizontally(self, board, ship, pos):
        if pos[1] + ship > BOARD_SIZE:
            return False
        
        if board[pos] != Tile.EMPTY:
            return False
        
        if board[pos[0], pos[1] - 1] != Tile.EMPTY or board[pos[0], pos[1] + ship] != Tile.EMPTY:
            return False

        for i in range(ship):
            if board[pos[0], pos[1] + i] != Tile.EMPTY:
                return False
            if board[pos[0] - 1, pos[1] + i] != Tile.EMPTY:
                return False
            if board[pos[0] + 1, pos[1] + i] != Tile.EMPTY:
                return False
            
        return True

    def can_place_ship_vertically(self, board, ship, pos):
        if pos[0] + ship > BOARD_SIZE:
            return False
        
        if board[pos] != Tile.EMPTY:
            return False
        
        if board[pos[0] - 1, pos[1]] != Tile.EMPTY or board[pos[0] + ship, pos[1]] != Tile.EMPTY:
            return False

        for i in range(ship):
            if board[pos[0] + i, pos[1]] != Tile.EMPTY:
                return False
            if board[pos[0] + i, pos[1] - 1] != Tile.EMPTY:
                return False
            if board[pos[0] + i, pos[1] + 1] != Tile.EMPTY:
                return False
            
        return True


    def place_ship_horizontally(self, board, ship, pos):
        for i in range(ship):
            board[pos[0], pos[1] + i] = Tile.SHIP
    
    def remove_ship_horizontally(self, board, ship, pos):
        for i in range(ship):
            board[pos[0], pos[1] + i] = Tile.EMPTY

    def place_ship_vertically(self, board, ship, pos):
        for i in range(ship):
            board[pos[0] + i, pos[1]] = Tile.SHIP

    def remove_ship_vertically(self, board, ship, pos):
        for i in range(ship):
            board[pos[0] + i, pos[1]] = Tile.EMPTY


class RandomShooter(Shooter):
    def shoot(self, board) -> tuple:
        viable_positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == Tile.EMPTY:
                    viable_positions.append((i, j))
        return random.choice(viable_positions)


class RandomPlayer(RandomPlacer, RandomShooter):
    pass