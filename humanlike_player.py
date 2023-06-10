import random

import matplotlib.pyplot as plt


from game import *
from player import *
from random_player import *
from stats import *

class HumanLikeShooter(Shooter):
    def shoot(self, board) -> tuple:
        if board.count(Tile.HIT) == 0:
            return self.shoot_random(board)
        else:
            return self.shoot_near(board)
        
    def shoot_random(self, board):
        while True:
            r, c = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            if r % 2 != c % 2:
                continue
            if Tile.SUNK in (board[r - 1, c], board[r + 1, c], board[r, c - 1], board[r, c + 1]):
                continue
            if board[r, c] == Tile.EMPTY:
                return r, c

    def shoot_near(self, board):
        shots = [
            self.shoot_left_to_right(board),
            self.shoot_right_to_left(board),
            self.shoot_top_to_bottom(board),
            self.shoot_bottom_to_top(board),
        ] 

        viable = [shot for shot in shots if shot is not None]
        if viable:
            return random.choice(viable)
        else:
            return self.shoot_around(board)
    
    def can_hit(self, board, r, c):
        if not board[r, c] == Tile.EMPTY:
            return False
        if Tile.SUNK in (board[r - 1, c], board[r + 1, c], board[r, c - 1], board[r, c + 1]):
            return False
        return True


    def shoot_left_to_right(self, board):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE - 1):
                if board[r, c - 1] == Tile.HIT and board[r, c] == Tile.HIT and board[r, c + 1] == Tile.EMPTY:
                    if self.can_hit(board, r, c + 1):
                        return r, c + 1

    def shoot_right_to_left(self, board):
        for r in range(BOARD_SIZE):
            for c in range(1, BOARD_SIZE):
                if board[r, c - 1] == Tile.EMPTY and board[r, c] == Tile.HIT and board[r, c + 1] == Tile.HIT:
                    if self.can_hit(board, r, c - 1):
                        return r, c - 1
                

    def shoot_top_to_bottom(self, board):
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE - 1):
                if board[r - 1, c] == Tile.HIT and board[r, c] == Tile.HIT and board[r + 1, c] == Tile.EMPTY:
                    return r + 1, c
                
    def shoot_bottom_to_top(self, board):
        for c in range(BOARD_SIZE):
            for r in range(1, BOARD_SIZE):
                if board[r - 1, c] == Tile.EMPTY and board[r, c] == Tile.HIT and board[r + 1, c] == Tile.HIT:
                    if self.can_hit(board, r - 1, c):
                        return r - 1, c
                
    def shoot_around(self, board):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == Tile.HIT:
                    around = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    around = [pos for pos in around if board[pos] == Tile.EMPTY and self.can_hit(board, *pos)]
                    if around:
                        return random.choice(around)