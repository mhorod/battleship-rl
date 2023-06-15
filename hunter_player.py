import random

import matplotlib.pyplot as plt


from game import *
from player import *
from random_player import *
from stats import *

class HunterPredictor(ShipPredictor):
    '''
    Predicts ships using human strategy - shoot at random 
    until hit, then shoot around until sunk.
    '''

    def predict_ships(self, board) -> np.ndarray:
        if board.count(Tile.HIT) == 0:
            return self.predict_ships_random(board)
        else:
            return self.predict_ships_near(board)

    def predict_ships_random(self, board):
        predictions = np.zeros((board.config.size, board.config.size))
        for pos in board_positions(board.config):
            if self.can_hit(board, pos):
                predictions[pos] = 1
        return predictions / np.sum(predictions)

    def predict_ships_near(self, board):
        functions = [
            self.predict_ships_horizontal,
            self.predict_ships_vertical,
            self.predict_ships_around,
        ]

        for f in functions:
            predictions, found = f(board)
            if found:
                return predictions

    def predict_ships_horizontal(self, board):
        '''
        Find at least two hits in a row and predict that the ship is to left or right
        '''

        predictions = np.zeros((board.config.size, board.config.size))
        found = False
        for pos in board_positions(board.config):
            if board[pos] == Tile.HIT and board[self.left(pos)] == Tile.HIT:
                hit_begin = self.find_horizontal_hit_begin(board, pos)
                hit_end = self.find_horizontal_hit_end(board, pos)

                candidates = [self.left(hit_begin), self.right(hit_end)]
                candidates = [pos for pos in candidates if self.can_hit(board, pos)]

                if candidates:
                    found = True
                    for pos in candidates:
                        predictions[pos] = 1 / len(candidates)
        return predictions, found


    def predict_ships_vertical(self, board):
        '''
        Find at least two hits in a column and predict that the ship is to up or down
        '''

        predictions = np.zeros((board.config.size, board.config.size))
        found = False
        for pos in board_positions(board.config):
            if board[pos] == Tile.HIT and board[self.up(pos)] == Tile.HIT:
                hit_begin = self.find_vertical_hit_begin(board, pos)
                hit_end = self.find_vertical_hit_end(board, pos)

                candidates = [self.up(hit_begin), self.down(hit_end)]
                candidates = [pos for pos in candidates if self.can_hit(board, pos)]

                if candidates:
                    found = True
                    for pos in candidates:
                        predictions[pos] = 1 / len(candidates)
        return predictions, found

    def predict_ships_around(self, board):
        '''
        Find at least one hit and predict that the ship is around it
        '''

        predictions = np.zeros((board.config.size, board.config.size))
        found = False
        for pos in board_positions(board.config):
            if board[pos] == Tile.HIT:
                candidates = [self.left(pos), self.right(pos), self.up(pos), self.down(pos)]
                candidates = [pos for pos in candidates if self.can_hit(board, pos)]
                if candidates:
                    found = True
                    for pos in candidates:
                        predictions[pos] += 1 / len(candidates)
        return predictions, found

    def find_horizontal_hit_begin(self, board, pos):
        while board[self.left(pos)] == Tile.HIT:
            pos = self.left(pos)
        return pos

    def find_horizontal_hit_end(self, board, pos):
        while board[self.right(pos)] == Tile.HIT:
            pos = self.right(pos)
        return pos

    def find_vertical_hit_begin(self, board, pos):
        while board[self.up(pos)] == Tile.HIT:
            pos = self.up(pos)
        return pos
    
    def find_vertical_hit_end(self, board, pos):
        while board[self.down(pos)] == Tile.HIT:
            pos = self.down(pos)
        return pos
        
    def can_hit(self, board, pos):
        if not board[pos] == Tile.EMPTY:
            return False
        around = [self.left(pos), self.right(pos), self.up(pos), self.down(pos)]
        if any(board[pos] == Tile.SUNK for pos in around):
            return False
        
        hits_around = [pos for pos in around if board[pos] == Tile.HIT]

        for r1, c1 in hits_around:
            for r2, c2 in hits_around:
                if r1 != r2 and c1 != c2:
                    return False

        return True

    def left(self, pos):
        return (pos[0], pos[1] - 1)

    def right(self, pos):
        return (pos[0], pos[1] + 1)
    
    def up(self, pos):
        return (pos[0] - 1, pos[1])
    
    def down(self, pos):
        return (pos[0] + 1, pos[1])


class EvenHunterPredictor(HunterPredictor):
    def predict_ships_random(self, board):
        predictions = np.zeros((board.config.size, board.config.size))
        marked = 0
        for pos in board_positions(board.config):
            if (pos[0] + pos[1]) % 2 == 0 and self.can_hit(board, pos):
                predictions[pos] = 1
                marked += 1
        
        if marked == 0:
            for pos in board_positions(board.config):
                if (pos[0] + pos[1]) % 2 != 0 and self.can_hit(board, pos):
                    predictions[pos] = 1
                    marked += 1

        return predictions / marked

class HunterShooter(PredictionShooter):
    def __init__(self):
        super().__init__(HunterPredictor())


class EvenHunterShooter(PredictionShooter):
    def __init__(self):
        super().__init__(EvenHunterPredictor())