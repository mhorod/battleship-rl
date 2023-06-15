import numpy as np
import matplotlib.pyplot as plt

from game import *
from player import *
from dataset import *
from random_player import *
from prediction_shooter import *

class NaiveBayesPredictor(ShipPredictor):
    def __init__(self, board_config):
        self.board_config = board_config

    def fit(self, xs, ys):
        rows, columns = self.board_config.dimensions
        self.y_phis = np.zeros((rows, columns))
        self.x_phis = np.zeros((rows, columns, 2, rows, columns, 4))
        for row, col in board_positions(self.board_config):
            self.fit_one(row, col, xs, ys)

    def fit_one(self, row, col, xs, ys):
        L = 1
        positive_count = 0
        negative_count = 0

        for x, y in zip(xs, ys):
            if y[row, col] == 1:
                positive_count += 1
            else:
                negative_count += 1
            
            for i, j in board_positions(self.board_config):
                k = np.argmax(x[i, j])
                self.x_phis[row, col, y[i, j], i, j, k] += 1

        self.y_phis[row, col] = (positive_count + L) / (positive_count + negative_count + 2 * L)

        for i, j in board_positions(self.board_config):
            for k in range(4):
                self.x_phis[row, col, 0, i, j, k] = (self.x_phis[row, col, 0, i, j, k] + L) / (negative_count + 2 * L)
                self.x_phis[row, col, 1, i, j, k] = (self.x_phis[row, col, 1, i, j, k] + L) / (positive_count + 2 * L)


    def shoot(self, board):
        predictions = self.predict(board)
        return self.select_best_pos(board, predictions)

    def predict(self, board):
        predictions = np.zeros(self.board_config.dimensions)
        board_repr = board.get_repr()
        for row, col in board_positions(self.board_config):
            predictions[row, col] = self.predict_one(row, col, board_repr)
        return predictions

    def select_best_pos(self, board, predictions):
        tried = 0
        while tried < self.board_config.tiles():
            pos = np.unravel_index(np.argmax(predictions), predictions.shape)
            if board[pos] == Tile.EMPTY:
                return pos
            predictions[pos] = -1
            tried += 1

    def predict_one(self, row, col, board_repr):
        prior_positive = self.y_phis[row, col]
        
        px_negative = np.log(1 - prior_positive)
        px_positive = np.log(prior_positive)

        for i, j in board_positions(self.board_config):
            k = np.argmax(board_repr[i, j])
            px_negative += np.log(self.x_phis[row, col, 0, i, j, k])
            px_positive += np.log(self.x_phis[row, col, 1, i, j, k])

        return np.exp(px_positive - px_negative)

class NaiveBayesShooter(PredictionShooter):
    def __init__(self, model):
        super().__init__(model)

def make_naive_bayes(board_config, placer, shooter, games):
    xs, ys = make_dataset(placer, shooter, games)
    nb = NaiveBayesPredictor(board_config)
    nb.fit(xs, ys)
    return NaiveBayesShooter(nb)  