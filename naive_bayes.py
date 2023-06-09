import numpy as np

from game import *
from player import *
from dataset import *
from random_player import *
from stats import *

class NaiveBayesShooter(Shooter):
    def fit(self, xs, ys):
        self.y_phis = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.x_phis = np.zeros((BOARD_SIZE, BOARD_SIZE, 2, BOARD_SIZE, BOARD_SIZE, 4))
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
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
            
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    k = np.argmax(x[i, j])
                    self.x_phis[row, col, y[i, j], i, j, k] += 1

        self.y_phis[row, col] = (positive_count + L) / (positive_count + negative_count + 2 * L)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                for k in range(4):
                    self.x_phis[row, col, 0, i, j, k] = (self.x_phis[row, col, 0, i, j, k] + L) / (negative_count + 2 * L)
                    self.x_phis[row, col, 1, i, j, k] = (self.x_phis[row, col, 1, i, j, k] + L) / (positive_count + 2 * L)


    def shoot(self, board):
        predictions = np.zeros((BOARD_SIZE, BOARD_SIZE))
        board_repr = board.get_repr()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                predictions[row, col] = self.predict_one(row, col, board_repr)
        return self.select_best_pos(board, predictions)

    def select_best_pos(self, board, predictions):
        tried = 0
        while tried < BOARD_SIZE * BOARD_SIZE:
            pos = np.unravel_index(np.argmax(predictions), predictions.shape)
            if board[pos] == Tile.EMPTY:
                return pos
            predictions[pos] = -1
            tried += 1

    def predict_one(self, row, col, board_repr):
        prior_positive = self.y_phis[row, col]
        
        px_negative = np.log(1 - prior_positive)
        px_positive = np.log(prior_positive)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                k = np.argmax(board_repr[i, j])
                px_negative += np.log(self.x_phis[row, col, 0, i, j, k])
                px_positive += np.log(self.x_phis[row, col, 1, i, j, k])

        return np.exp(px_positive) / (np.exp(px_positive) + np.exp(px_negative))


xs, ys = make_dataset(RandomPlacer(), RandomShooter(), 20000)
print("Dataset made")
nb = NaiveBayesShooter()
nb.fit(xs, ys)
print("Naive Bayes Shooter fitted")

game_lengths = compare_placer_with_shooter(RandomPlacer(), nb, 10)
print("Naive Bayes Shooter")
print("Average game length: ", np.mean(game_lengths))
print("Standard deviation: ", np.std(game_lengths))
print("Median: ", np.median(game_lengths))
