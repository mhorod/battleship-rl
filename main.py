from enum import Enum, auto

import random

import pygame

import matplotlib.pyplot as plt

from game import *
from stats import *

from random_player import *
from montecarlo_player import *
from tf_player import *

TILE_SIZE = 50
MARGIN = 50

BOARD_WIDTH, BOARD_HEIGHT = BOARD_SIZE * TILE_SIZE, BOARD_SIZE * TILE_SIZE

TILE_TO_COLOR = {
    Tile.EMPTY: (255, 255, 255),
    Tile.MISS: (50, 153, 213),
    Tile.HIT: (220, 80, 70),
    Tile.SUNK: (50, 50, 50),
}

SHIP_COLOR = (160, 160, 160)


class HitBoardDisplay:
    def __init__(self, surface, board):
        self.surface = surface
        self.board = board

    def draw(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.draw_tile(r, c)
        pygame.draw.rect(self.surface, (0, 0, 0), pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT), 2)

    def draw_tile(self, r, c):
        color = TILE_TO_COLOR[self.board[r, c]]
        pygame.draw.rect(self.surface, color, pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        

class ShipBoardDisplay:
    def __init__(self, surface, board):
        self.surface = surface
        self.board = board

    def draw(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.draw_tile(r, c)
        pygame.draw.rect(self.surface, (0, 0, 0), pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT), 2)

    def draw_tile(self, r, c):
        if self.board.ship_board[(r, c)] is None:
            color = TILE_TO_COLOR[Tile.EMPTY]
        else:
            color = SHIP_COLOR

        if self.board[r, c] == Tile.SUNK:
            color = TILE_TO_COLOR[Tile.SUNK]
        elif self.board[r, c] == Tile.HIT:
            color = TILE_TO_COLOR[Tile.HIT]
        elif self.board[r, c] == Tile.MISS:
            color = TILE_TO_COLOR[Tile.MISS]

        pygame.draw.rect(self.surface, color, pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))


class PredictionBoardDisplay:
    def __init__(self, surface, board, shooter):
        self.surface = surface
        self.board = board
        self.shooter = shooter

    def draw(self):
        predictions = self.shooter.predict_raw(self.board)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.draw_tile(r, c, predictions[r, c])
        pygame.draw.rect(self.surface, (0, 0, 0), pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT), 2)

    def draw_tile(self, r, c, value):
        target_color = (50, 93, 213)
        diff = (255 - target_color[0], 255 - target_color[1], 255 - target_color[2])
        value = min(1, max(0, 1 - value))
        color = (target_color[0] + diff[0] * value, target_color[1] + diff[1] * value, target_color[2] + diff[2] * value)
        pygame.draw.rect(self.surface, color, pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))

class Visualization:
    def __init__(self, shooter):
        pygame.init()

        self.shooter = shooter
        self.board = Board(RandomPlacer().place_ships())

        display_count = 3 if isinstance(shooter, PredictionShooter) else 2

        self.width = BOARD_WIDTH * display_count + 2 * MARGIN + (display_count - 1) * MARGIN
        self.height = BOARD_HEIGHT + 2 * MARGIN

        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Battleship")

        ship_board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_WIDTH, BOARD_HEIGHT)
        hit_board_rect = pygame.Rect(MARGIN + BOARD_WIDTH + MARGIN, MARGIN, BOARD_WIDTH, BOARD_HEIGHT)
        prediction_board_rect = pygame.Rect(MARGIN + 2 * (BOARD_WIDTH + MARGIN), MARGIN, BOARD_WIDTH, BOARD_HEIGHT)

        ship_board_surface = self.screen.subsurface(ship_board_rect)
        hit_board_surface = self.screen.subsurface(hit_board_rect)

        self.displays = [ShipBoardDisplay(ship_board_surface, self.board), HitBoardDisplay(hit_board_surface, self.board)]

        if isinstance(shooter, PredictionShooter):
            prediction_board_surface = self.screen.subsurface(prediction_board_rect)
            self.displays.append(PredictionBoardDisplay(prediction_board_surface, self.board, self.shooter))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.board.shoot(self.shooter.shoot(self.board))

            self.draw()
            pygame.display.update()


    def draw(self):
        self.screen.fill((255, 255, 255))
        for display in self.displays:
            display.draw()


shooter = MonteCarloShooter(200)
visualization = Visualization(shooter)
visualization.run()