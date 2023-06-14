from enum import Enum, auto

import random

import pygame

import matplotlib.pyplot as plt

from game import *
from stats import *

from random_player import *
from montecarlo_player import *
from tf_player import *
from humanlike_player import *
from hybrid_player import *

TILE_SIZE = 40
MARGIN = 10

# Size of the board without the frame
TILES_WIDTH = TILE_SIZE * BOARD_SIZE
TILES_HEIGHT = TILE_SIZE * BOARD_SIZE

# Add one tile to have space for the labels
BOARD_WIDTH = TILES_WIDTH + TILE_SIZE
BOARD_HEIGHT = TILES_HEIGHT + TILE_SIZE

INFO_HEIGHT = 100

TILE_TO_COLOR = {
    Tile.EMPTY: (255, 255, 255),
    Tile.MISS: (50, 153, 213),
    Tile.HIT: (220, 80, 70),
    Tile.SUNK: (50, 50, 50),
}

SHIP_COLOR = (160, 160, 160)


class BoardDisplay:
    def __init__(self, window_surface):
        self.window_surface = window_surface
        self.board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
        self.tile_surface = self.board_surface.subsurface(
            (TILE_SIZE, TILE_SIZE, TILES_WIDTH, TILES_HEIGHT))

    def display(self):
        self.window_surface.blit(self.board_surface, (0, 0))

    def update(self):
        self.board_surface.fill((255, 255, 255))
        self.draw_tiles()
        self.draw_grid()
        self.draw_board_frame()
        self.draw_board_labels()

    def draw_grid(self):
        grid_color = (200, 200, 200)
        for r in range(1, BOARD_SIZE):
            pygame.draw.line(self.tile_surface, grid_color,
                             (0, r * TILE_SIZE), (TILES_WIDTH, r * TILE_SIZE))
        for c in range(1, BOARD_SIZE):
            pygame.draw.line(self.tile_surface, grid_color,
                             (c * TILE_SIZE, 0), (c * TILE_SIZE, TILES_HEIGHT))

    def draw_board_frame(self):
        pygame.draw.rect(self.tile_surface, (0, 0, 0),
                         (0, 0, TILES_WIDTH, TILES_HEIGHT), 2)

    def draw_board_labels(self):
        self.draw_row_labels()
        self.draw_column_labels()

    def draw_row_labels(self):
        for i in range(1, 11):
            text = str(i)
            text = pygame.font.SysFont(
                'Arial', 14).render(text, True, (0, 0, 0))
            tw, th = text.get_size()
            x = (TILE_SIZE - tw) / 2
            y = i * TILE_SIZE + (TILE_SIZE - th) / 2
            self.board_surface.blit(text, (x, y))

    def draw_column_labels(self):
        for i in range(1, 11):
            text = chr(ord('A') + i - 1)
            text = pygame.font.SysFont(
                'Arial', 14).render(text, True, (0, 0, 0))
            tw, th = text.get_size()
            x = i * TILE_SIZE + (TILE_SIZE - tw) / 2
            y = (TILE_SIZE - th) / 2
            self.board_surface.blit(text, (x, y))

    def draw_tiles(self):
        pass

    def draw_tile(self, r, c, color):
        pygame.draw.rect(self.tile_surface, color,
                         (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))


class HitBoardDisplay(BoardDisplay):
    def __init__(self, window_surface, board):
        super().__init__(window_surface)
        self.board = board

    def draw_tiles(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = TILE_TO_COLOR[self.board[r, c]]
                self.draw_tile(r, c, color)


class ShipBoardDisplay(BoardDisplay):
    def __init__(self, window_surface, board):
        super().__init__(window_surface)
        self.window = window_surface
        self.board = board

    def draw_tiles(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.draw_tile(r, c, self.get_tile_color(r, c))

    def get_tile_color(self, r, c):
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

        return color


class PredictionBoardDisplay(BoardDisplay):
    def __init__(self, window_surface, board, shooter):
        super().__init__(window_surface)
        self.board = board
        self.shooter = shooter

    def draw_tiles(self):
        predictions = self.shooter.predict_raw(self.board)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.draw_tile(r, c, predictions[r, c])

    def draw_tile(self, r, c, value):
        super().draw_tile(r, c, self.get_prediction_color(value))
        if value > 0.01:
            self.draw_tile_value(r, c, value)

    def get_prediction_color(self, value):
        target_color = (50, 93, 213)
        diff = (255 - target_color[0], 255 -
                target_color[1], 255 - target_color[2])
        draw_value = 1 - (1 - value) ** 2
        draw_value = min(1, max(0, 1 - draw_value))
        color = (target_color[0] + diff[0] * draw_value, target_color[1] +
                 diff[1] * draw_value, target_color[2] + diff[2] * draw_value)
        return color

    def draw_tile_value(self, r, c, value):
        text = pygame.font.SysFont('Arial', 14).render(
            str(round(value, 2)), True, (0, 0, 0))
        tw = text.get_width()
        th = text.get_height()
        self.tile_surface.blit(
            text, (c * TILE_SIZE + TILE_SIZE / 2 - tw / 2, r * TILE_SIZE + TILE_SIZE / 2 - th / 2))


class Visualization:
    def __init__(self, shooter, shooter_name):
        pygame.init()

        self.shooter = shooter
        self.board = Board(RandomPlacer().place_ships())

        display_count = 3 if isinstance(shooter, PredictionShooter) else 2

        self.width = BOARD_WIDTH * display_count + \
            2 * MARGIN + (display_count - 1) * MARGIN
        self.height = BOARD_HEIGHT + 2 * MARGIN + INFO_HEIGHT

        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Battleship")

        ship_board_rect = pygame.Rect(
            MARGIN, MARGIN, BOARD_WIDTH, BOARD_HEIGHT)
        hit_board_rect = pygame.Rect(
            MARGIN + BOARD_WIDTH + MARGIN, MARGIN, BOARD_WIDTH, BOARD_HEIGHT)
        prediction_board_rect = pygame.Rect(
            MARGIN + 2 * (BOARD_WIDTH + MARGIN), MARGIN, BOARD_WIDTH, BOARD_HEIGHT)

        ship_board_surface = self.screen.subsurface(ship_board_rect)
        hit_board_surface = self.screen.subsurface(hit_board_rect)

        self.displays = [ShipBoardDisplay(
            ship_board_surface, self.board), HitBoardDisplay(hit_board_surface, self.board)]

        if isinstance(shooter, PredictionShooter):
            prediction_board_surface = self.screen.subsurface(
                prediction_board_rect)
            self.displays.append(PredictionBoardDisplay(
                prediction_board_surface, self.board, self.shooter))

        self.info_surface = self.screen.subsurface(pygame.Rect(
            0, BOARD_HEIGHT + 2 * MARGIN, self.width, INFO_HEIGHT))
        self.shooter_name = shooter_name
        self.shots = 0

    def run(self):
        for display in self.displays:
            display.update()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.board.shoot(self.shooter.shoot(self.board))
                    self.shots += 1
                    for display in self.displays:
                        display.update()

            self.draw()
            pygame.display.update()

    def draw(self):
        self.screen.fill((255, 255, 255))
        for display in self.displays:
            display.display()
        self.draw_shooter_info()

    def draw_shooter_info(self):
        shots_text = pygame.font.SysFont('Arial', 24).render(
            "Shots: " + str(self.shots), True, (0, 0, 0))
        tw, th = shots_text.get_size()
        self.info_surface.blit(shots_text, (self.width / 2 - tw / 2, th / 2))

        shooter_name_text = pygame.font.SysFont('Arial', 24).render(
            "Shooter: " + self.shooter_name, True, (0, 0, 0))
        tw, th = shooter_name_text.get_size()
        self.info_surface.blit(
            shooter_name_text, (self.width / 2 - tw / 2, INFO_HEIGHT / 2 + th / 2))


monte_carlo_shooter = MonteCarloShooter(1000)
perceptron_shooter = load_model("models/perceptron.model")
#hybrid_shooter = HybridShooter(monte_carlo_shooter)
visualization = Visualization(perceptron_shooter, "Perceptron")
visualization.run()
