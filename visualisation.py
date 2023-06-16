from dataclasses import dataclass

import pygame

from game import *
from player import *
from prediction_shooter import *

TILE_TO_COLOR = {
    Tile.EMPTY: (255, 255, 255),
    Tile.MISS: (50, 153, 213),
    Tile.HIT: (220, 80, 70),
    Tile.SUNK: (50, 50, 50),
}

SHIP_COLOR = (160, 160, 160)

@dataclass
class BoardDisplayConfig:
    board_config: BoardConfig
    tile_size: int
    margin: int
    info_height: int

    @property
    def tiles_width(self):
        return self.tile_size * self.board_config.size
    
    @property
    def tiles_height(self):
        return self.tile_size * self.board_config.size

    @property
    def tiles_size(self):
        return (self.tiles_width, self.tiles_height)

    @property
    def board_width(self):
        return self.tiles_width + self.tile_size
    
    @property
    def board_height(self):
        return self.tiles_height + self.tile_size

    @property
    def board_size(self):
        return (self.board_width, self.board_height)


class BoardDisplay:
    def __init__(self, config, window_surface):
        self.window_surface = window_surface
        self.config = config
        self.board_surface = pygame.Surface(self.config.board_size)
        self.tile_surface = self.board_surface.subsurface(
            (self.config.tile_size, self.config.tile_size,
            self.config.tiles_width, self.config.tiles_height)
        )

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
        for r in range(1, self.config.board_config.size):
            y = r * self.config.tile_size
            x0 = 0
            x1 = self.config.tiles_width
            pygame.draw.line(self.tile_surface, grid_color, (x0, y), (x1, y))

        for c in range(1, self.config.board_config.size):
            x = c * self.config.tile_size
            y0 = 0
            y1 = self.config.tiles_height
            pygame.draw.line(self.tile_surface, grid_color, (x, y0), (x, y1))

    def draw_board_frame(self):
        pygame.draw.rect(self.tile_surface, (0, 0, 0),
                         (0, 0, self.config.tiles_width, self.config.tiles_height), 2)

    def draw_board_labels(self):
        self.draw_row_labels()
        self.draw_column_labels()

    def draw_row_labels(self):
        for i in range(1, 11):
            text = str(i)
            self.draw_label(text, (i, 0))

    def draw_column_labels(self):
        for i in range(1, 11):
            text = chr(ord('A') + i - 1)
            self.draw_label(text, (0, i))

    def draw_label(self, text, pos):
        text = pygame.font.SysFont('Arial', 14).render(text, True, (0, 0, 0))
        x, y = self.center_text_on_tile(text, pos)
        self.board_surface.blit(text, (x, y))
    
    def draw_text_on_tile(self, text, pos):
        text = pygame.font.SysFont('Arial', 14).render(text, True, (0, 0, 0))
        self.tile_surface.blit(text, self.center_text_on_tile(text, pos))

    def center_text_on_tile(self, text, pos):
        tw, th = text.get_size()
        x = pos[1] * self.config.tile_size + (self.config.tile_size - tw) / 2
        y = pos[0] * self.config.tile_size + (self.config.tile_size - th) / 2
        return (x, y)

    def tile_rect(self, pos):
        r, c = pos
        return pygame.Rect(c * self.config.tile_size, r * self.config.tile_size, self.config.tile_size, self.config.tile_size)

    def draw_tile(self, pos, color):
        pygame.draw.rect(self.tile_surface, color, self.tile_rect(pos))

    def draw_tiles(self):
        pass

class HitBoardDisplay(BoardDisplay):
    def __init__(self, config, window_surface, board):
        super().__init__(config, window_surface)
        self.board = board

    def draw_tiles(self):
        for pos in board_positions(self.board.config):
            color = TILE_TO_COLOR[self.board[pos]]
            self.draw_tile(pos, color)


class ShipBoardDisplay(BoardDisplay):
    def __init__(self, config, window_surface, board):
        super().__init__(config, window_surface)
        self.window = window_surface
        self.board = board

    def draw_tiles(self):
        for pos in board_positions(self.board.config):
            self.draw_tile(pos, self.get_tile_color(pos))

    def get_tile_color(self, pos):
        if self.board.ship_board[pos] is None:
            color = TILE_TO_COLOR[Tile.EMPTY]
        else:
            color = SHIP_COLOR

        if self.board[pos] == Tile.SUNK:
            color = TILE_TO_COLOR[Tile.SUNK]
        elif self.board[pos] == Tile.HIT:
            color = TILE_TO_COLOR[Tile.HIT]
        elif self.board[pos] == Tile.MISS:
            color = TILE_TO_COLOR[Tile.MISS]

        return color


class PredictionBoardDisplay(BoardDisplay):
    def __init__(self, config, window_surface, board, shooter):
        super().__init__(config, window_surface)
        self.board = board
        self.shooter = shooter

    def draw_tiles(self):
        predictions = self.shooter.predict_raw(self.board)
        for pos in board_positions(self.board.config):
            self.draw_tile(pos, predictions[pos])

    def draw_tile(self, pos, value):
        super().draw_tile(pos, self.get_prediction_color(value))
        if abs(value) > 0.01:
            self.draw_tile_value(pos, value)

    def get_prediction_color(self, value):
        target_color = (50, 93, 213)
        neg_target_color = (213, 93, 50)
        if value >= 0:
            diff = (255 - target_color[0], 255 -
                    target_color[1], 255 - target_color[2])
            draw_value = (1 - min(value, 1))**2
            draw_value = min(1, max(0, draw_value))
            color = (target_color[0] + diff[0] * draw_value, target_color[1] +
                    diff[1] * draw_value, target_color[2] + diff[2] * draw_value)
        else:
            diff = (255 - neg_target_color[0], 255 -
                    neg_target_color[1], 255 - neg_target_color[2])
            draw_value = (1+max(value, -1))**2
            draw_value = min(1, max(0, draw_value))
            color = (neg_target_color[0] + diff[0] * draw_value, neg_target_color[1] +
                    diff[1] * draw_value, neg_target_color[2] + diff[2] * draw_value)
        return color

    def draw_tile_value(self, pos, value):
        text = str(round(value, 2))
        self.draw_text_on_tile(text, pos)


class Visualization:
    def __init__(self, config, placer, shooter, shooter_name):
        pygame.init()

        self.config = config
        self.shooter = shooter
        self.placer = placer
        self.board = Board(self.placer.place_ships())

        self.display_count = 3 if isinstance(shooter, PredictionShooter) else 2
        self.width = self.get_width()
        self.height = self.get_height()

        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Battleship")

        ship_board_surface = self.screen.subsurface(self.get_nth_board_rect(0))
        hit_board_surface = self.screen.subsurface(self.get_nth_board_rect(1))

        self.displays = [
            ShipBoardDisplay(config, ship_board_surface, self.board), 
            HitBoardDisplay(config, hit_board_surface, self.board)
        ]

        if isinstance(shooter, PredictionShooter):
            prediction_board_surface = self.screen.subsurface(self.get_nth_board_rect(2))
            self.displays.append(
                PredictionBoardDisplay(config, prediction_board_surface, self.board, self.shooter)
            )

        self.info_surface = self.screen.subsurface(self.get_info_rect())
        self.shooter_name = shooter_name
        self.shots = 0

    def get_width(self):
        boards_width = self.config.board_width * self.display_count
        margin = 2 * self.config.margin + (self.display_count - 1) * self.config.margin
        return boards_width + margin

    def get_height(self):
        board_height = self.config.board_height
        info_height = self.config.info_height
        margin = 2 * self.config.margin
        return board_height + info_height + margin

    def get_nth_board_rect(self, n):
        x = self.config.margin + n * (self.config.board_width + self.config.margin)
        y = self.config.margin
        w = self.config.board_width
        h = self.config.board_height
        return pygame.Rect(x, y, w, h)

    def get_info_rect(self):
        x = 0
        y = self.config.board_height + 2 * self.config.margin
        w = self.width
        h = self.config.info_height
        return pygame.Rect(x, y, w, h)


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
            shooter_name_text, (self.width / 2 - tw / 2, self.config.info_height / 2 + th / 2))

