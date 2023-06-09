from abc import ABC, abstractmethod
from enum import Enum, auto

import random

import pygame

from game import *
from stats import *
from random_player import *

TILE_SIZE = 50
MARGIN = 50



TILE_COLORS = {
    Tile.EMPTY: (68, 114, 202),
    Tile.MISS: (10, 54, 157),
    Tile.HIT: (255, 0, 0),
    Tile.SHIP: (207, 222, 231),
}


class EventTypes(Enum):
    SHOOT = auto()
    END = auto()


class Event:
    def __init__(self, cause_player, event_type):
        self.cause_player = cause_player
        self.event_type = event_type


class Observer(ABC):
    @abstractmethod
    def update(self, event):
        pass


class Controls:
    def __init__(self, player_id, game_core):
        self.player_id = player_id
        self.game_core = game_core

    def shoot(self, pos):
        return self.game_core.shoot(self.player_id, pos)

    def is_self_caused(self, event):
        return event.cause_player == self.player_id


class GameCore:
    def __init__(self, boards):
        self.observers = []
        self.boards = boards
        self.current_player = 0
        self.ship_count = [board.count(Tile.SHIP)
                           for board in boards]
        self.finished = False

    def add_observer(self, observer):
        self.observers.append(observer)

    def get_controls(self, player_id):
        return Controls(player_id, self)

    def shoot(self, player_id, pos):
        if player_id != self.current_player:
            return False
        attacked = 1 - player_id
        if self.boards[attacked][pos] not in [Tile.SHIP, Tile.EMPTY]:
            return False

        event = Event(player_id, EventTypes.SHOOT)
        event.pos = pos
        if self.boards[attacked][pos] == Tile.EMPTY:
            self.boards[attacked][pos] = Tile.MISS
            event.result = Tile.MISS
        else:
            self.boards[attacked][pos] = Tile.HIT
            event.result = Tile.HIT
            self.ship_count[attacked] -= 1
            if self.ship_count[attacked] == 0:
                self.finished = True

        self.current_player = 1 - self.current_player
        self.notify_observers(event)
        if self.finished:
            event = Event(player_id, EventTypes.END)
            event.winner = player_id
            self.notify_observers(event)
        return True

    def notify_observers(self, event):
        for observer in self.observers:
            observer.update(event)


class Player(Observer):
    '''
    Observer that receives only events that are relevant to the player
    '''

    def __init__(self, player):
        self.player = player

    def update(self, event):
        if event.event_type == EventTypes.SHOOT:
            self.player.update(event)


class RandomPlayer(Observer):
    def __init__(self, controls):
        self.empty_tiles = [(i, j) for i in range(BOARD_SIZE)
                            for j in range(BOARD_SIZE)]
        self.controls = controls

    def shoot(self):
        pos = random.choice(self.empty_tiles)
        self.empty_tiles.remove(pos)
        return self.controls.shoot(pos)

    def update(self, event):
        pass


class HitObserver(Observer):
    def __init__(self, player_id, board):
        self.player_id = player_id
        self.board = board

    def update(self, event):
        if event.event_type == EventTypes.SHOOT and event.cause_player == self.player_id:
            self.board[event.pos] = event.result


class Game:
    def __init__(self):
        pygame.init()
        self.width = 3 * MARGIN + 2 * TILE_SIZE * BOARD_SIZE
        self.height = 2 * MARGIN + TILE_SIZE * BOARD_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.ship_board = RandomPlacer().place_ships()
        self.enemy_board = RandomPlacer().place_ships()

        self.hit_board = Board()

        self.game_core = GameCore([self.ship_board, self.enemy_board])
        self.game_core.add_observer(HitObserver(0, self.hit_board))
        self.game_core.add_observer(HitObserver(1, self.ship_board))

        self.player0 = RandomPlayer(self.game_core.get_controls(0))
        self.player1 = RandomPlayer(self.game_core.get_controls(1))

        self.game_core.add_observer(Player(self.player0))
        self.game_core.add_observer(Player(self.player1))

        self.players = [self.player0, self.player1]
        self.current_player = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not self.game_core.finished:
                self.players[self.current_player].shoot()
                self.current_player = 1 - self.current_player

            self.screen.fill((20, 20, 20))
            self.draw_board(self.ship_board, (MARGIN, MARGIN))
            self.draw_board(self.hit_board, (2 * MARGIN +
                            TILE_SIZE * BOARD_SIZE, MARGIN))

            pygame.display.update()
            pygame.time.wait(10)

        pygame.quit()

    def draw_board(self, board, pos):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                x = pos[0] + i * TILE_SIZE
                y = pos[1] + j * TILE_SIZE
                tile = board[i, j]
                color = TILE_COLORS[tile]
                pygame.draw.rect(self.screen, color, pygame.Rect(
                    x, y, TILE_SIZE, TILE_SIZE))


Game().run()