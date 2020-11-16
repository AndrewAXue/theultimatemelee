import pygame
import sys
from common import State
from pygame.locals import *
from typing import List

HEIGHT = 450
WIDTH = 400
FPS = 60
FramePerSec = pygame.time.Clock()


class StateSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, player):
        super().__init__()
        self.surf = pygame.Surface((5, 5))
        if player:
            self.surf.fill((0, 255, 0))
        else:
            self.surf.fill((255, 0, 0))
        self.rect = self.surf.get_rect(center=(x, y))


class Visualizer:
    def __init__(self):
        pygame.init()
        self.displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Visualizer")
        self.all_sprites = pygame.sprite.Group()

    def generate_sprites(self, state: State) -> List[StateSprite]:
        return [
            StateSprite(state.player_x, state.player_y, True),
            StateSprite(state.enemy_x, state.enemy_y, False),
        ]

    def draw_states(self, states: List[State]):
        for state in states:
            sprites = self.generate_sprites(state)
            self.all_sprites.add(sprites)

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self.displaysurface.fill((0, 0, 0))

            for entity in self.all_sprites:
                self.displaysurface.blit(entity.surf, entity.rect)

            pygame.display.update()
            FramePerSec.tick(FPS)


Visualizer().draw_states(
    [
        State(
            player_x=100,
            player_y=100,
            player_x_vel=0,
            player_y_vel=0,
            player_dmg=0,
            player_current_action=None,
            enemy_x=200,
            enemy_y=100,
            enemy_x_vel=0,
            enemy_y_vel=0,
            enemy_dmg=0,
            enemy_current_action=None,
        )
    ]
)
