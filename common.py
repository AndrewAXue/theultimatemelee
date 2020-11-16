from enum import Enum
from melee.enums import Button
import numpy as np

# Main stick directions
up = (0.5, 1)
down = (0.5, 0)
right = (1, 0.5)
left = (0, 0.5)
up_tilt = (0.5, 0.75)
down_tilt = (0.5, 0.25)
right_tilt = (0.75, 0.5)
left_tilt = (0.25, 0.5)


class Action(Enum):
    '''
        for action: Action
        action.value defines a list of buttons to input
        I assume that a tuple is a main stick input
    '''
    # Specials
    NEUTRAL_B = [Button.BUTTON_B]
    LEFT_B = [left, Button.BUTTON_B]
    RIGHT_B = [right, Button.BUTTON_B]
    UP_B = [up, Button.BUTTON_B]
    DOWN_B = [down, Button.BUTTON_B]
    # Smashes
    UP_SMASH = [up, Button.BUTTON_A]
    LEFT_SMASH = [left, Button.BUTTON_A]
    RIGHT_SMASH = [right, Button.BUTTON_A]
    DOWN_SMASH = [down, Button.BUTTON_A]
    # Tilts
    UP_TILT = [up_tilt, Button.BUTTON_A]
    LEFT_TILT = [left_tilt, Button.BUTTON_A]
    RIGHT_TILT = [right_tilt, Button.BUTTON_A]
    DOWN_TILT = [down_tilt, Button.BUTTON_A]
    # Jab
    JAB = [Button.BUTTON_A]
    # Movement
    RUN_RIGHT = [right]
    RUN_LEFT = [left]
    JUMP = [Button.BUTTON_X]  # SHORT HOP?
    # Defensive options
    SHIELD = [Button.BUTTON_L]
    ROLL_RIGHT = [right, Button.BUTTON_L]
    ROLL_LEFT = [left, Button.BUTTON_L]
    SPOT_DODGE = [down, Button.BUTTON_L]
    # Grabs
    GRAB = [Button.BUTTON_Z]


NUM_ACTIONS = len(list(Action))


class State:
    """
    Overview of the game's state at a certain point in time
    """

    def __init__(
            self,
            player_x,
            player_y,
            player_x_vel,
            player_y_vel,
            player_dmg,
            player_current_action,
            player_direction_facing,
            player_jumps_left,
            player_on_ground,
            player_off_stage,
            player_hitlag, 
            player_hitstun,
            enemy_x,
            enemy_y,
            enemy_x_vel,
            enemy_y_vel,
            enemy_dmg,
            enemy_current_action,
            enemy_direction_facing,
            enemy_jumps_left,
            enemy_on_ground,
            enemy_off_stage,
            enemy_hitlag, 
            enemy_hitstun,
    ):
        self.player_x = player_x
        self.player_y = player_y
        self.player_x_vel = player_x_vel
        self.player_y_vel = player_y_vel
        self.player_dmg = player_dmg
        self.player_current_action = player_current_action
        self.player_direction_facing = player_direction_facing
        self.player_jumps_left = player_jumps_left
        self.player_on_ground = player_on_ground
        self.player_off_stage = player_off_stage
        self.player_hitlag = player_hitlag
        self.player_hitstun = player_hitstun

        self.enemy_x = enemy_x
        self.enemy_y = enemy_y
        self.enemy_x_vel = enemy_x_vel
        self.enemy_y_vel = enemy_y_vel
        self.enemy_dmg = enemy_dmg
        self.enemy_current_action = enemy_current_action
        self.enemy_direction_facing = enemy_direction_facing
        self.enemy_jumps_left = enemy_jumps_left
        self.enemy_on_ground = enemy_on_ground
        self.enemy_off_stage = enemy_off_stage
        self.enemy_hitlag = enemy_hitlag
        self.enemy_hitstun = enemy_hitstun

    def __str__(self):
        return f'''
        player_x = {self.player_x}
        player_y = {self.player_y}
        player_x_vel = {self.player_x_vel}
        player_y_vel = {self.player_y_vel}
        player_dmg = {self.player_dmg}
        player_current_action = {self.player_current_action}
        player_direction_facing = {self.player_direction_facing}
        player_jumps_left = {self.player_jumps_left}
        player_on_ground = {self.player_on_ground}
        player_off_stage = {self.player_off_stage}
        player_hitlag = {self.player_hitlag}
        player_hitstun = {self.player_hitstun}

        enemy_x = {self.enemy_x}
        enemy_y = {self.enemy_y}
        enemy_x_vel = {self.enemy_x_vel}
        enemy_y_vel = {self.enemy_y_vel}
        enemy_dmg = {self.enemy_dmg}
        enemy_current_action = {self.enemy_current_action}
        enemy_direction_facing = {self.enemy_direction_facing}
        enemy_jumps_left = {self.enemy_jumps_left}
        enemy_on_ground = {self.enemy_on_ground}
        enemy_off_stage = {self.enemy_off_stage}
        enemy_hitlag = {self.enemy_hitlag}
        enemy_hitstun = {self.enemy_hitstun}
        '''

    def to_np_ndarray(self):
        return np.array([
            self.player_x,
            self.player_y,
            self.player_x_vel,
            self.player_y_vel,
            float(self.player_dmg),
            float(self.player_current_action.value),
            float(self.player_direction_facing),
            float(self.player_jumps_left),
            float(self.player_on_ground),
            float(self.player_off_stage),
            float(self.player_hitlag),
            float(self.player_hitstun),

            self.enemy_x,
            self.enemy_y,
            self.enemy_x_vel,
            self.enemy_y_vel,
            float(self.enemy_dmg),
            float(self.enemy_current_action.value),
            float(self.enemy_direction_facing),
            float(self.enemy_jumps_left),
            float(self.enemy_on_ground),
            float(self.enemy_off_stage),
            float(self.enemy_hitlag),
            float(self.enemy_hitstun),
        ])


STATE_DIMENSION = 12 * 2
