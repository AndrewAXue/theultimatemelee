import signal
import sys

import melee

from common import Action, State

# Path to slippi
SLIPPI_EXECUTABLE_PATH = None
assert SLIPPI_EXECUTABLE_PATH, 'Path to slippi executable must be configured. e.g. "C:\\Users\\andre\\Downloads\\FM-Slippi"'
# Path to melee ISO
MELEE_ISO = None
assert MELEE_ISO, 'Path to Melee ISO must be configured. e.g. "C:\\Users\\andre\\Documents\\DolphinGames\\smash.iso"'
AI_PORT = 1
ENEMY_PORT = 2
STAGES = [melee.Stage.FINAL_DESTINATION, melee.Stage.BATTLEFIELD, melee.Stage.POKEMON_STADIUM]


class Emulator:
    def __init__(self, trainer, enemy_difficulty, possible_stages = None):
        assert 1 <= enemy_difficulty <= 9
        self.trainer = trainer
        self.console_setup()
        self.move = 0
        self.enemy_difficulty = enemy_difficulty
        self.stages = possible_stages or STAGES

    # Makes it so that Dolphin will get killed when you ^C
    def signal_handler(self, sig, frame):
        self.console.stop()
        print("Shutting down cleanly...")
        sys.exit(0)

    def console_setup(self):
        # Game console
        self.console = melee.Console(path=SLIPPI_EXECUTABLE_PATH, slippi_address="127.0.0.1", slippi_port=51441,
                                     blocking_input=False, polling_mode=False)
        # Your / Our AI's controller
        self.ai_controller = melee.Controller(console=self.console, port=AI_PORT, type=melee.ControllerType.STANDARD)
        self.enemy_controller = melee.Controller(console=self.console, port=ENEMY_PORT,
                                                 type=melee.ControllerType.STANDARD)

        signal.signal(signal.SIGINT, self.signal_handler)

        self.console.run(iso_path=MELEE_ISO)

        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)

        if not self.ai_controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)

        if not self.enemy_controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        print("Controller connected")

    # Implemented separately by child classes
    def calc_reward(self, prev_gamestate, gamestate):
        pass

    def get_ai_state(self, gamestate: melee.GameState) -> State:
        # Given the raw gamestate, transform into an State object
        ai_player: melee.PlayerState = gamestate.player[AI_PORT]
        enemy_player: melee.PlayerState = gamestate.player[ENEMY_PORT]
        return State(
            player_x = ai_player.x, player_y=ai_player.y, player_x_vel=ai_player.speed_x_attack,
            player_y_vel=ai_player.speed_y_attack, player_dmg=ai_player.percent,
            player_current_action=ai_player.action, player_direction_facing=ai_player.facing, 
            player_jumps_left=ai_player.jumps_left, player_on_ground=ai_player.on_ground, 
            player_off_stage=ai_player.off_stage, player_hitlag=ai_player.hitlag,
            player_hitstun=ai_player.hitstun_frames_left,

            enemy_x=enemy_player.x, enemy_y=enemy_player.y, enemy_x_vel=enemy_player.speed_x_attack,
            enemy_y_vel=enemy_player.speed_y_attack, enemy_dmg=enemy_player.percent,
            enemy_current_action=enemy_player.action, enemy_direction_facing=enemy_player.facing,
            enemy_jumps_left=enemy_player.jumps_left, enemy_on_ground=enemy_player.on_ground,
            enemy_off_stage=enemy_player.off_stage, enemy_hitlag=enemy_player.hitlag,
            enemy_hitstun=enemy_player.hitstun_frames_left,
        )

    def input_action(self, ai_player, action: Action):
        if not action:
            return
        for button in action.value:
            if isinstance(button, tuple):
                self.ai_controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, *button)
            else:
                self.ai_controller.press_button(button)