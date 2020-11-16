from emulator import Emulator
import melee
import random
from common import STATE_DIMENSION, NUM_ACTIONS, Action

AI_PORT = 1
ENEMY_PORT = 2

class DQNEmulator(Emulator):
    chosen_stage = None
    prev_action = None
    skippedLastFrame = False

    def setup_new_round(self):
        done_set = {}
        self.chosen_stage = None
        gamestate = self.console.step()

        while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            if not self.chosen_stage:
                self.chosen_stage = self.stages[random.randint(0, len(self.stages) - 1)]
            gamestate.player.items()
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.ai_controller,
                                                character_selected=melee.Character.MARTH,
                                                stage_selected=self.chosen_stage,
                                                connect_code='',
                                                autostart=False,
                                                )
                
            if 2 in gamestate.player:
                if len(done_set) < 3 or gamestate.player[2].is_holding_cpu_slider:
                    ret = melee.MenuHelper.choose_character(character=melee.Character.MARTH,
                                                            gamestate=gamestate,
                                                            controller=self.enemy_controller,
                                                            cpu_level=self.enemy_difficulty)
                    if ret:
                        done_set[ret] = 1
                    if self.enemy_difficulty == 1:
                        done_set['done1'] = 1
                        done_set['done2'] = 1

                else:
                    melee.MenuHelper.choose_character(character=melee.Character.MARTH,
                                                      gamestate=gamestate,
                                                      controller=self.enemy_controller,
                                                      cpu_level=0,
                                                      start=True)
            gamestate = self.console.step()
 
        return (gamestate, self.chosen_stage)

    def calc_reward(self, prev_gamestate, gamestate):
        if not prev_gamestate:
            return 0, 0, 0
        # Calculate the reward by looking at the deltas
        enemy_damage_delta = int(gamestate.player[ENEMY_PORT].percent - prev_gamestate.player[ENEMY_PORT].percent) * 10
        ai_damage_delta = int(gamestate.player[AI_PORT].percent - prev_gamestate.player[AI_PORT].percent) * 10

        # When respawning the "delta" is negative.
        if enemy_damage_delta < 0:
            enemy_damage_delta = 0
        if ai_damage_delta < 0:
            ai_damage_delta = 0

        player_x = abs(gamestate.player[AI_PORT].x)
        player_y = gamestate.player[AI_PORT].y

        below_platform_punishment = 0

        # Only update if it's below negative 4, since some landings on jumps on the platform
        # seem to cause cause the y-value to go up to -3.
        if (player_y < -4):
            below_platform_punishment = player_y * 10

        enemy_game_win = 0
        ai_game_win = 0
        if gamestate.player[AI_PORT].stock == 0:
            enemy_game_win = 1
        if gamestate.player[ENEMY_PORT].stock == 0:
            ai_game_win = 1
        return (80 - player_x) + below_platform_punishment + enemy_damage_delta - ai_damage_delta,\
               enemy_game_win, ai_game_win


    def end_game_and_shutdown_emulator(self):
        self.console.stop()

    def input_action_and_get_reward(self, action, curr_gamestate):
        # Every other frame, release all buttons only if the AI isn't running
        if not self.skippedLastFrame:
            if self.prev_action not in [Action.RUN_RIGHT, Action.RUN_LEFT]:
                self.ai_controller.release_all()
            g = self.console.step() # Step so we can skip the next frame on our next get frame call
            reward, e_win, ai_win = self.calc_reward(curr_gamestate, g)
            self.skippedLastFrame = True

            # Do not skip this frame in the event of a win, we need to log the last move
            if (e_win or ai_win):
                return reward, e_win, ai_win, g
            return None, None, None, None

        if self.prev_action in [Action.RUN_RIGHT, Action.RUN_LEFT] and self.prev_action != action:
            self.ai_controller.release_all()

        self.input_action(curr_gamestate.player[AI_PORT], action)
        self.prev_action = action
        next_game_state = self.console.step()
        reward, e_win, ai_win = self.calc_reward(curr_gamestate, next_game_state)
        self.skippedLastFrame = False
        return reward, e_win, ai_win, next_game_state

    def get_next_game_state(self):
        return self.console.step()
