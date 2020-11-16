import melee
import time
import random
from emulator import Emulator
from common import Action, State

AI_PORT = 1
ENEMY_PORT = 2

class PolicyEmulator(Emulator):

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
		if (player_y < -4):
			below_platform_punishment = player_y * 10

		# enemy_lives_delta = int(prev_gamestate.player[ENEMY_PORT].stock - gamestate.player[ENEMY_PORT].stock)
		# ai_lives_delta = int(prev_gamestate.player[AI_PORT].stock - gamestate.player[AI_PORT].stock)
		enemy_game_win = 0
		ai_game_win = 0
		if gamestate.player[AI_PORT].stock == 0:
			enemy_game_win = 1
		if gamestate.player[ENEMY_PORT].stock == 0:
			ai_game_win = 1
		return (80 - player_x) + below_platform_punishment + enemy_damage_delta - ai_damage_delta,\
			   enemy_game_win, ai_game_win

	def game_loop(self):
		start_time = time.time()
		done_set = {}
		chosen_stage = None
		total_reward = 0

		prevstate = None
		prev_action = None
		while True:
			gamestate = self.console.step()

			if gamestate is None:
				continue

			# This allows us to see how long a frame takes to process
			if self.console.processingtime * 1000 > 12:
				print("WARNING: Last frame took " + str(self.console.processingtime * 1000) + "ms to process.")

			if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
				chosen_stage = None
				# Use every other frame to release all buttons
				reward, enemy_win, ai_win = self.calc_reward(prevstate, gamestate)
				self.trainer.reward_trainer(reward)
				total_reward += reward
				# Return list of all rewards achieved, whether the AI won, length of episode
				if enemy_win:
					self.console.stop()
					return total_reward, False, time.time() - start_time
				if ai_win:
					self.console.stop()
					return total_reward, True, time.time() - start_time

				if gamestate.frame % 2 == 1:
					if prev_action not in [Action.RUN_RIGHT, Action.RUN_LEFT]:
						self.ai_controller.release_all()
					continue

				action = self.trainer.get_action(self.get_ai_state(gamestate))
				if prev_action in [Action.RUN_RIGHT, Action.RUN_LEFT] and prev_action != action:
					self.ai_controller.release_all()

				self.input_action(gamestate.player[AI_PORT], action)
				prev_action = action
				prevstate = gamestate
			else:
				if not chosen_stage:
					chosen_stage = self.stages[random.randint(0, len(self.stages) - 1)]
				gamestate.player.items()

				melee.MenuHelper.menu_helper_simple(gamestate,
													self.ai_controller,
													character_selected=melee.Character.MARTH,
													stage_selected=chosen_stage,
													connect_code='',
													autostart=False,
													)
				if ENEMY_PORT in gamestate.player:
					if len(done_set) < 3 or gamestate.player[ENEMY_PORT].is_holding_cpu_slider:
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
