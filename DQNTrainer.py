from DQNEmulator import DQNEmulator
import torch
import torch.nn as nn
import torch.optim as optim
import melee
import math
import random
import time
from common import STATE_DIMENSION, NUM_ACTIONS, Action
from DQN import *
from ReplayMemory import *

# Hyperparameter Selections ********************
NUM_TRAINING_GAMES = 10000
NN_UPDATE = 2 # Update the model every 2 games
BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 0.09
EPS_END = 0.005
EPS_DECAY = 70
LEARNING_RATE = 1e-3
# **********************************************

class DQNTrainer():
    steps_done = 0
    device = 'cpu'
    actionList = list(Action)
    model_path = "dqn_model_v5/model"
    log_path = "dqn_model_v5/round_logs"

    def __init__(self, load_from_previous_model):
        self.policy_net = DQN(STATE_DIMENSION, NUM_ACTIONS).to(self.device)
        self.target_net = DQN(STATE_DIMENSION, NUM_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replayMemory = ReplayMemory(10000)

        if load_from_previous_model:
        	self.load_model()

    def save_model(self):
    	state = {
    		'policy_net': self.policy_net.state_dict(),
    		'target_net': self.target_net.state_dict(),
    		'optimizer': self.optimizer.state_dict(),
    		'steps_done': self.steps_done
    	}
    	torch.save(state, self.model_path)

    	# Replay memory needs to be saved separately
    	self.replayMemory.save_replay_mem_to_file()

    def load_model(self):
    	checkpoint = torch.load(self.model_path)
    	self.policy_net.load_state_dict(checkpoint['policy_net'])
    	self.target_net.load_state_dict(checkpoint['target_net'])
    	self.optimizer.load_state_dict(checkpoint['optimizer'])
    	self.steps_done = checkpoint['steps_done']

    	with open('dqn_model_v5/replay_memory', 'rb') as rm_input:
    		self.replayMemory = pickle.load(rm_input)


    def select_action(self, state):
    	sample = random.random()
    	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    	  math.exp(-1. * self.steps_done / EPS_DECAY)
    	self.steps_done +=1

    	if sample > eps_threshold:
    		with torch.no_grad():
    		# Pick action with the larger expected reward.
    			return self.policy_net(state).max(1)[1].view(1, 1)
    	else:
        	return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return
        transitions = self.replayMemory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the action that would be take by the NN
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def log_round_stats(self, ai_win, total_round_reward, length, stage, cpu_difficulty):
    	log_str = str(ai_win) + "," + str(total_round_reward) + "," + str(length) + "," + str(stage) + "," + str(cpu_difficulty) + "\n"
    	log_file = open(self.log_path, 'a')
    	log_file.write(log_str)
    	log_file.close()


    def run_training_loop(self):
        for episode in range(NUM_TRAINING_GAMES):
            # We make a new emulator for every game. This is becase
            # the emulator is somewhat flakey. This way, we can save
            # and shut down and just re-run the setup code so that if
            # for some reason the emulator breaks down, our training is
            # only defunct for that game one, not for the whole training set
            cpu_difficulty = random.randint(1, 9)
            emulator = DQNEmulator(self, cpu_difficulty, [melee.Stage.FINAL_DESTINATION])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            round_reward = 0

            # Initialize the environment and state
            # setup_new_roung will return the first state of the fight.
            start_time = time.time()
            end_time = 0
            curr_state, chosen_stage = emulator.setup_new_round()
            next_state = emulator.get_next_game_state()

            while True:
                # Select and perform an action
                emState = emulator.get_ai_state(curr_state)
                state = torch.from_numpy(emState.to_np_ndarray()).float().unsqueeze(0)
                action_idx = self.select_action(state)
                reward, enemy_win, ai_win, next_state = emulator.input_action_and_get_reward(self.actionList[action_idx], curr_state)
                
                # Don't update prev_state here. We don't want to change it since
                # we are skipping this input frame, essentially, to skip
                if (reward == None and enemy_win == None and ai_win == None and next_state == None):
                	continue

                round_reward += reward
                reward = torch.tensor([reward], device=device, dtype=torch.float)

                if enemy_win or ai_win:
                	end_time = time.time()
                	self.log_round_stats(ai_win, round_reward, end_time - start_time, chosen_stage, cpu_difficulty)

                # Store the transition in memory.
                curr_state_tensor = torch.from_numpy(emulator.get_ai_state(curr_state).to_np_ndarray()).float().unsqueeze(0)
                next_state_tensor = torch.from_numpy(emulator.get_ai_state(next_state).to_np_ndarray()).float().unsqueeze(0)
                self.replayMemory.push(curr_state_tensor, action_idx, next_state_tensor, reward)
                self.optimize_model()

                if enemy_win or ai_win:
                    emulator.end_game_and_shutdown_emulator()
                    break

                # Get next state
                curr_state = next_state
            time.sleep(5)

            # Update the target network, copying all weights and biases in DQN
            if (episode % NN_UPDATE == 0):
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

if __name__ == '__main__':
    DQNTrainer(False).run_training_loop()
