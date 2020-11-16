import pickle
import random

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from PolicyEmulator import PolicyEmulator
from Trainer import Trainer
from common import STATE_DIMENSION, NUM_ACTIONS, Action

GAMMA = 0.99
MODEL_PATH = 'policy_models/v1'

# Randomly picks a number in [ENEMY_DIFFICULTY_LO, ENEMY_DIFFICULTY_HI] for the enemy's difficulty this episode
ENEMY_DIFFICULTY_LO = 8
ENEMY_DIFFICULTY_HI = 9

# Per episode, emulator will choose a random stage from STAGES
STAGES = [melee.Stage.FINAL_DESTINATION]

class Policy(nn.Module, Trainer):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(STATE_DIMENSION, 128)
        self.affine2 = nn.Linear(128, NUM_ACTIONS)
        self.eps = np.finfo(np.float32).eps.item()
        self.saved_log_probs = []
        self.rewards = []
        self.total_rewards = []
        self._verify_and_write_metadata()

    def _verify_and_write_metadata(self):
        # Verifies that metadata at this location, if it exists is the same as current metadata. Also writes metadata
        existing_metadata = None
        try:
            with open(MODEL_PATH + '_metadata') as file:
                existing_metadata = file.read()
        except FileNotFoundError:
            pass
        reward_f_serialized = pickle.dumps(PolicyEmulator.calc_reward)
        current_metadata = \
            f'''
            GAMMA {GAMMA}
            ENEMY_DIFFICULTY_LO {ENEMY_DIFFICULTY_LO}
            ENEMY_DIFFICULTY_HI {ENEMY_DIFFICULTY_HI}
            STAGES {[x.name for x in STAGES]}
            ACTION_SPACE {[x.name for x in list(Action)]}
            REWARD_FUNCTION_SERIALIZED {reward_f_serialized}
            '''
        if existing_metadata:
            assert existing_metadata == current_metadata, "There exists a model with the same name with different metadata. Rename this model"
        with open(MODEL_PATH + '_metadata', 'w') as file:
            file.write(current_metadata)


    def forward(self, x):
        x = self.affine1(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_action(self, state):
        state = torch.from_numpy(state.to_np_ndarray()).float().unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        chosen_action = list(Action)[action.item()]
        return chosen_action

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def reward_trainer(self, reward):
        self.rewards.append(reward)

    def save_to(self):
        state = {
            'total_rewards': self.total_rewards,
            'affine1': self.affine1.state_dict(),
            'affine2': self.affine2.state_dict(),
        }
        torch.save(state, MODEL_PATH)

    def load_from(self):
        try:
            state = torch.load(MODEL_PATH)
            self.total_rewards = state['total_rewards']
            self.affine1.load_state_dict(state['affine1'])
            self.affine2.load_state_dict(state['affine2'])
        except FileNotFoundError:
            pass

    def start_train(self):
        self.load_from()
        running_reward = None
        while True:
            difficulty = random.randint(ENEMY_DIFFICULTY_LO, ENEMY_DIFFICULTY_HI)
            stage_choice = STAGES[random.randint(0, len(STAGES) - 1)]
            emulator = PolicyEmulator(self, difficulty, [stage_choice])
            total_reward, did_ai_win, episode_leng = emulator.game_loop()
            if not running_reward:
                running_reward = total_reward
            running_reward = 0.05 * total_reward + (1 - 0.05) * running_reward
            self.finish_episode()
            self.total_rewards.append(total_reward)
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                len(self.total_rewards), total_reward, running_reward), flush=True)
            with open(MODEL_PATH + '_reward.csv', 'a') as f:
                if len(self.total_rewards) == 1:
                    f.write('episode_num,total_reward,did_ai_win,episode_length,enemy_difficulty,stage')
                f.write(f'{len(self.total_rewards)},{total_reward},{1 if did_ai_win else 0},{episode_leng},{difficulty},{stage_choice}')
            self.save_to()

if __name__ == '__main__':
    Policy().start_train()