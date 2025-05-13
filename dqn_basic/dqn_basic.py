import vizdoom as vzd
import torch
import numpy as np
import cv2  

import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random
from hyperparameter import *


    
# CNN
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # [1,84,84] -> [32,20,20]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # -> [64,9,9]
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # -> [64,7,7]

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 1, 84, 84] -> [B, 32, 20, 20]
        x = F.relu(self.conv2(x))  # -> [B, 64, 9, 9]
        x = F.relu(self.conv3(x))  # -> [B, 64, 7, 7]
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Qvalue







def preprocess(img):
    if img.ndim == 3 and img.shape[0] == 3:
        #  [C, H, W] → [H, W, C]
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img.squeeze()

    img = cv2.resize(img, (84, 84))
    img = img.astype(np.float32) / 255.0
    return img

# initial
game = vzd.DoomGame()
game.load_config("basic.cfg")
game.set_doom_scenario_path("basic.wad")
game.init()

# get action
n_actions = game.get_available_buttons_size()

# buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


model = DQN(n_actions)
target_model = DQN(n_actions)
target_model.load_state_dict(model.state_dict())  
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

# Epsilon-greedy 
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            return model(state).argmax(1).item()


epsilon = EPSILON_START

for episode in range(NUM_EPISODES):
    game.new_episode()
    total_reward = 0
    step = 0

    state = preprocess(game.get_state().screen_buffer)
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 84, 84]

    while not game.is_episode_finished():
        # choose action
        action_idx = select_action(state, epsilon)
        action = [0] * n_actions
        action[action_idx] = 1

        # do action
        reward = game.make_action(action)
        done = game.is_episode_finished()
        total_reward += reward

        # get next action
        if not done:
            next_state = preprocess(game.get_state().screen_buffer)
            next_state = torch.tensor(next_state).unsqueeze(0).unsqueeze(0)
        else:
            next_state = torch.zeros_like(state)

        
        memory.push(state.squeeze(0).numpy(), action_idx, reward, next_state.squeeze(0).numpy(), done)

     
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_model(next_states).max(1)[0]
            expected_q = rewards + GAMMA * next_q_values * (1 - dones.float())

            loss = nn.MSELoss()(q_values, expected_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        step += 1

    # epsilon 
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

   
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"[Episode {episode}] Reward: {total_reward:.2f} | Steps: {step} | Epsilon: {epsilon:.3f}")

torch.save(model.state_dict(), "dqn_basic.pt")

