import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import random
from model import DQN
from collections import deque


EPISODES = 600
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000
FRAME_SKIP = 4


def preprocess(img):
    if img is None:
        return np.zeros((84, 84), dtype=np.float32)
    img = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84))
    return img.astype(np.float32) / 255.0


game = vzd.DoomGame()
game.load_config("defend_the_line.cfg")
game.set_window_visible(True)
game.init()
n_actions = game.get_available_buttons_size()


policy_net = DQN(n_actions)
target_net = DQN(n_actions)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
step_count = 0

for episode in range(EPISODES):
    game.new_episode()
    state = preprocess(game.get_state().screen_buffer)
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0)

    total_reward = 0

    while not game.is_episode_finished():
        step_count += 1

        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax().item()

        reward = game.make_action([1 if i == action else 0 for i in range(n_actions)], FRAME_SKIP)
        done = game.is_episode_finished()

        next_state = preprocess(game.get_state().screen_buffer) if not done else np.zeros((84, 84), dtype=np.float32)
        next_state = torch.tensor(next_state).unsqueeze(0).unsqueeze(0)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

            b_states = torch.cat(b_states)
            b_next_states = torch.cat(b_next_states)
            b_actions = torch.tensor(b_actions).unsqueeze(1)
            b_rewards = torch.tensor(b_rewards, dtype=torch.float32).unsqueeze(1)
            b_dones = torch.tensor(b_dones, dtype=torch.float32).unsqueeze(1)

            q_values = policy_net(b_states).gather(1, b_actions)
            next_q = target_net(b_next_states).max(1)[0].unsqueeze(1)
            target = b_rewards + GAMMA * next_q * (1 - b_dones)

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"[Episode {episode}] Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")


torch.save(policy_net.state_dict(), "dqn_model_defend_line.pth")

