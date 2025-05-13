import vizdoom as vzd
import torch
import numpy as np
import cv2
import time

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def preprocess(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img.squeeze()

    img = cv2.resize(img, (84, 84))
    img = img.astype(np.float32) / 255.0
    return img


game = vzd.DoomGame()
game.load_config("basic.cfg")
game.set_doom_scenario_path("basic.wad")
game.set_window_visible(True)
game.init()

n_actions = game.get_available_buttons_size()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = DQN(n_actions).to(device)
model.load_state_dict(torch.load("dqn_basic.pt", map_location=device))
model.eval()

for episode in range(10):
    game.new_episode()
    total_reward = 0

    state = preprocess(game.get_state().screen_buffer)
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device)

    while not game.is_episode_finished():
        with torch.no_grad():
            q_values = model(state)
            action_idx = q_values.argmax().item()

        action = [0] * n_actions
        action[action_idx] = 1

        reward = game.make_action(action)
        total_reward += reward

        time.sleep(0.02)  

        if not game.is_episode_finished():
            next_state = preprocess(game.get_state().screen_buffer)
            state = torch.tensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

    print(f"[EVAL {episode}] Total Reward: {total_reward:.2f}")

game.close()
