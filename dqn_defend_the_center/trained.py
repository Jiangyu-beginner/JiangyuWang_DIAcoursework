import vizdoom as vzd
import torch
import numpy as np
import cv2
from model import DQN
import time

def preprocess(img):
    img = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84))
    img = img.astype(np.float32) / 255.0
    return img

game = vzd.DoomGame()
game.load_config("defend_the_center.cfg")
game.set_window_visible(True)
game.init()

n_actions = game.get_available_buttons_size()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(n_actions).to(device)
model.load_state_dict(torch.load("dqn_model_defend.pth", map_location=device))
model.eval()

for episode in range(5):
    game.new_episode()
    state = preprocess(game.get_state().screen_buffer)
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device)

    total_reward = 0
    while not game.is_episode_finished():
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax().item()

        reward = game.make_action([1 if i == action else 0 for i in range(n_actions)], 4)
        total_reward += reward
        time.sleep(0.05)  
        if not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device)

    print(f"[EVAL {episode}] Total Reward: {total_reward}")

