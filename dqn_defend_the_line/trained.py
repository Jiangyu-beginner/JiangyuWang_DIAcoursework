import vizdoom as vzd
import torch
import numpy as np
import cv2
import time
from model import DQN  


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = DQN(n_actions).to(device)
model.load_state_dict(torch.load("dqn_model_defend_line.pth", map_location=device))
model.eval()


for episode in range(3):
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
        reward = game.make_action(action, 4)
        total_reward += reward

        time.sleep(0.07)  

        if not game.is_episode_finished():
            next_state = preprocess(game.get_state().screen_buffer)
            state = torch.tensor(next_state).unsqueeze(0).unsqueeze(0).to(device)

    print(f"[DEFEND EVAL {episode}] Total Reward: {total_reward:.2f}")

game.close()
