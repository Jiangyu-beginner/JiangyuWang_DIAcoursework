import vizdoom as vzd
import numpy as np
import random
import time
import cv2

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
actions = [[1 if i == j else 0 for i in range(n_actions)] for j in range(n_actions)]

for episode in range(3):
    game.new_episode()
    total_reward = 0
    while not game.is_episode_finished():
        reward = game.make_action(random.choice(actions), 4)
        total_reward += reward
        time.sleep(0.05)

    print(f"[RANDOM {episode}] Total Reward: {total_reward:.2f}")

game.close()
