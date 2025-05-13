from vizdoom import *
import random
import time

game = DoomGame()
game.load_config("basic.cfg")
game.set_window_visible(True)
game.init()

actions = [
    [1, 0, 0],  # move left
    [0, 1, 0],  # move right
    [0, 0, 1],  # shoot
    [1, 0, 1],  # move left + shoot
    [0, 1, 1],  # move right + shoot
]

episodes = 3
for i in range(episodes):
    print(f"Episode #{i+1}")
    game.new_episode()

    while not game.is_episode_finished():
        reward = game.make_action(random.choice(actions))
        print("Reward:", reward)
        time.sleep(0.02)

    print("Total reward:", game.get_total_reward())
    print("=======================")

game.close()
