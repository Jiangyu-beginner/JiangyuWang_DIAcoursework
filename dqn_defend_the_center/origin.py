from vizdoom import *
import random
import time


game = DoomGame()
game.load_config("defend_the_center.cfg")
game.set_window_visible(True)
game.init()


# [TURN_LEFT, TURN_RIGHT, ATTACK]
actions = [
    [1, 0, 0],  # turn left
    [0, 1, 0],  # right
    [0, 0, 1],  # shoot
    [0, 0, 0],  # none
    [1, 0, 1],  # left + shoot
    [0, 1, 1],  # right + shoot
]


episodes = 3
for i in range(episodes):
    print(f"Episode #{i + 1}")
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        reward = game.make_action(random.choice(actions))
        print("Reward:", reward)
        time.sleep(0.03)  

    print("Total reward:", game.get_total_reward())
    print("========================")

game.close()
