import gym
import numpy as np
import matplotlib.pyplot as plt

from double_dqn.core.q_agent import DDQNAgent


class DynamicChart:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot(self.x_data, self.y_data)
        plt.ion()  # Turn on interactive mode

    def update(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()


if __name__ == "__main__":

    # Defining parameters
    ENV_NAME = "CartPole-v1"
    NUM_EPISODES = 100000
    OBS_SPACE_SIZE = 4
    ACTION_SPACE_SIZE = 2
    MEM_SIZE = 100000
    BATCH_SIZE = 64
    GAMMA = 0.9
    LR = 5e-4
    EPS = 0.999
    EPS_DECAY = 4e-5
    EPS_MIN = 0.1
    UPDATE_NETWORK_COUNTER = 1000
    CHKPT_DIR = "runs/exp/artifact/"

    env = gym.make(ENV_NAME)

    agent = DDQNAgent(
        MEM_SIZE,
        BATCH_SIZE,
        OBS_SPACE_SIZE,
        ACTION_SPACE_SIZE,
        GAMMA,
        LR,
        EPS,
        EPS_DECAY,
        EPS_MIN,
        UPDATE_NETWORK_COUNTER,
        CHKPT_DIR,
    )

    # Dnnamic Chart
    dynamic_chart = DynamicChart()

    scores = []
    best_score = -float("inf")
    for episode_idx in range(NUM_EPISODES):
        episode_score = 0
        done = False
        truncated = False
        obs, _ = env.reset()
        while not done and not truncated:

            # Get actions from Agent
            action = agent.get_action(obs)

            # Get interactions and rewards from next state
            obs_, reward, done, truncated, _ = env.step(action)

            done_or_truncated = done | truncated

            # Add the data into Replay Buffer
            agent.replay_buffer.push(obs, action, reward, obs_, done_or_truncated)

            # Train the Agent
            agent.train()
            episode_score += reward

            # Mark the next_state as the current state
            obs = obs_
        scores.append(episode_score)
        moving_average = np.mean(scores[-50:])
        print(f"Average Score [EPISODE : {episode_idx + 1}] -> {moving_average}")
        dynamic_chart.update(episode_idx, moving_average)
        if episode_idx > 0 and episode_idx % 500 == 0:
            print(f"Current Epsilon : {agent.eps}")
        if episode_idx > 0 and episode_idx % 10 == 0:
            if best_score < moving_average:
                best_score = moving_average
                agent.save_artifact()
    env.close()
