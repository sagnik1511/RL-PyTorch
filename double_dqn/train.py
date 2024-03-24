import gym
import numpy as np

from double_dqn.core.q_agent import DDQNAgent


if __name__ == "__main__":

    # Defining parameters
    ENV_NAME = "LunarLander-v2"
    NUM_EPISODES = 10000
    OBS_SPACE_SIZE = 8
    ACTION_SPACE_SIZE = 4
    MEM_SIZE = 1000000
    BATCH_SIZE = 64
    GAMMA = 0.9
    LR = 1e-4
    EPS = 0.99
    EPS_DECAY = 5e-7
    EPS_MIN = 0.05
    UPDATE_NETWORK_COUNTER = 2000
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
    scores = []
    best_score = -float("inf")
    for episode_idx in range(NUM_EPISODES):
        episode_score = 0
        done = False
        obs, _ = env.reset()
        while not done:

            # Get actions from Agent
            action = agent.get_action(obs)

            # Get interactions and rewards from next state
            obs_, reward, done, truncated, _ = env.step(action)

            # Add the data into Replay Buffer
            agent.replay_buffer.push(obs, action, reward, obs_, done)

            # Train the Agent
            agent.train()
            episode_score += reward

            # Mark the next_state as the current state
            obs = obs_
        scores.append(episode_score)
        moving_average = np.mean(scores[-100:])
        print(f"Average Score [EPISODE] : {episode_idx + 1}] -> {moving_average}")
        if episode_idx > 0 and episode_idx % 10 == 0:
            if best_score < moving_average:
                best_score = moving_average
                agent.save_artifact()
    env.close()
