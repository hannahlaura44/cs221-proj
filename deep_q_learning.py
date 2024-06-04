import numpy as np
import random
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please 
# use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
from tensorflow.keras.optimizers.legacy import Adam
from mdp import EVEnvironment, Actions
import os
import argparse

class DQNAgentSGD:
    def __init__(self, state_size, action_size, model_path=None, metadata_path=None, 
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = self._build_model()
        
        self.metadata_path = metadata_path
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.episodes = metadata['episodes']
                self.total_rewards = metadata['total_rewards']
                self.epsilon = metadata['epsilon']
            print(f"Loaded metadata from {metadata_path}")
        else:
            self.episodes = 0
            self.total_rewards = []
            self.epsilon = epsilon

    def _build_model(self):
        # Neural Network model for Deep Q-learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Actions))
        q_values = self.model.predict(state)
        return Actions(np.argmax(q_values[0]))

    def train(self, state, action: Actions, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action.value] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, model_path, metadata_path):
        self.model.save(model_path)
        metadata = {
            'episodes': self.episodes,
            'total_rewards': self.total_rewards,
            'epsilon': self.epsilon
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Model and metadata saved to {model_path} and {metadata_path}")

def train_agent(agent, env, save_model_path, save_metadata_path, episodes=1000):
    start_episode_num = agent.episodes
    for e in range(episodes):
        state = env.reset()
        state_array = np.reshape(env.state_to_array(state), [1, agent.state_size])
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state_array)
            next_state, reward, done = env.step(action)
            next_state_array = np.reshape(env.state_to_array(next_state), [1, agent.state_size])
            agent.train(state_array, action, reward, next_state_array, done)
            state_array = next_state_array
            total_reward += reward

        agent.episodes += 1
        agent.total_rewards.append(total_reward)
        print(f"Episode {agent.episodes}/{start_episode_num + episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Save the model and metadata at the end of each episode
        agent.save_model(save_model_path, save_metadata_path)
        if agent.episodes % 500 == 0:
            # save model weights every 500 iterations
            path = save_model_path.split(".keras")[0]
            agent.save_model(f"{path}_{agent.episodes}_episodes.keras", save_metadata_path)

    print("Training completed.")

    # Plot total rewards per episode
    plt.plot(agent.total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()


# Test the learned policy
def test_learned_policy(agent, env, num_episodes=100, verbose=False):
    total_rewards = []
    agent.epsilon = 0 # Use greedy policy

    for episode in range(num_episodes):
        state = env.reset()
        state_array = np.reshape(env.state_to_array(state), [1, agent.state_size])
        total_reward = 0
        done = False
        if verbose:
            env.plot_revenue()

        while not done:
            action = agent.act(state_array)
            next_state, reward, done = env.step(action)
            if verbose:
                print("---------------")
                print(f"State: {state}")
                print(f"Taking action {action}")
                print(f"Reward: {reward}")
                print(f"Total Reward (so far): {total_reward}")
                print(f"Next State: {next_state}")
            state = next_state
            state_array = np.reshape(env.state_to_array(state), [1, agent.state_size])
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    average_total_reward = sum(total_rewards) / num_episodes
    print(f"Total Rewards: {total_rewards}")
    print(f"Average total reward over {num_episodes} test episodes: {average_total_reward}")


def main(args):
    # Initialize environment and agent
    env = EVEnvironment()
    state_size = 5  # The length of the state array
    action_size = len(Actions)
    agent = DQNAgentSGD(state_size, action_size, model_path=args.load_model_path, metadata_path=args.load_metadata_path)

    if args.train:
        # Train the agent
        train_agent(agent, env, episodes=1000, save_model_path=args.save_model_path, save_metadata_path=args.save_metadata_path)
    elif args.test:
        # Run the test
        test_learned_policy(agent, env, num_episodes=10, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Agent Training and Testing')
    parser.add_argument('--train', action='store_true', help='Train the DQN agent')
    parser.add_argument('--test', action='store_true', help='Test the DQN agent')
    parser.add_argument('--load_model_path', type=str, help='Path to model weights')
    parser.add_argument('--load_metadata_path', type=str, help='Path to model metadata')
    parser.add_argument('--save_model_path', type=str, help='Save path for model weights')
    parser.add_argument('--save_metadata_path', type=str, help='Save path for model metadata')
    args = parser.parse_args()
    main(args)

    # example train command:
    # python deep_q_learning.py --train --save_model_path dqn_model.keras --save_metadata_path dqn_metadata.json

    # test:
    # python deep_q_learning.py --test --load_model_path dqn_model.keras --load_metadata_path dqn_metadata.json
