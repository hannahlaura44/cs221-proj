from mdp import *
import random
import matplotlib.pyplot as plt

class LinearQAgent:
    def __init__(self, state_size, action_size, eta=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.eta = eta
        self.gamma = gamma
        self.weights = {action.value: np.random.randn(self.state_size) for action in Actions}

    def get_q_value(self, state, action):
        return np.dot(self.weights[action.value], state)

    def update(self, state, action, reward, next_state, done):
        q_pred = self.get_q_value(state, action)
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * max(self.get_q_value(next_state, a) for a in Actions)
        self.weights[action.value] -= self.eta * (q_pred - q_target) * state

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.choice(list(Actions))
        q_values = {action: self.get_q_value(state, action) for action in Actions}
        return max(q_values, key=q_values.get)

# Initialize environment and agent
env = EVEnvironment()
state_size = 6  # Added by Marcelo to take time in consideration
action_size = len(Actions)
agent = LinearQAgent(state_size, action_size)

episodes = 1000
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

total_rewards = []


for e in range(episodes):
    state = env.reset()
    state_array = env.state_to_array(state)
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(state_array, epsilon)
        next_state, reward, done = env.step(action)
        next_state_array = env.state_to_array(next_state)
        agent.update(state_array, action, reward, next_state_array, done)
        state_array = next_state_array
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    total_rewards.append(total_reward)
    print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

print("Training completed.")
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()
