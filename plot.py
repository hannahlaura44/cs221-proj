import json
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

metadata_path = "dqn_metadata.json"
# metadata_path = "dqn_metadata_backup.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
f.close()

total_rewards = metadata['total_rewards']
# Calculate the moving average with a window size of 100
window_size = 100
ma_data = moving_average(total_rewards, window_size)

# Plot the original data and the moving average
plt.figure(figsize=(10, 6))
plt.plot(total_rewards)
plt.plot(range(window_size - 1, len(total_rewards)), ma_data, label='Moving Average (over 100 episodes)', color='red')
plt.xlabel('Episode')
plt.ylabel('Total Rewards')
plt.title('Training: Total Rewards Per Episode')
plt.legend()
plt.show()
