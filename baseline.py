from mdp import *

# Execute baseline model with rule-based logic
# Simulating a day with decision-making based on SOC and price comparison
# BASELINE: rule-based policy:
# - if battery is less than 10% -> charge
# - if the discharge revenue is greater than the ride revenue in the *current timestep* -> discharge, otherwise, provide a ride. Note that the actual revenue is unknown for next timestep.

def run_episode(env, verbose=False):
    state = env.reset()
    state_array = env.state_to_array(state)  # Normalize the initial state
    total_reward = 0

    for t in range(env.max_time):
        if state['Current SOC'] < 10:
            action = Actions.CHARGE
        elif state['Discharge Price'] > state['Ride Price']:
            action = Actions.DISCHARGE
        else:
            action = Actions.RIDE
        
        next_state, reward, done = env.step(action)
        if verbose:
            print("---------------")
            print(f"State: {state}")
            print(f"Taking action {action}")
            print(f"Reward: {reward}")
            print(f"Next State: {next_state}")
        state = next_state
        state_array = env.state_to_array(state)  # Normalize the next state
        total_reward += reward
        if done:
            break

    return total_reward

def main():
    num_episodes = 100
    total_rewards = []

    for episode in range(num_episodes):
        env = EVEnvironment()
        episode_reward = run_episode(env, verbose=True)
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}, Total reward: {episode_reward}")

    average_total_reward = sum(total_rewards) / num_episodes
    print(f"Average total reward over {num_episodes} episodes: {average_total_reward}")

if __name__ == "__main__":
    main()
