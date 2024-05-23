from mdp import *

#Execute baseline model with rule based logic
def main():
    env = EVEnvironment()
    state = env.reset()
    state_array = env.state_to_array(state)  # Normalize the initial state
    print(state)

    # Simulating a day with decision-making based on SOC and price comparison
    total_reward = 0
    for step in range(1, env.max_time):  # Adjusted range to 47 to avoid out-of-bounds error
        if state_array[0] < 0.1:
            action = Actions.CHARGE
        elif state_array[1] > state_array[3]:
            action = Actions.DISCHARGE
        else:
            action = Actions.RIDE
            
        next_state, reward, done = env.step(action)
        state_array = env.state_to_array(next_state)  # Normalize the next state
        print(f"Step: {step}, Action: {action.name}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()