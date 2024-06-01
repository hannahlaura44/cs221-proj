from mdp import *

#Execute baseline model with rule based logic
# Simulating a day with decision-making based on SOC and price comparison
# BASELINE: rule-based policy:
# - if battery is less than 10% -> charge
# - if the discharge price is greater than the ride price in the *current timestep* -> discharge, otherwise, provide a ride. Note that the actual price is unknown for next timestep.
def main():
    env = EVEnvironment()
    state = env.reset()
    state_array = env.state_to_array(state)
    print(state)

    total_reward = 0
    for t in range(0, env.max_time - 1):
        if state_array[0] < 0.1:
            action = Actions.CHARGE
        elif state_array[1] > state_array[3]:
            action = Actions.DISCHARGE
        else:
            action = Actions.RIDE

        next_state, reward, done = env.step(action)
        state_array = env.state_to_array(next_state)
        print(f"Time: {t}, Action: {action.name}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()
