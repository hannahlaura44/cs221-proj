import numpy as np
import enum

np.random.seed(42)

class Actions(enum.Enum):
    CHARGE = 0
    DISCHARGE = 1
    RIDE = 2

class EVEnvironment:
    def __init__(self, initial_soc=0.5, max_soc=1.0, min_soc=0.0, charge_rate=0.1, discharge_rate=0.1, ride_energy=0.2):
        self.current_soc = initial_soc
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.ride_energy = ride_energy
        self.time = 0
        self.max_time = 24  # 24 hours in a day
        self.prices = self.generate_prices()
        self.ride_demand = self.generate_ride_demand()

    def generate_prices(self):
        return {
            # Price the EV owner is paid. (Usually lower than what you are charged for consumption)
            'discharge': np.random.rand(self.max_time) * 0.3 + 0.1,  # Initialize price per hour as a random float between 0.1 and 0.4
            # Price charged for consuming electricity
            'charge': np.random.rand(self.max_time) * 0.4 + 0.2  # Initialize price per timestep as a random float between 0.2 and 0.6
        }

    def generate_ride_demand(self):
        return np.random.randint(1, 5, self.max_time)  # Demand between 1 and 5

    def reset(self):
        self.current_soc = 0.5
        self.time = 0
        return self.get_state()

    def get_state(self):
        # return np.array([self.current_soc, self.prices['discharge'][self.time], self.prices['charge'][self.time], self.ride_demand[self.time]])
        return {
            "Current SOC": self.current_soc,
            "Discharge Price": self.prices['discharge'][self.time],
            "Charge Price": self.prices['charge'][self.time],
            "Ride Demand": self.ride_demand[self.time]
        }

    def step(self, action):
        if action == Actions.CHARGE:
            self.current_soc = min(self.current_soc + self.charge_rate, self.max_soc)
            reward = -self.prices['charge'][self.time] * self.charge_rate  # Cost for charging
        elif action == Actions.DISCHARGE:  # Discharge
            self.current_soc = max(self.current_soc - self.discharge_rate, self.min_soc)
            reward = self.prices['discharge'][self.time] * self.discharge_rate  # Revenue from discharging
            # reward should be 0 if not enough energy to discharge
        elif action == Actions.RIDE:
            if self.current_soc >= self.ride_energy:
                self.current_soc -= self.ride_energy
                reward = self.ride_demand[self.time] * 10  # Revenue from providing a ride
            else:
                reward = -10  # Penalty for not having enough charge to provide a ride
        else:
            reward = 0

        self.time += 1
        done = self.time >= (self.max_time - 1)

        next_state = self.get_state()
        return next_state, reward, done

    @classmethod
    def state_to_array(cls, state):
        return np.array([state["Current SOC"], state["Discharge Price"], state["Charge Price"], state["Ride Demand"]])


# Example usage
env = EVEnvironment()
state = env.reset()
print(state)

next_state, reward, done = env.step(Actions.CHARGE)  # Charge
print(next_state, reward, done)
