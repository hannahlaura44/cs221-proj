import numpy as np
import enum

np.random.seed(42)


class Actions(enum.Enum):
    CHARGE = 0
    DISCHARGE = 1
    RIDE = 2


class EVEnvironment:
    def __init__(self, initial_soc=0.5, max_soc=75.0, min_soc=0.0, charge_rate=18.75, discharge_rate=7.5,
                 ride_energy=7.5, ride_duration=0.5,
                 base_off_peak_price=0.24, base_peak_price=0.48, base_high_demand_price=1.00,
                 peak_multiplier=(2.0, 4.0), extreme_peak_multiplier=(20, 40), off_peak_variance=0.02,
                 peak_variance=0.05, high_demand_variance=0.1,
                 base_ride_price=20, ride_price_increment=5, penalty_no_charge=-10, peak_hours=range(32, 40),
                 extreme_peak_hours=range(39, 42), max_time=48):

        # Initial state of charge (SOC) of the battery in kWh, set to 50% of max SOC
        # Source: https://evcompare.io/cars/tesla/tesla_model_y/
        self.initial_soc = initial_soc

        # Maximum state of charge (SOC) of the battery in kWh (Tesla Model Y: 75 kWh)
        self.max_soc = max_soc

        self.current_soc = self.initial_soc * self.max_soc

        # Minimum state of charge (SOC) of the battery in kWh
        self.min_soc = min_soc

        # Rate at which the battery charges per 30 minutes (18.75 kWh per 30 minutes, assuming Supercharger V3 peak rate)
        # Source: https://www.tesla.com/supercharger
        self.charge_rate = charge_rate

        # Rate at which the battery discharges per 30 minutes for V2G (7.5 kW)
        # Source: https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/infrastructure/vehicle-grid-integration
        self.discharge_rate = discharge_rate

        # Energy consumed per ride in kWh
        self.ride_energy = ride_energy

        # Duration of the ride as a percentage of an hour (0.5 represents 30 minutes)
        self.ride_duration = ride_duration

        self.time = 0
        self.max_time = max_time  # 24 hours with 30-minute intervals
        self.ride_demand = self.generate_ride_demand()

        # Price parameters
        # Base off-peak price per kWh during off-peak hours
        self.base_off_peak_price = base_off_peak_price
        # Base peak price per kWh during peak hours
        self.base_peak_price = base_peak_price
        # Base high-demand price per kWh during extreme peak hours
        self.base_high_demand_price = base_high_demand_price
        # Multiplier range for discharge price during exteme peak hours
        self.extreme_peak_multiplier = extreme_peak_multiplier
        # Multiplier range for discharge price during peak hours
        self.peak_multiplier = peak_multiplier
        # Variance for off-peak prices
        self.off_peak_variance = off_peak_variance
        # Variance for peak prices
        self.peak_variance = peak_variance
        # Variance for high-demand prices
        self.high_demand_variance = high_demand_variance

        # Ride pricing parameters
        # Base price for providing a ride
        self.base_ride_price = base_ride_price
        # Increment to the base ride price based on demand
        self.ride_price_increment = ride_price_increment

        # Penalty for not having enough charge to provide a ride
        self.penalty_no_charge = penalty_no_charge

        # Peak and off-peak hours
        self.peak_hours = peak_hours
        self.extreme_peak_hours = extreme_peak_hours

        self.prices = self.generate_prices()

    def generate_prices(self):
        prices = {'charge': np.zeros(self.max_time), 'discharge': np.zeros(self.max_time),
                  'ride': np.zeros(self.max_time)}

        for hour in range(self.max_time):
            if hour in self.extreme_peak_hours:  # Extreme peak hours
                prices['charge'][hour] = self.base_high_demand_price + np.random.normal(0, self.high_demand_variance)
                prices['discharge'][hour] = prices['charge'][hour] * np.random.uniform(*self.extreme_peak_multiplier)
            elif hour in self.peak_hours:  # Peak hours
                prices['charge'][hour] = self.base_peak_price + np.random.normal(0, self.peak_variance)
                prices['discharge'][hour] = prices['charge'][hour] * np.random.uniform(*self.peak_multiplier)
            else:  # off peak hours
                prices['charge'][hour] = self.base_off_peak_price + np.random.normal(0, self.off_peak_variance)
                prices['discharge'][hour] = prices['charge'][hour] * np.random.uniform(0.8,
                                                                                       1.0)  # Off peak usually 80 % of on peak

            prices['ride'][hour] = self.base_ride_price + (self.ride_demand[hour] - 1) * self.ride_price_increment

        return prices

    def generate_ride_demand(self):
        return np.random.randint(1, 5, self.max_time)  # Demand between 1 and 5

    def reset(self):
        self.current_soc = self.max_soc * self.initial_soc  # Reset to 50% of maximum SOC
        self.time = 0
        return self.get_state()

    def get_state(self):
        return {
            "Current SOC": self.current_soc,
            "Discharge Price": self.prices['discharge'][self.time],
            "Charge Price": self.prices['charge'][self.time],
            "Ride Price": self.prices['ride'][self.time],
            "Ride Demand": self.ride_demand[self.time],
            "Time": self.time  # Added by Marcelo
        }

    def step(self, action):
        if self.time >= self.max_time:
            raise IndexError("Time index exceeded max_time")

        if action == Actions.CHARGE:
            self.current_soc = min(self.current_soc + self.charge_rate, self.max_soc)
            reward = -self.prices['charge'][self.time] * self.charge_rate
        elif action == Actions.DISCHARGE:
            self.current_soc = max(self.current_soc - self.discharge_rate, self.min_soc)
            reward = self.prices['discharge'][self.time] * self.discharge_rate
        elif action == Actions.RIDE:
            if self.current_soc >= self.ride_energy:
                self.current_soc -= self.ride_energy * self.ride_duration
                reward = self.prices['ride'][self.time]
            else:
                reward = self.penalty_no_charge
        else:
            reward = 0

        self.time += 1
        done = self.time >= self.max_time

        if done:
            self.time = self.max_time - 1

        next_state = self.get_state()
        return next_state, reward, done

    @classmethod
    def state_to_array(cls, state):
        return np.array([state["Current SOC"], state["Discharge Price"], state["Charge Price"], state["Ride Price"],
                         state["Ride Demand"], state["Time"]])  # Time added by Marcelo.


# Example usage
env = EVEnvironment()
state = env.reset()
print(state)

# Simulating a day with decision-making based on SOC and price comparison
total_reward = 0
for step in range(1, env.max_time):  # Adjusted range to 47 to avoid out-of-bounds error
    if env.current_soc < 10:
        action = Actions.CHARGE
    elif env.prices['discharge'][env.time] > env.prices['ride'][env.time]:
        action = Actions.DISCHARGE
    else:
        action = Actions.RIDE

    next_state, reward, done = env.step(action)
    print(f"Step: {step}, Action: {action.name}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    total_reward += reward
    if done:
        break
print(f"Total reward: {total_reward}")
