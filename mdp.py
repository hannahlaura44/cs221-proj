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
                 peak_multiplier=(2.0, 4.0), extreme_peak_multiplier=(20, 40),
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
        self.max_time = max_time

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

        # Base price for providing a ride
        self.base_ride_price = base_ride_price

        # Increment to the base ride price based on demand
        self.ride_price_increment = ride_price_increment

        # Penalty for not having enough charge to provide a ride
        self.penalty_no_charge = penalty_no_charge

        # Peak hours
        self.peak_hours = peak_hours
        self.extreme_peak_hours = extreme_peak_hours

        self.ride_demand = self.generate_ride_demand()
        self.prices = self.generate_prices()


    def generate_prices(self):
        prices = {'charge': np.zeros(self.max_time), 'discharge': np.zeros(self.max_time),
                  'ride': np.zeros(self.max_time)}
        for hour in range(self.max_time):
            if hour in self.extreme_peak_hours:
                prices['charge'][hour] = self.base_high_demand_price
                prices['discharge'][hour] = prices['charge'][hour] * np.random.uniform(*self.extreme_peak_multiplier)
            elif hour in self.peak_hours:
                prices['charge'][hour] = self.base_peak_price
                prices['discharge'][hour] = prices['charge'][hour] * np.random.uniform(*self.peak_multiplier)
            else:
                prices['charge'][hour] = self.base_off_peak_price
                prices['discharge'][hour] = prices['charge'][hour] * 0.8

            prices['ride'][hour] = self.base_ride_price + (self.ride_demand[hour] - 1) * self.ride_price_increment
        return prices

    def generate_ride_demand(self):
        hours = np.arange(self.max_time)
        demand_pattern = 2.5 * (1 + np.sin(2 * np.pi * hours / 24))
        noise = np.random.normal(0, 0.5, self.max_time)
        demand = np.clip(demand_pattern + noise, 1, 5)
        return demand

    def reset(self):
        self.current_soc = self.max_soc * self.initial_soc
        self.time = 0
        return self.get_state()
    def get_state(self):
        return {
            "Current SOC": self.current_soc,
            "Discharge Price": self.prices['discharge'][self.time],
            "Charge Price": self.prices['charge'][self.time],
            "Ride Price": self.prices['ride'][self.time],
            "Ride Demand": self.ride_demand[self.time],
            "Time": self.time
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

    def state_to_array(self, state):
        # State values
        norm_soc = state["Current SOC"] / self.max_soc

        # Prices
        min_charge_price = self.base_off_peak_price
        max_charge_price = self.base_high_demand_price
        norm_charge_price = (state["Charge Price"] - min_charge_price) / (max_charge_price - min_charge_price)

        min_discharge_price = min_charge_price * 0.8
        max_discharge_price = max_charge_price * max(self.extreme_peak_multiplier)
        norm_discharge_price = (state["Discharge Price"] - min_discharge_price) / (
                    max_discharge_price - min_discharge_price)

        # Ride Price
        min_ride_price = self.base_ride_price
        max_ride_price = self.base_ride_price + 4 * self.ride_price_increment
        norm_ride_price = (state["Ride Price"] - min_ride_price) / (max_ride_price - min_ride_price)

        # Ride Demand
        norm_ride_demand = (state["Ride Demand"] - 1) / 4

        # Time
        norm_time = state["Time"] / self.max_time

        return np.array(
            [norm_soc, norm_discharge_price, norm_charge_price, norm_ride_price, norm_ride_demand, norm_time])
