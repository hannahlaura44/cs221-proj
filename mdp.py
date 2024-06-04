import numpy as np
import enum
import matplotlib.pyplot as plt

# np.random.seed(42)

class Actions(enum.Enum):
    CHARGE = 0
    DISCHARGE = 1
    RIDE = 2

class EVEnvironment:
    def __init__(self, initial_soc=0.5, max_soc=75.0, min_soc=0.0, charge_rate=125, discharge_rate=5, ride_energy=7.5, ride_duration=0.5,
                 base_off_peak_price=0.24, base_peak_price=0.48, base_high_demand_price=1.00, peak_multiplier=(2.0, 4.0), extreme_peak_multiplier=(5,20), off_peak_variance=0.02, peak_variance=0.05, high_demand_variance=0.1,
                 base_ride_price=10, ride_price_increment=5, penalty_no_charge=-10, peak_hours=range(32, 40), extreme_peak_hours=range(39,42), max_time=48):
        
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

        # Price parameters
        # Base off-peak price ($ per kWh) during off-peak hours
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
        
        self.revenue = self.generate_revenue()

    def generate_revenue(self):
        prices = {'charge': np.zeros(self.max_time), 'discharge': np.zeros(self.max_time)}

        for t in range(self.max_time):
            if t in self.extreme_peak_hours: # Extreme peak hours
                prices['charge'][t] = self.base_high_demand_price + np.random.normal(0, self.high_demand_variance)
                prices['discharge'][t] = prices['charge'][t] * np.random.uniform(*self.extreme_peak_multiplier)
            elif t in self.peak_hours:  # Peak hours
                prices['charge'][t] = self.base_peak_price + np.random.normal(0, self.peak_variance)
                prices['discharge'][t] = prices['charge'][t] * np.random.uniform(*self.peak_multiplier)
            else:  # off peak hours
                prices['charge'][t] = self.base_off_peak_price + np.random.normal(0, self.off_peak_variance)
                prices['discharge'][t] = prices['charge'][t] * np.random.uniform(0.8, 1.0) #Off peak usually 80 % of on peak

        revenue = {'ride': self.generate_ride_prices()}
        revenue['charge'] = - prices['charge'] * self.charge_rate
        revenue['discharge'] = prices['discharge'] * self.discharge_rate

        return revenue
    
    def plot_revenue(self):
        # Number of 30-minute intervals
        intervals = len(self.revenue['ride'])

        # Generate x values for hours
        x_values = np.arange(intervals) / 2

        # Plotting the revenue arrays
        plt.plot(x_values, self.revenue['charge'], label='Charge Revenue')
        plt.plot(x_values, self.revenue['discharge'], label='Discharge Revenue')
        plt.plot(x_values, self.revenue['ride'], label='Ride Revenue')

        # Adding titles and labels
        plt.title('Revenue Over Time')
        plt.xlabel('Time Interval')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_ride_prices(self):
        # Initialize price array
        prices = np.zeros(self.max_time)
        
        # Constants for different time periods
        daytime_start, daytime_end = 6, 22  # 6 AM to 10 PM
        nighttime_base_price = 1
        daytime_base_price = 5
        peak_price = 15
        
        for thirty_min_interval in range(self.max_time):
            hour_of_day = thirty_min_interval / 2  # To get the hour of the day
            
            if daytime_start <= hour_of_day < daytime_end:
                # Daytime hours with potential peak demand
                if (7 <= hour_of_day < 9) or (17 <= hour_of_day < 19):  # Peak hours within daytime
                    prices[thirty_min_interval] = np.random.poisson(peak_price)
                else:
                    # Daytime hours with elevated base demand
                    prices[thirty_min_interval] = np.random.poisson(daytime_base_price + np.random.randint(1, 4))
            else:
                # Nighttime hours with very low demand
                prices[thirty_min_interval] = np.random.poisson(nighttime_base_price)
        
        return prices

    def reset(self):
        self.current_soc = self.max_soc * self.initial_soc  # Reset to 50% of maximum SOC
        self.time = 0
        self.revenue = self.generate_revenue() # generate new electricity and ride prices (with some variation)
        return self.get_state()

    def get_state(self):
        return {
            "Current SOC": self.current_soc,
            "Discharge Price": self.revenue['discharge'][self.time], # potential discharge revenue from previous timestep
            "Charge Price": self.revenue['charge'][self.time],
            "Ride Price": self.revenue['ride'][self.time],
            "Time": self.time / 2
        }

    def step(self, action):
        self.time += 1
        done = self.time >= self.max_time

        if done:
            self.time = self.max_time - 1

        if action == Actions.CHARGE:
            self.current_soc = min(self.current_soc + self.charge_rate, self.max_soc)
            reward = self.revenue['charge'][self.time]  # Cost for charging
        elif action == Actions.DISCHARGE:
            # Calculate the possible discharge amount
            possible_discharge = self.current_soc - self.min_soc
            actual_discharge = min(self.discharge_rate, possible_discharge)
            self.current_soc -= actual_discharge
            self.current_soc = max(self.current_soc - self.discharge_rate, self.min_soc) #TODO bug
            # Calculate reward based on the fraction of the discharge rate
            reward = self.revenue['discharge'][self.time] * (actual_discharge / self.discharge_rate)      
        elif action == Actions.RIDE:
            if self.current_soc >= self.ride_energy:
                self.current_soc -= self.ride_energy * self.ride_duration
                reward = self.revenue['ride'][self.time] # Revenue from providing a ride
            else:
                reward = self.penalty_no_charge # Penalty for not having enough charge to provide a ride
        else:
            reward = 0

        next_state = self.get_state()
        return next_state, reward, done

    # @classmethod
    def state_to_array(self, state):
        # normalize
        norm_soc = state["Current SOC"] / self.max_soc
        norm_discharge = state["Discharge Price"] / np.max(self.revenue['discharge'])
        norm_charge = state["Charge Price"] / np.max(self.revenue['charge'])
        norm_ride = state["Ride Price"] / np.max(self.revenue['ride'])
        norm_time = state["Time"] / self.max_time

        return np.array([norm_soc, norm_discharge, norm_charge, norm_ride, norm_time])
