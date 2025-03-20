import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import XRSite
import xarray as xr
from py_wake import NOJ

import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import timeit

# =============================================================================
# Hill-Climbing Algorithm - PBA
# Note: This code must be run through the separate environment, please refer to
# the README file on the GitHub page to configure this. 
# Note: The Power CSV file is required for this code. 
# =============================================================================

# Variable Inputs
grid_size = 2000 
cell_size = 200 # (m)
num_turbines = 30
v0 = 13  # Initial wind speed (m/s)
a = 0.3267949192 # Axial induction factor (dimensionless)
k = 0.0943695829  # Wake decay constant (dimensionless)
r = 20  # Radius of wake region directly behind rotor (m)
D = 40  # Rotor diameter of turbines (m)
f = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.07, 0.1, 0.13, 0.15, 0.13, 0.07] # wind distribution
wd = np.linspace(0, 360, len(f), endpoint=False) # wind direction (degrees)

# Set up for the Site variable
site = XRSite(ds=xr.Dataset(data_vars={'WS': v0,  
                                             'P': ('wd', f),
                                             'TI': 0,},
                                  coords={'wd': wd}))
# Reading the power file
file = open("power_thrust_mos.csv") #Add in file path here if not in same location as this Python file
csvreader = csv.reader(file)
header = next(csvreader)

u, power, ct = [], [], []

for row in csvreader:
    u.append(int(row[0]))
    ct.append(float(row[1]))
    power.append(float(row[2]))
file.close()
    
windTurbines =WindTurbine(name='Vesta112_3MW',
                           diameter=40,
                           hub_height=60,
                           powerCtFunction=PowerCtTabular(u, power, 'kW', ct))
wf_model = NOJ(site, windTurbines)


# Generate random turbine positions
def create_positions():
    # Create a list of all possible grid positions (x, y)
    grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # Randomly select positions for the turbines
    return random.sample(grid_positions, num_turbines)

# Verify the positions are valid
def overlap_check(positions):
    unique_positions = list(np.unique(positions))  # Remove duplicates
    if len(unique_positions) < num_turbines:
        # Refill the layout with new positions to match the correct number of turbines
        grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        remaining_positions = list(np.unique(grid_positions) - np.unique(unique_positions))
        new_positions = random.sample(remaining_positions, num_turbines - len(unique_positions))
        positions[:] = unique_positions + new_positions
    return positions

# Works differently than in EBA to ensure PyWake does not position turbines too close together.
def generate_neighbours(solution, max_distance=200):
    neighbours = []

    # Calculate the distance between two points
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    for i in range(num_turbines):
        # Create a new neighbour by copying the current solution
        neighbour = solution[:]
        
        # Flag to track whether a valid position is found
        valid_position_found = False

        while not valid_position_found:
            # Randomly move the turbine at index i within the max_distance limit
            x, y = neighbour[i]
            
            # Generate a new x, y position within the max_distance limit
            new_x = x + random.choice([-max_distance, 0, max_distance])
            new_y = y + random.choice([-max_distance, 0, max_distance])

            # Ensure the new position is within the grid bounds
            new_x = max(0, min(grid_size - 1, new_x))
            new_y = max(0, min(grid_size - 1, new_y))

            # Assign the new position to the turbine at index i
            neighbour[i] = (new_x, new_y)

            # Check for overlaps with other turbines (all turbines in the current solution)
            is_valid_position = True
            for j in range(num_turbines):
                if i != j: 
                    if distance(neighbour[i], neighbour[j]) < 200:  # If turbines are too close
                        is_valid_position = False
                        break

            # If the new position is valid, add it to the neighbours list
            if is_valid_position:
                valid_position_found = True
            else:
                # If not valid, retry 
                continue

        neighbours.append(neighbour)
    return neighbours

# Hill climbing algorithm
def hill_climbing(num_iterations):
    # Initialize a random solution
    current_solution = create_positions()
    current_solution = overlap_check(current_solution)
    current_solution = np.asarray(current_solution)
    x = current_solution[:, 0]
    y = current_solution[:, 1]

    # Create wind farm model (values taken from windTurbines variable)
    sim_res = wf_model(x, y,  
                       h=None,  
                       type=0,  
                       wd=wd,
                       ws=v0,  
                       )
    # Calculate AEP from the wind farm model
    AEP = float(sim_res.aep().sum())
    current_power = AEP

    best_solution = current_solution
    best_power = current_power
    store_power = []

    iteration = 0
    while iteration < num_iterations:
        print(iteration)
        iteration += 1
        neighbours = generate_neighbours(current_solution)
        best_neighbour = None
        best_neighbour_power = -float('inf')

        # Evaluate neighbors
        for neighbour in neighbours:
            xn = neighbour[:, 0]
            yn = neighbour[:, 1]
            n_sim_res = wf_model(xn, yn,  
                               h=None,  
                               type=0,  
                               wd=wd,  
                               ws=v0,  
                               )
            npower = float(n_sim_res.aep().sum())
            best_neighbour_power = -float('inf')
            if npower > best_neighbour_power:
                best_neighbour = neighbour
                best_neighbour_power = npower
            
        # If a better neighbor was found, move to it
        if best_neighbour_power > current_power:
            current_solution = best_neighbour
            current_power = best_neighbour_power
            
            if current_power > best_power:
                        best_solution = current_solution
                        best_power = current_power
        store_power.append(best_power)       

    return best_solution, best_power, store_power

# Plot the layout using PyWake Flow Map
def plot_graph(points):

    x = points[:, 0]
    y = points[:, 1]
    wfm = wf_model(x, y,  
                      h=None,  
                      type=0,  
                      wd=wd,  
                      ws=v0,  
                           )  
    flow_map = wfm.flow_map(grid=None, wd=0, ws=v0)
    flow_map.plot_wake_map()
    plt.grid()
    plt.xlabel('Distance (m)')
    plt.ylabel('Distance (m)')
    plt.show()
    
# Main function to run the hill climbing algorithm    
def main():
    start = timeit.default_timer()
    best_individual, best_power, store_power = hill_climbing(2000)
    end = timeit.default_timer()
    print(f"Calculation time: {end-start}")
    plot_graph(best_individual)
    
    # Plotting the power output against number of iterations
    iteration=np.arange(0,len(store_power),1)
    plt.plot(iteration, store_power)
    plt.xlabel("Iteration")
    plt.ylabel("Annual Energy Produced (GWh)")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.show()
    

    print(f"Best Total Power Output in 1 year: {best_power:.2f} GWh")
    return best_individual

if __name__ == "__main__":
    best_individual = main()



    