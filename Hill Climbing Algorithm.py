import random
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit


# =============================================================================
# Hill-Climbing Algorithm - EBA
# Note: This code can be run through the standard Python terminal and does not 
# require the separate environment.
# =============================================================================

# Variable Inputs
grid_size = 10
cell_size = 200 # in metres
num_turbines = 30


# Generate random turbine positions
def create_positions():
    # Create a list of all possible grid positions (x, y)
    grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # Randomly select positions for the turbines
    return random.sample(grid_positions, num_turbines)

# Verify the positions are valid
def overlap_check(positions):
    unique_positions = list(set(positions))  # Remove duplicates
    if len(unique_positions) < num_turbines:
        # Refill the layout with new positions to match the correct number of turbines
        grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        remaining_positions = list(set(grid_positions) - set(unique_positions))
        new_positions = random.sample(remaining_positions, num_turbines - len(unique_positions))
        positions[:] = unique_positions + new_positions
    return positions

# Power calculation function considering wake effects
def calc_power(positions):
    # Values given from Mosetti et al.
    cell_size = 200
    rotor_radius = 20 # Radius of wake region directly behind rotor (m)
    wake_decay = 0.0943695829 # Wake decay constant (dimensionless)
    a = 0.3267949192 # Axial induction factor (dimensionless)
    thrust_coeff = 0.88
    wind_velocity = 12 # Initial wind speed (m/s)
    b = (1 - a) / (1 - 2 * a)
    r = rotor_radius * math.sqrt(b) 

    total_power = 0
    turbines_power = [1 for _ in range(num_turbines)]  # initialise power of each turbine

    for i, (x1, y1) in enumerate(positions):
        for j, (x2, y2) in enumerate(positions):
            if i != j:
                
                # Jensen Wake Model Equation
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * cell_size
                
                c = 1 - math.sqrt(1 - thrust_coeff)
                d = (1 + (wake_decay * dist) / r) ** 2
                wind_velocity_due_to_wake = wind_velocity * (1 - c / d)
                
                # Manikowski et al. power equation
                if 2.3 <= wind_velocity_due_to_wake <= 12.8:
                    turbines_power[i] = 0.3 * wind_velocity_due_to_wake ** 3
                elif 12.8 < wind_velocity_due_to_wake <= 18:
                    turbines_power[i] = 630
                else:
                    turbines_power[i] = 0

    turbines_power *= 8760 # Annual energy output in GWh
    total_power = sum(turbines_power)
    return total_power

def evaluate(positions):
    power = calc_power(positions)
    return power

def generate_neighbours(solution, max_distance=1):
    neighbours = []

    for i in range(num_turbines):
        # Create a new neighbour by copying the current solution
        neighbour = solution[:]

        # Randomly move the turbine at index i within the max_distance limit
        x, y = neighbour[i]
        
        # Generate a new x, y position within the max_distance limit
        new_x = x + random.randint(-max_distance, max_distance)
        new_y = y + random.randint(-max_distance, max_distance)

        # Ensure the new position is within the grid bounds
        new_x = max(0, min(grid_size - 1, new_x))
        new_y = max(0, min(grid_size - 1, new_y))

        # Assign the new position to the turbine at index i
        neighbour[i] = (new_x, new_y)
        
        # Check for overlaps with other turbines
        neighbour = overlap_check(neighbour)

        # Add the neighbour to the list
        neighbours.append(neighbour)

    return neighbours

# Hill climbing algorithm
def hill_climbing(num_iterations):
    # Initialise a random solution
    current_solution = create_positions()
    current_solution = overlap_check(current_solution)
    print(current_solution)
    
    current_power = evaluate(current_solution)

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
            power = evaluate(neighbour)
            if power > best_neighbour_power:
                best_neighbour = neighbour
                best_neighbour_power = power


        # If a better neighbor was found, move to it
        if best_neighbour and best_neighbour_power > current_power:
            current_solution = best_neighbour
            current_power = best_neighbour_power

            # Track the best solution
            if current_power > best_power:
                best_solution = current_solution
                best_power = current_power
        #plot_turbines(current_solution, current_power) 
        store_power.append(best_power)
        

    return best_solution, best_power, store_power


def plot_turbines(positions, total_power):
    # Extract x and y coordinates of turbine positions
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Create a grid plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='red', label='Turbines', marker='o')
    plt.xlim(0, grid_size-1)
    plt.ylim(0, grid_size-1)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()


# Main function to run the hill climbing algorithm
def main():
    best_individual, best_power, store_power = hill_climbing(500)
    plot_turbines(best_individual, best_power)
    
    # Plotting the power output against number of iterations
    iteration=np.arange(0,500,1)
    store_power = [x / 1000000 for x in store_power]
    plt.plot(iteration, store_power)
    plt.xlabel("Iteration")
    plt.ylabel("Annual Energy Produced (GWh)")
    plt.yticks([115,120,125,130,135])

    print(f"Best Total Power Output in 1 year: {best_power:.2f} kWh")
    return best_individual

if __name__ == "__main__":
    start = timeit.default_timer()
    best_individual = main()
    print(f"Best turbine placement: {best_individual}")
    end = timeit.default_timer()
    print(f"Calculation time: {end-start}")
    
    