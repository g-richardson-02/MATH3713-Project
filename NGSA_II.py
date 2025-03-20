
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from deap import base, creator, tools
import timeit

# =============================================================================
# NSGA-II - EBA
# Note: This code can be run through the standard Python terminal and does not 
# require the separate environment. 
# Note: Please ensure the DEAP library is installed.
# =============================================================================

# Variable Inputs
grid_size = 10 
num_turbines = 30

# Power calculation function considering wake effects
def calc_power(turbine_positions):
    cell_size = 200 #metres
    rotor_radius = 20 # Radius of wake region directly behind rotor (m)
    wake_decay = 0.0943695829 # Wake decay constant (dimensionless)
    a = 0.3267949192 # Axial induction factor (dimensionless)
    thrust_coeff = 0.88
    wind_velocity = 12 # Initial wind speed (m/s)
    
    b = (1-a)/(1-2*a)
    r = rotor_radius * math.sqrt(b)

    total_power = 0
    turbines_power = [1 for _ in range(num_turbines)] # initialise power of each turbine
    
    for i, (x1, y1) in enumerate(turbine_positions):
        for j, (x2, y2) in enumerate(turbine_positions):
            if i != j:
                # Jensen Wake Model Equation
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * cell_size
                c = 1 - math.sqrt(1-thrust_coeff)
                d = (1 + (wake_decay * dist)/r)**2
                wind_velocity_due_to_wake = wind_velocity * (1 - c/d)
                
                # Manikowski et al. power equation
                if 2.3 <= wind_velocity_due_to_wake <= 12.8:
                    turbines_power[i] = 0.3 * wind_velocity_due_to_wake**3
                elif 12.8 < wind_velocity_due_to_wake <=18:
                    turbines_power[i] = 630
                else:
                    turbines_power[i] = 0
                    
    turbines_power *= 8760 # Annual energy output in GWh
    total_power = sum(turbines_power)
    return total_power

# Cost calculation function (to minimize cost)
def calc_cost(turbine_positions):
    cost = num_turbines * (2/3 +1/3 * math.exp(-0.0017 * num_turbines**2))
    return cost

# Create the genetic algorithm framework
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # First maximise power, then minimise cost
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Generate random turbine positions
def create_positions():
    # Create a list of all possible grid positions (x, y)
    grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # Randomly select unique positions for the turbines without replacement
    return random.sample(grid_positions, num_turbines)

# Verify the positions are valid
def overlap_check(positions):
    # Ensure there are no duplicate positions
    unique_positions = list(set(positions))  # Remove duplicates
    if len(unique_positions) < num_turbines:
        # Refill the individual with new positions to match the correct number of turbines
        grid_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        remaining_positions = list(set(grid_positions) - set(unique_positions))
        new_positions = random.sample(remaining_positions, num_turbines - len(unique_positions))
        positions[:] = unique_positions + new_positions
    return positions

# Evaluation function for NSGA-II (multi-objective)
def evaluate(individual):
    power = calc_power(individual)
    cost = calc_cost(individual)
    return power, cost

# Set up the GA using NSGA-II
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_positions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selNSGA2)  
toolbox.register("evaluate", evaluate)

# Set parameters for GA
population = toolbox.population(n=100)
CXPB, MUTPB = 0.7, 0.2
NGEN = 100

# Plotting function to visualize turbine locations
def plot_turbines(turbine_positions, generation, total_power, total_cost):
    # Extract x and y coordinates of turbine positions
    x_coords = [pos[0] for pos in turbine_positions]
    y_coords = [pos[1] for pos in turbine_positions]
    
    # Create a grid plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='red', label='Turbines', marker='o')
    plt.xlim(0, grid_size-1)
    plt.ylim(0, grid_size-1)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

# Run the Genetic Algorithm with NSGA-II
def nsga_ii():
    best_individual = None
    best_power = -float('inf')
    best_cost = float('inf')
    store_power = []
    store_cost = []
    
    # Evaluate the individuals in the population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    for gen in range(NGEN):
        print(f"Generation {gen} calculated")
        
        # Select the next generation using NSGA-II
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Verify validity of offspring positions
        for ind in offspring:
            overlap_check(ind)
        
        # Reevaluate individuals with invalid fitness
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_individuals:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Replace the old population by the new one
        population[:] = offspring
        
        # Track the best individual in this generation
        current_best = tools.selBest(population, 1)[0]
        current_power = current_best.fitness.values[0]
        current_cost = current_best.fitness.values[1]
        
        if current_power > best_power:
            best_individual = current_best
            best_power = current_power
            best_cost = current_cost
   
        store_power.append(best_power)
        cost_ratio = best_cost / best_power
        store_cost.append(cost_ratio)
        
    # Plot the best individual's turbine locations and output
    plot_turbines(best_individual, NGEN-1, best_power, best_cost)
    terrawatt_power = best_power / 1000000
    print(f"Best Total Power Output in 1 year: {terrawatt_power:.2f} GWh | Best Total Cost: {best_cost:.2f}")
    
    # Return the best solution
    return best_individual, store_power, store_cost

if __name__ == "__main__":
    start = timeit.default_timer()
    best_individual, power, cost = nsga_ii()
    print(f"Best turbine placement: {best_individual}")
    end = timeit.default_timer()
    
    generation = np.arange(0,len(power), 1)
    power = [x / 1000000 for x in power]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(generation, power)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Annual Energy Output (GWh)")
    ax2.plot(generation, cost, color='green')
    ax2.set_ylabel("Cost to Power ratio")
    plt.show()

    print(f"calculation time: {end-start}")
    
    