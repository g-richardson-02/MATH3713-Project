import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import timeit
import csv
import xarray as xr
import scipy.spatial as scispa
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import XRSite
from py_wake import NOJ

# =============================================================================
# NSGA-II - PBA (Power and Cost)
# Note: This code must be run through the separate environment, please refer to
# the README file on the GitHub page to configure this. 
# Note: The Power CSV file is required for this code. 
# Note: Please ensure the DEAP library is installed.
# =============================================================================

# Variable Inputs

grid_size = 2000 # Square grid in metres
cell_size = 200 # in metres
num_turbines = 30
v0 = 13  # Initial wind speed (m/s)
a = 0.3267949192 # Axial induction factor (dimensionless)
k = 0.0943695829  # Wake decay constant (dimensionless)
r = 20  # Radius of wake region directly behind rotor (m)
D = 40  # Rotor diameter of turbines (m)
f = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.07, 0.1, 0.13, 0.15, 0.13, 0.07] # wind distribution
wd = np.linspace(0, 360, len(f), endpoint=False)  # wind direction (degrees)
duplicates = False

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
        terminate = False
        wt_list = []
        x, y = random.uniform(0, grid_size), random.uniform(0, grid_size)

        wt_list.append([x,y])

        for i in range(num_turbines):
            attempt = 0
            while True:
                attempt += 1
                new_x, new_y = random.uniform(0, grid_size), random.uniform(0, grid_size)
                distance = []
                for turbine in wt_list:
                    distance.append(scispa.distance.pdist([[new_x, new_y], turbine]))
                if min(distance) > cell_size:
                    wt_list.append(([new_x, new_y]))
                    break
                if attempt > 1000:
                    print('No chance!')
                    terminate = True
                    break
            if terminate:
                nwt = i
                break

        points = wt_list
        return points

def evaluate(individual):
    
    current_solution = np.asarray(individual)
    x = current_solution[:, 0]
    y = current_solution[:, 1]

    sim_res = wf_model(x, y,
                       h=None,  
                       type=0,  
                       wd=wd,   
                       ws=v0,  
                       )
    
    power = float(sim_res.aep().sum())
    
    # Calculate LCOE
    if num_turbines < 10 or duplicates:
        lcoe = 30
    nominal_power = num_turbines * 3
    capex = 2370000             
    opex = 76000  
    I0 = capex * nominal_power
    eff = float(sim_res.aep().sum()) / float(sim_res.aep(with_wake_loss=False).sum())
    lifetime = 25               
    wacc = 0.06                 
    num, den = 0, 0
    for i in range(1, lifetime+1, 1):
        num += (opex*nominal_power) / ((1 + wacc)**i)
        den += (0.84 * eff * 8760 * nominal_power) / ((1+wacc)**i)
    lcoe = (I0 + num) / den

    
    return power, lcoe


# Create the genetic algorithm framework
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Set up the GA using NSGA-II
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_positions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selNSGA2)  # Use NSGA-II selection method
toolbox.register("evaluate", evaluate)

# Set parameters for GA
population = toolbox.population(n=100)
CXPB, MUTPB = 0.7, 0.2
NGEN = 100

# Plot the layout using PyWake Flow Map
def plot_graph(points, power, cost):


    points = np.asarray(points)    
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
    
# Run the Genetic Algorithm with NSGA-II
def main():
    best_individual = None
    best_power = -float('inf')
    best_cost = float('inf')
    store_power = []
    store_cost = []
    start = timeit.default_timer()
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
        store_cost.append(best_cost)
        
    # Plot the best individual's turbine locations and output
    plot_graph(best_individual, best_power, best_cost)
    print(f"Best Total Power Output in 1 year: {best_power} GWh | Best Total Cost: {best_cost:.2f}")
    end = timeit.default_timer()
    print(f"calculation time: {end-start}")
    # Return the best solution
    return best_individual, store_power, store_cost

if __name__ == "__main__":
    best_individual, power, cost = main()
    
    generation = np.arange(0,len(power), 1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(generation, power)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Annual Energy Output (GWh)")
    ax2.plot(generation, cost, color='green')
    ax2.set_ylabel("LCOE (£)")

    plt.show()
    #print(f"Best turbine placement: {best_individual}")



