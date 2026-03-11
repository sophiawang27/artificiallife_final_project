from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots, evolve
import numpy as np
import copy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set the random seed for reproducibility
    np.random.seed(config["seed"])
    # Randomly sample robots
    # NOTE: the number of robots should match the number of parallel simulations allocated in the simulator config
    robots = load_robots(num_robots=config["simulator"]["n_sims"]) 

    # Extract the number of masses and springs in each robot
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]
    # Find the largest number of masses and springs in any robot
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    # Save the maximum number of masses and springs to the simulator config
    # NOTE: this is essential to ensure the simulator allocates the correct amount of memory for the simulation
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    # Initialize the simulator
    simulator = Simulator(sim_config=config["simulator"], taichi_config=config["taichi"],seed=config["seed"], needs_grad=True)

    # Extract the masses and springs from each robot
    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]


    # Initialize the simulator state with the unique geometries of the robots
    simulator.initialize(masses, springs)

    # Train the robots to perform locomotion
    # The number of learning steps is specified in the configuration
    fitness_history = simulator.train() # numpy array of shape (n_robots, n_learning_steps)
    # Save the fitness history to a file
    np.save("fitness_history.npy", fitness_history)

    # Select the final fitness of each robot after training
    fitness = fitness_history[:, -1]

    control_params = simulator.get_control_params(range(len(robots)))
    for i in range(len(robots)):
        robots[i]["control_params"] = control_params[i]

    #number of evolutions 
    NGEN = 30
    
    # prev_robots = [robot.copy() for robot in robots]
    # final_robots = [robot.copy() for robot in robots]
    prev_robots = [copy.deepcopy(robot) for robot in robots]
    final_robots = [copy.deepcopy(robot) for robot in robots]
    prev_final_fitness = fitness.copy()


    fitness_over_time = []
    best_fitness_over_time = []

    for gen in range(NGEN):
        new_robots = []
        #evolving the robot
        new_robots = [evolve(robot, max_masses = max_num_masses, max_springs = max_num_springs) for robot in prev_robots]
        
        simulator = Simulator(sim_config=config["simulator"], taichi_config=config["taichi"],seed=config["seed"], needs_grad=True)
        simulator.initialize([robot["masses"] for robot in new_robots], [robot["springs"] for robot in new_robots])
        # collect inherited control params
        sim_indices = []
        control_params_list = []

        for i, robot in enumerate(new_robots):
            if "control_params" in robot:
                sim_indices.append(i)
                control_params_list.append(robot["control_params"])

        if len(sim_indices) > 0:
            simulator.set_control_params(sim_indices, control_params_list)

        new_fitness_history = simulator.train() # numpy array of shape (n_robots, n_learning_steps)
        np.save(f"fitness_history{gen}.npy", new_fitness_history)
        
        #select parent/child to be previous for next generation
        new_final_fitness = new_fitness_history[:, -1]

        new_control_params = simulator.get_control_params(range(len(new_robots)))

        for i in range(len(new_robots)):
            new_robots[i]["control_params"] = new_control_params[i]

        next_gen_robots = []
        next_gen_fitness = []

        for i in range(len(new_robots)):
            if new_final_fitness[i] > prev_final_fitness[i]:
                chosen = copy.deepcopy(new_robots[i])
                chosen_fitness = new_final_fitness[i]
            else:
                chosen = copy.deepcopy(prev_robots[i])
                chosen_fitness = prev_final_fitness[i]
            next_gen_robots.append(chosen)
            next_gen_fitness.append(chosen_fitness)

        #prepare for next generation
        prev_robots = [copy.deepcopy(r) for r in next_gen_robots]
        prev_final_fitness = np.array(next_gen_fitness)

        fitness_over_time.append(prev_final_fitness)
        best_fitness_over_time.append(np.max(next_gen_fitness))
        # next generation parents
        #prev_robots = new_robots

        print(f"Generation {gen}: Best Fitness = {best_fitness_over_time[-1]:.4f}")
        for i, robot in enumerate(prev_robots):
            robot_copy = robot.copy()
            robot_copy["max_n_masses"] = max_num_masses
            robot_copy["max_n_springs"] = max_num_springs
            np.save(f"gen_{gen}_robot_{i}.npy", robot_copy)


    import matplotlib.pyplot as plt
    plt.plot(best_fitness_over_time)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.savefig('evolution_prog.png')

    # # Sort the robots by fitness
    # ranking = np.argsort(prev_final_fitness)[::-1]
    # ranked_robots = [final_robots[i] for i in ranking]
    # # Select the top 3 performers
    # top_3_idxs = ranking[:3]
    # top_3_robots = [final_robots[i] for i in top_3_idxs]
    # # Extract the control parameters of the top 3 performers
    # top_3_control_params = simulator.get_control_params(top_3_idxs)
    # # Save each of the top 3 robots and their control parameters to a file
    # for i in range(3):
    #     robot = top_3_robots[i]
    #     control_params = top_3_control_params[i]
    #     robot["control_params"] = control_params
    #     # Save the max dimensions used during training so visualizer can recreate the same memory allocation setup in the simulator
    #     robot["max_n_masses"] = max_num_masses
    #     robot["max_n_springs"] = max_num_springs
    #     np.save(f"robot_{i}.npy", robot)



# from simulator import Simulator
# from utils import load_config
# from argparse import ArgumentParser
# from robot import load_robots
# import numpy as np

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config", type=str, default="config.yaml")
#     args = parser.parse_args()

#     # Load the configuration
#     config = load_config(args.config)

#     # Set the random seed for reproducibility
#     np.random.seed(config["seed"])
#     # Randomly sample robots
#     # NOTE: the number of robots should match the number of parallel simulations allocated in the simulator config
#     robots = load_robots(num_robots=config["simulator"]["n_sims"])

#     # Extract the number of masses and springs in each robot
#     num_masses = [robot["n_masses"] for robot in robots]
#     num_springs = [robot["n_springs"] for robot in robots]
#     # Find the largest number of masses and springs in any robot
#     max_num_masses = max(num_masses)
#     max_num_springs = max(num_springs)
#     # Save the maximum number of masses and springs to the simulator config
#     # NOTE: this is essential to ensure the simulator allocates the correct amount of memory for the simulation
#     config["simulator"]["n_masses"] = max_num_masses
#     config["simulator"]["n_springs"] = max_num_springs

#     # Initialize the simulator
#     simulator = Simulator(sim_config=config["simulator"], taichi_config=config["taichi"],seed=config["seed"], needs_grad=True)

#     # Extract the masses and springs from each robot
#     masses = [robot["masses"] for robot in robots]
#     springs = [robot["springs"] for robot in robots]
#     # Initialize the simulator state with the unique geometries of the robots
#     simulator.initialize(masses, springs)

#     # Train the robots to perform locomotion
#     # The number of learning steps is specified in the configuration
#     fitness_history = simulator.train() # numpy array of shape (n_robots, n_learning_steps)
#     # Save the fitness history to a file
#     np.save("fitness_history.npy", fitness_history)

#     # Select the final fitness of each robot after training
#     fitness = fitness_history[:, -1]

#     NGEN = 10
#     #initialize first generation copy

#     for gen in range(NGEN):
#         #initialize the fitness history
#         #empty next gen
#         #evolve prev gen
#         #train new gen (same parameters)
#         #compare final fitness
#         #select for parent/child
#         #prepare next iteration





#     # # Sort the robots by fitness
#     # ranking = np.argsort(fitness)[::-1]
#     # ranked_robots = [robots[i] for i in ranking]
#     # # Select the top 3 performers
#     # top_3_idxs = ranking[:3]
#     # top_3_robots = [robots[i] for i in top_3_idxs]
#     # # Extract the control parameters of the top 3 performers
#     # top_3_control_params = simulator.get_control_params(top_3_idxs)
#     # # Save each of the top 3 robots and their control parameters to a file
#     # for i in range(3):
#     #     robot = top_3_robots[i]
#     #     control_params = top_3_control_params[i]
#     #     robot["control_params"] = control_params
#     #     # Save the max dimensions used during training so visualizer can recreate the same memory allocation setup in the simulator
#     #     robot["max_n_masses"] = max_num_masses
#     #     robot["max_n_springs"] = max_num_springs
#     #     np.save(f"robot_{i}.npy", robot)