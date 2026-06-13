from functools import partial

import narmedbandit
import algorithms
import methods
import visualization

# parameters
TRIALS = 2000
ACTION_SPACE_SIZE = 10
STEPS = 1000

if __name__ == "__main__":
    vizualizer = visualization.Visualizer()
    vizualizer.start(TRIALS, ACTION_SPACE_SIZE, STEPS)
    run_trials_prefilled = partial(narmedbandit.run_trials, TRIALS, ACTION_SPACE_SIZE, STEPS, methods.sample_average, 0)

    vizualizer.begin_algorithm("greedy")
    run_trials_prefilled(algorithms.greedy, observer=vizualizer.record)
    
    vizualizer.begin_algorithm("e-greedy (ε=0.01)")
    run_trials_prefilled(partial(algorithms.e_greedy, 0.01), observer=vizualizer.record)
    
    vizualizer.begin_algorithm("e-greedy (ε=0.1)")
    run_trials_prefilled(partial(algorithms.e_greedy, 0.1), observer=vizualizer.record)

    vizualizer.plot()