import matplotlib.pyplot as plot


class Visualizer:
    def __init__(self):
        self.trials = 0
        self.action_space_size = 0
        self.steps = 0
        self.algorithms = {}  # algo_name -> {total_rewards, action_counts}
        self.current_algo = None

    def start(self, trials: int, action_space_size: int, steps: int):
        self.trials = trials
        self.action_space_size = action_space_size
        self.steps = steps
        self.algorithms = {}
        self.current_algo = None

    def begin_algorithm(self, name: str):
        self.current_algo = name
        self.algorithms[name] = {
            "total_rewards": [0.0] * self.steps,
            "action_counts": [0] * self.action_space_size,
        }

    def record(self, step: int, action: int, reward: float):
        self.algorithms[self.current_algo]["total_rewards"][step] += reward
        self.algorithms[self.current_algo]["action_counts"][action] += 1

    def plot(self):
        if not self.algorithms:
            raise RuntimeError("No data to plot; run trials first.")

        # Plot average rewards for each algorithm on the same graph
        plot.figure(figsize=(10, 5))
        for algo_name, data in self.algorithms.items():
            avg_rewards = [r / self.trials for r in data["total_rewards"]]
            plot.plot(avg_rewards, label=algo_name, alpha=0.8)
        
        plot.xlabel("Step")
        plot.ylabel("Average reward")
        plot.title("Algorithm comparison: Average reward per step")
        plot.legend()
        plot.grid(True)
        plot.tight_layout()
        plot.show()
